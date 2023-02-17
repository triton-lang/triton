#include "Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace mlir {
namespace {
using triton::DotOp;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::MmaEncodingAttr;
using triton::gpu::SharedEncodingAttr;
using triton::gpu::SliceEncodingAttr;

// Get the wpt for MMAv1 using more information.
// Reference the original logic here
// https://github.com/openai/triton/blob/0e4691e6dd91e001a8d33b71badf8b3314325459/lib/codegen/analysis/layout.cc#L223
SmallVector<unsigned> getWarpsPerCTA(ArrayRef<int64_t> shape, bool isARow,
                                     bool isBRow, bool isAVec4, bool isBVec4,
                                     int numWarps) {
  // TODO[Superjomn]: Share code with
  // DotOpMmaV1ConversionHelper::AParam/BParam, since same code to compute the
  // rep,spw and fpw.
  SmallVector<unsigned> wpt({1, 1});
  SmallVector<unsigned> wpt_nm1;

  SmallVector<int, 2> rep(2), spw(2);
  std::array<int, 3> fpw{{2, 2, 1}};
  int packSize0 = (isARow || isAVec4) ? 1 : 2;
  rep[0] = 2 * packSize0;
  spw[0] = fpw[0] * 4 * rep[0];

  int packSize1 = (isBRow && !isBVec4) ? 2 : 1;
  rep[1] = 2 * packSize1;
  spw[1] = fpw[1] * 4 * rep[1];

  do {
    wpt_nm1 = wpt;
    if (wpt[0] * wpt[1] < numWarps)
      wpt[0] = std::clamp<int>(wpt[0] * 2, 1, shape[0] / spw[0]);
    if (wpt[0] * wpt[1] < numWarps)
      wpt[1] = std::clamp<int>(wpt[1] * 2, 1, shape[1] / spw[1]);
  } while (wpt_nm1 != wpt);

  return wpt;
}

// Given a (potentially malformed) DotOp, determines the optimal
// MMAEncoding to use on V100
LogicalResult getOptimizedV100MMaLayout(triton::DotOp dotOp,
                                        MmaEncodingAttr &old,
                                        MmaEncodingAttr &ret) {
  auto *ctx = dotOp->getContext();
  auto AT = dotOp.a().getType().cast<RankedTensorType>();
  auto BT = dotOp.b().getType().cast<RankedTensorType>();
  auto DT = dotOp.d().getType().cast<RankedTensorType>();
  auto shapeA = AT.getShape();
  auto shapeB = BT.getShape();
  if (!DT.getEncoding())
    return mlir::failure();
  auto mmaLayout = DT.getEncoding().dyn_cast<MmaEncodingAttr>();
  if (!(mmaLayout && mmaLayout.isVolta()))
    return mlir::failure();
  // We have an MmaEncodingAttr here. Find the correct layout for it.
  auto dotOperandA = AT.getEncoding().cast<DotOperandEncodingAttr>();
  auto dotOperandB = BT.getEncoding().cast<DotOperandEncodingAttr>();
  bool isARow = dotOperandA.getIsMMAv1Row().cast<BoolAttr>().getValue();
  bool isBRow = dotOperandB.getIsMMAv1Row().cast<BoolAttr>().getValue();
  auto [isARow_, isBRow_, isAVec4_, isBVec4_, mmaId] =
      mmaLayout.decodeVoltaLayoutStates();
  bool isAVec4 = !isARow && (shapeA[isARow] <= 16);
  bool isBVec4 = isBRow && (shapeB[isBRow] <= 16);
  // The wpt of MMAv1 is also determined by isARow, isBRow and shape, and it
  // could only be set here for those states might be updated by previous
  // patterns in the Combine Pass.
  auto tgtWpt = getWarpsPerCTA(DT.getShape(), isARow, isBRow, isAVec4, isBVec4,
                               product(mmaLayout.getWarpsPerCTA()));
  if (isARow == isARow_ && isBRow == isBRow_ && isAVec4 == isAVec4_ &&
      isBVec4 == isBVec4_) {
    if (tgtWpt == mmaLayout.getWarpsPerCTA())
      return mlir::failure();
  }
  // Recalculate the wpt, for here we could get the latest information, the
  // wpt should be updated.
  auto updatedWpt =
      getWarpsPerCTA(DT.getShape(), isARow, isBRow, isAVec4, isBVec4,
                     product(mmaLayout.getWarpsPerCTA()));
  // return results
  old = mmaLayout;
  ret =
      MmaEncodingAttr::get(ctx, mmaLayout.getVersionMajor(), updatedWpt,
                           AT.getShape(), BT.getShape(), isARow, isBRow, mmaId);
  return mlir::success();
}

// Replace result op type
void setOpResultType(Operation *op, ArrayRef<Type> newTypes) {
  if (op->getNumResults() != newTypes.size())
    llvm_unreachable("number of types different from number of results");
  // nothing to do
  if (op->getNumResults() == 0)
    return;
  // replace types
  for (unsigned i = 0; i < op->getNumResults(); i++) {
    Type newType = newTypes[i];
    op->getResult(i).setType(newType);
  }
  // special case: arith.constant: we need to change the value attr
  if (isa<arith::ConstantOp>(op)) {
    Type newType = newTypes[0];
    auto attr = op->getAttrDictionary()
                    .get("value")
                    .dyn_cast<mlir::DenseElementsAttr>();
    if (attr) {
      auto newAttr =
          mlir::DenseElementsAttr::getFromRawBuffer(newType, attr.getRawData());
      op->setAttr("value", newAttr);
    }
  }
}

// update style type given the provided layoutMap
Type updateStaleType(
    const DenseMap<MmaEncodingAttr, MmaEncodingAttr> &layoutMap,
    RankedTensorType type) {
  auto encoding = type.getEncoding();
  // mma encoding
  if (auto mma = encoding.dyn_cast<MmaEncodingAttr>()) {
    auto newMma = layoutMap.lookup(mma);
    if (!newMma)
      return Type();
    return RankedTensorType::get(type.getShape(), type.getElementType(),
                                 newMma);
  }
  // slice encoding
  else if (auto slice = encoding.dyn_cast<SliceEncodingAttr>()) {
    if (auto mma = slice.getParent().dyn_cast<MmaEncodingAttr>()) {
      auto newMma = layoutMap.lookup(mma);
      if (!newMma)
        return Type();
      auto newSlice =
          SliceEncodingAttr::get(slice.getContext(), slice.getDim(), newMma);
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   newSlice);
    }
  }
  // dot operand encoding
  else if (auto dotOp = encoding.dyn_cast<DotOperandEncodingAttr>()) {
    if (auto mma = dotOp.getParent().dyn_cast<MmaEncodingAttr>()) {
      auto newMma = layoutMap.lookup(mma);
      if (!newMma)
        return Type();
      auto newDotOp = DotOperandEncodingAttr::get(
          dotOp.getContext(), dotOp.getOpIdx(), newMma, dotOp.getIsMMAv1Row());
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   newDotOp);
    }
  }
  return Type();
}

} // namespace

class UpdateMmaForVoltaPass
    : public UpdateMmaForVoltaBase<UpdateMmaForVoltaPass> {
public:
  UpdateMmaForVoltaPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    // Step 1:
    // Build a map from old MMA encoding to new MMA encoding.
    DenseMap<MmaEncodingAttr, MmaEncodingAttr> layoutMap;
    m.walk([&layoutMap](triton::DotOp dotOp) {
      MmaEncodingAttr newLayout;
      MmaEncodingAttr oldLayout;
      if (failed(getOptimizedV100MMaLayout(dotOp, oldLayout, newLayout)))
        return;
      layoutMap[oldLayout] = newLayout;
    });
    // Step 2:
    // Replace all wrong layouts with the right one
    m.walk([&layoutMap](Operation *op) {
      if (op->getNumResults() != 1)
        return;
      auto type = op->getResult(0).getType().dyn_cast<RankedTensorType>();
      if (!type)
        return;
      Type newType = updateStaleType(layoutMap, type);
      if (!newType)
        return;
      setOpResultType(op, {newType});
    });
    // Step 3:
    // We may have messed up some loops in the process.
    // Fix them up
    if (fixupLoops(m).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createTritonGPUUpdateMmaForVoltaPass() {
  return std::make_unique<UpdateMmaForVoltaPass>();
}

} // namespace mlir
