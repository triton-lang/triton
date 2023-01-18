#include "Utility.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir {
namespace {
using triton::DotOp;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::MmaEncodingAttr;
using triton::gpu::SliceEncodingAttr;

// This pattern collects the wrong Mma those need to update and create the right
// ones for each.
// TODO[Superjomn]: RewirtePattern is not needed here, Rewrite this to a method
class CollectMmaToUpdateForVolta : public mlir::RewritePattern {
  // Holds the mapping from old(wrong) mmaEncodingAttr to the new(correct)
  // mmaEncodingAttr.
  DenseMap<MmaEncodingAttr, MmaEncodingAttr> &mmaToUpdate;

public:
  CollectMmaToUpdateForVolta(
      mlir::MLIRContext *ctx,
      DenseMap<MmaEncodingAttr, MmaEncodingAttr> &mmaToUpdate)
      : mlir::RewritePattern(triton::DotOp::getOperationName(), 1, ctx),
        mmaToUpdate(mmaToUpdate) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {

    auto dotOp = cast<triton::DotOp>(op);
    auto *ctx = dotOp->getContext();
    auto AT = dotOp.a().getType().cast<RankedTensorType>();
    auto BT = dotOp.b().getType().cast<RankedTensorType>();
    auto DT = dotOp.d().getType().cast<RankedTensorType>();
    if (!DT.getEncoding())
      return failure();
    auto mmaLayout = DT.getEncoding().dyn_cast<MmaEncodingAttr>();
    if (!(mmaLayout && mmaLayout.isVolta()))
      return failure();

    // Has processed.
    if (mmaToUpdate.count(mmaLayout))
      return failure();

    auto dotOperandA = AT.getEncoding().cast<DotOperandEncodingAttr>();
    auto dotOperandB = BT.getEncoding().cast<DotOperandEncodingAttr>();
    bool isARow = dotOperandA.getIsMMAv1Row().cast<BoolAttr>().getValue();
    bool isBRow = dotOperandB.getIsMMAv1Row().cast<BoolAttr>().getValue();
    auto [isARow_, isBRow_, isAVec4, isBVec4, mmaId] =
        mmaLayout.decodeVoltaLayoutStates();

    // The wpt of MMAv1 is also determined by isARow, isBRow and shape, and it
    // could only be set here for those states might be updated by previous
    // patterns in the Combine Pass.
    if (isARow_ == isARow && isBRow_ == isBRow) {
      auto tgtWpt =
          getWarpsPerCTA(DT.getShape(), isARow, isBRow, isAVec4, isBVec4,
                         product(mmaLayout.getWarpsPerCTA()));
      // Check if the wpt should be updated.
      if (tgtWpt == mmaLayout.getWarpsPerCTA() ||
          !MmaEncodingAttr::_mmaV1UpdateWpt)
        return failure();
    }

    MmaEncodingAttr newMmaLayout;
    {
      // Create a temporary MMA layout to obtain the isAVec4 and isBVec4
      auto tmpMmaLayout = MmaEncodingAttr::get(
          ctx, mmaLayout.getVersionMajor(), mmaLayout.getWarpsPerCTA(),
          AT.getShape(), BT.getShape(), isARow, isBRow, mmaId);
      auto [isARow_, isBRow_, isAVec4_, isBVec4_, _] =
          tmpMmaLayout.decodeVoltaLayoutStates();

      // Recalculate the wpt, for here we could get the latest information, the
      // wpt should be updated.
      auto updatedWpt =
          getWarpsPerCTA(DT.getShape(), isARow_, isBRow_, isAVec4_, isBVec4_,
                         product(mmaLayout.getWarpsPerCTA()));
      auto newWpt = MmaEncodingAttr::_mmaV1UpdateWpt
                        ? updatedWpt
                        : mmaLayout.getWarpsPerCTA();
      newMmaLayout = MmaEncodingAttr::get(ctx, mmaLayout.getVersionMajor(),
                                          newWpt, AT.getShape(), BT.getShape(),
                                          isARow, isBRow, mmaId);
    }

    // Collect the wrong MMA Layouts, and mark need to update.
    mmaToUpdate.try_emplace(mmaLayout, newMmaLayout);

    return failure();
  }

  // Get the wpt for MMAv1 using more information.
  // Reference the original logic here
  // https://github.com/openai/triton/blob/0e4691e6dd91e001a8d33b71badf8b3314325459/lib/codegen/analysis/layout.cc#L223
  SmallVector<unsigned, 2> getWarpsPerCTA(ArrayRef<int64_t> shape, bool isARow,
                                          bool isBRow, bool isAVec4,
                                          bool isBVec4, int numWarps) const {
    // TODO[Superjomn]: Share code with
    // DotOpMmaV1ConversionHelper::AParam/BParam, since same code to compute the
    // rep,spw and fpw.
    SmallVector<unsigned, 2> wpt({1, 1});
    SmallVector<unsigned, 2> wpt_nm1;

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
};

class UpdateMMAForMMAv1 : public mlir::RewritePattern {
  const DenseMap<MmaEncodingAttr, MmaEncodingAttr> &mmaToUpdate;

public:
  UpdateMMAForMMAv1(
      MLIRContext *context,
      const DenseMap<MmaEncodingAttr, MmaEncodingAttr> &mmaToUpdate)
      : RewritePattern(MatchAnyOpTypeTag{}, 1, context),
        mmaToUpdate(mmaToUpdate) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // Nothing to update
    if (mmaToUpdate.empty())
      return failure();

    if (auto dotOp = llvm::dyn_cast<DotOp>(op))
      return rewriteDotOp(op, rewriter);
    else if (auto cvtOp = llvm::dyn_cast<ConvertLayoutOp>(op))
      return rewriteCvtOp(op, rewriter);
    else if (auto expandDimsOp = llvm::dyn_cast<triton::ExpandDimsOp>(op))
      return rewriteExpandDimsOp(op, rewriter);
    else if (auto constOp = llvm::dyn_cast<arith::ConstantOp>(op))
      return rewriteConstantOp(op, rewriter);
    else
      return rewriteElementwiseOp(op, rewriter);
    return failure();
  }

  LogicalResult rewriteDotOp(Operation *op,
                             mlir::PatternRewriter &rewriter) const {
    auto dotOp = llvm::cast<DotOp>(op);
    auto tensorTy = dotOp->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!tensorTy)
      return failure();

    auto mma = dotOp.d()
                   .getType()
                   .cast<RankedTensorType>()
                   .getEncoding()
                   .dyn_cast<MmaEncodingAttr>();
    if (!mma || !mmaToUpdate.count(mma))
      return failure();

    auto newTensorTy = getUpdatedType(tensorTy);
    rewriter.replaceOpWithNewOp<DotOp>(op, newTensorTy, dotOp.a(), dotOp.b(),
                                       dotOp.c(), dotOp.allowTF32());
    return success();
  }

  LogicalResult rewriteCvtOp(Operation *op,
                             mlir::PatternRewriter &rewriter) const {
    auto cvt = llvm::cast<ConvertLayoutOp>(op);
    if (!needUpdate(cvt.getResult().getType()))
      return failure();
    auto tensorTy = cvt.result().getType().dyn_cast<RankedTensorType>();

    auto newTensorTy = getUpdatedType(tensorTy);
    auto newOp = rewriter.replaceOpWithNewOp<ConvertLayoutOp>(op, newTensorTy,
                                                              cvt.getOperand());
    return success();
  }

  LogicalResult rewriteExpandDimsOp(Operation *op,
                                    mlir::PatternRewriter &rewriter) const {
    auto expandDims = llvm::cast<triton::ExpandDimsOp>(op);
    auto srcTy = expandDims.src().getType();
    auto resTy = expandDims.getResult().getType();

    // the result type need to update
    if (!needUpdate(srcTy) && needUpdate(resTy)) {
      rewriter.replaceOpWithNewOp<triton::ExpandDimsOp>(op, expandDims.src(),
                                                        expandDims.axis());
      return success();
    }

    return failure();
  }

  LogicalResult rewriteConstantOp(Operation *op,
                                  mlir::PatternRewriter &rewriter) const {
    auto constant = llvm::cast<arith::ConstantOp>(op);
    auto resTy = constant.getResult().getType();
    if (!needUpdate(resTy))
      return failure();

    auto tensorTy = constant.getResult().getType().cast<RankedTensorType>();
    auto mma = tensorTy.getEncoding().dyn_cast<MmaEncodingAttr>();
    auto dot = tensorTy.getEncoding().dyn_cast<DotOperandEncodingAttr>();
    if (!mma && !dot)
      return failure();

    auto newTensorTy = getUpdatedType(tensorTy);
    if (auto attr = constant.getValue().dyn_cast<SplatElementsAttr>()) {
      auto newRet =
          SplatElementsAttr::get(newTensorTy, attr.getSplatValue<Attribute>());
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newRet);
      return success();
    }

    return failure();
  }

  LogicalResult rewriteElementwiseOp(Operation *op,
                                     mlir::PatternRewriter &rewriter) const {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();

    auto srcTy = op->getOperand(0).getType();
    auto resTy = op->getResult(0).getType();
    if (!needUpdate(srcTy) && needUpdate(resTy)) {
      op->getResult(0).setType(
          getUpdatedType(resTy.dyn_cast<RankedTensorType>()));
      return success();
    }
    return failure();
  }

  RankedTensorType getUpdatedType(RankedTensorType type) const {
    if (!needUpdate(type))
      return type;
    auto encoding = type.getEncoding();
    if (auto mma = encoding.dyn_cast<MmaEncodingAttr>()) {
      auto newMma = mmaToUpdate.lookup(mma);
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   newMma);
    } else if (auto slice = encoding.dyn_cast<SliceEncodingAttr>()) {
      if (auto mma = slice.getParent().dyn_cast<MmaEncodingAttr>()) {
        auto newMma = mmaToUpdate.lookup(mma);
        auto newSlice =
            SliceEncodingAttr::get(slice.getContext(), slice.getDim(), newMma);
        return RankedTensorType::get(type.getShape(), type.getElementType(),
                                     newSlice);
      }
    } else if (auto dotOp = encoding.dyn_cast<DotOperandEncodingAttr>()) {
      if (auto mma = dotOp.getParent().dyn_cast<MmaEncodingAttr>()) {
        auto newMma = mmaToUpdate.lookup(mma);
        auto newDotOp =
            DotOperandEncodingAttr::get(dotOp.getContext(), dotOp.getOpIdx(),
                                        newMma, dotOp.getIsMMAv1Row());
        return RankedTensorType::get(type.getShape(), type.getElementType(),
                                     newDotOp);
      }
    }
    return type;
  }

  // Tell if this type contains a wrong MMA encoding and need to update.
  bool needUpdate(Type type) const {
    auto tensorTy = type.dyn_cast<RankedTensorType>();
    if (!tensorTy)
      return false;
    return needUpdate(tensorTy);
  }

  // Tell if this type contains a wrong MMA encoding and need to update.
  bool needUpdate(RankedTensorType type) const {
    auto encoding = type.getEncoding();
    if (!encoding)
      return false;

    MmaEncodingAttr mma;
    if ((mma = encoding.dyn_cast<MmaEncodingAttr>())) {
    } else if (auto slice = encoding.dyn_cast<SliceEncodingAttr>()) {
      mma = slice.getParent().dyn_cast<MmaEncodingAttr>();
    } else if (auto dotOp = encoding.dyn_cast<DotOperandEncodingAttr>()) {
      mma = dotOp.getParent().dyn_cast<MmaEncodingAttr>();
    }

    return mma && mmaToUpdate.count(mma);
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class UpdateMmaForVoltaPass
    : public UpdateMmaForVoltaBase<UpdateMmaForVoltaPass> {
public:
  UpdateMmaForVoltaPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    llvm::DenseMap<MmaEncodingAttr, MmaEncodingAttr> mmaToUpdate;
    {
      mlir::RewritePatternSet patterns(context);
      patterns.add<CollectMmaToUpdateForVolta>(context, mmaToUpdate);

      GreedyRewriteConfig config;
      config.enableRegionSimplification =
          false; // The pattern doesn't modify the IR
      if (applyPatternsAndFoldGreedily(m, std::move(patterns), config).failed())
        signalPassFailure();
    }

    if (!mmaToUpdate.empty()) {
      mlir::RewritePatternSet patterns(context);
      patterns.add<UpdateMMAForMMAv1>(context, mmaToUpdate);

      mlir::GreedyRewriteConfig config;
      // Make sure the slice and dot_operand layouts' parent mma are updated
      // before updating DotOp or it will get a mismatch parent-encoding.
      config.useTopDownTraversal = true;

      if (applyPatternsAndFoldGreedily(m, std::move(patterns), config).failed())
        signalPassFailure();

      if (fixupLoops(m).failed())
        signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createTritonGPUUpdateMmaForVoltaPass() {
  return std::make_unique<UpdateMmaForVoltaPass>();
}

} // namespace mlir
