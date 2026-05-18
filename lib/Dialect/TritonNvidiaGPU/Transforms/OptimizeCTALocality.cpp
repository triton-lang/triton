#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZECTALOCALITYPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

bool isCrossCTAConversion(ttg::ConvertLayoutOp convert) {
  auto srcTy = dyn_cast<RankedTensorType>(convert.getSrc().getType());
  auto dstTy = dyn_cast<RankedTensorType>(convert.getType());
  if (!srcTy || !dstTy || !srcTy.getEncoding() || !dstTy.getEncoding())
    return false;
  if (isa<ttg::DotOperandEncodingAttr>(srcTy.getEncoding()) ||
      isa<ttg::DotOperandEncodingAttr>(dstTy.getEncoding()))
    return false;

  LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
  auto kBlock = StringAttr::get(convert.getContext(), "block");
  return conversion.hasInDim(kBlock);
}

Attribute cloneWithCGALayout(Attribute layout, ttg::CGAEncodingAttr cgaLayout) {
  if (auto blockedLayout = dyn_cast<ttg::BlockedEncodingAttr>(layout)) {
    if (blockedLayout.getOrder().size() != cgaLayout.getRank())
      return {};
    return ttg::BlockedEncodingAttr::get(
        layout.getContext(), blockedLayout.getSizePerThread(),
        blockedLayout.getThreadsPerWarp(), blockedLayout.getWarpsPerCTA(),
        blockedLayout.getOrder(), cgaLayout);
  }

  if (auto sliceLayout = dyn_cast<ttg::SliceEncodingAttr>(layout)) {
    Attribute parentLayout =
        cloneWithCGALayout(sliceLayout.getParent(), cgaLayout);
    if (!parentLayout)
      return {};
    return ttg::SliceEncodingAttr::get(
        layout.getContext(), sliceLayout.getDim(),
        cast<ttg::DistributedEncodingTrait>(parentLayout));
  }

  return {};
}

ttg::CGAEncodingAttr getSourceCGALayoutForDestination(Attribute srcLayout,
                                                      Attribute dstLayout) {
  if (auto dstSlice = dyn_cast<ttg::SliceEncodingAttr>(dstLayout)) {
    auto srcSlice = dyn_cast<ttg::SliceEncodingAttr>(srcLayout);
    if (!srcSlice || srcSlice.getDim() != dstSlice.getDim())
      return {};
    return ttg::getCGALayout(srcSlice.getParent());
  }
  return ttg::getCGALayout(srcLayout);
}

bool canRematerializeConvert(OpOperand &operand, Attribute layout) {
  llvm::SetVector<Value> slice;
  llvm::DenseMap<Value, Attribute> layouts;
  if (failed(getConvertBackwardSlice(operand, slice, layout, layouts)))
    return false;

  return llvm::all_of(slice, [](Value value) {
    Operation *op = value.getDefiningOp();
    return !op || canBeRematerialized(op);
  });
}

Value convertValue(OpBuilder &builder, Location loc, Value value,
                   Attribute layout) {
  auto ty = cast<RankedTensorType>(value.getType());
  if (ty.getEncoding() == layout)
    return value;
  return ttg::ConvertLayoutOp::create(builder, loc,
                                      ty.cloneWithEncoding(layout), value);
}

bool rewriteUser(ttg::ConvertLayoutOp convert, OpOperand &use) {
  Operation *op = use.getOwner();
  if (!op->getResults().empty())
    return false;

  auto srcTy = cast<RankedTensorType>(convert.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(convert.getType());
  int64_t rank = srcTy.getRank();
  ttg::CGAEncodingAttr cgaLayout = getSourceCGALayoutForDestination(
      srcTy.getEncoding(), dstTy.getEncoding());
  if (!cgaLayout)
    return false;
  Attribute targetLayout = cloneWithCGALayout(dstTy.getEncoding(), cgaLayout);
  if (!targetLayout)
    return false;

  for (OpOperand &operand : op->getOpOperands()) {
    if (&operand == &use)
      continue;
    auto operandTy = dyn_cast<RankedTensorType>(operand.get().getType());
    if (!operandTy)
      continue;
    if (operandTy.getRank() != rank)
      return false;
    if (!canRematerializeConvert(operand, targetLayout))
      return false;
  }

  OpBuilder builder(op);
  Location loc = op->getLoc();
  for (OpOperand &operand : op->getOpOperands()) {
    if (&operand == &use) {
      operand.set(convertValue(builder, loc, convert.getSrc(), targetLayout));
    } else if (isa<RankedTensorType>(operand.get().getType())) {
      operand.set(convertValue(builder, loc, operand.get(), targetLayout));
    }
  }
  return true;
}

void optimizeConvertLayout(ttg::ConvertLayoutOp convert,
                           llvm::SetVector<Operation *> &opsToErase) {
  if (!isCrossCTAConversion(convert))
    return;

  SmallVector<OpOperand *> uses;
  for (OpOperand &use : convert.getResult().getUses())
    uses.push_back(&use);

  for (OpOperand *use : uses) {
    (void)rewriteUser(convert, *use);
  }

  if (convert->use_empty())
    opsToErase.insert(convert);
}

struct OptimizeCTALocalityPass
    : public impl::TritonNvidiaGPUOptimizeCTALocalityPassBase<
          OptimizeCTALocalityPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (ttg::TritonGPUDialect::getNumCTAs(mod) == 1)
      return;

    SmallVector<ttg::ConvertLayoutOp> converts;
    mod.walk(
        [&](ttg::ConvertLayoutOp convert) { converts.push_back(convert); });

    llvm::SetVector<Operation *> opsToErase;
    for (ttg::ConvertLayoutOp convert : converts)
      optimizeConvertLayout(convert, opsToErase);
    for (Operation *op : opsToErase)
      op->erase();
  }
};

} // namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
