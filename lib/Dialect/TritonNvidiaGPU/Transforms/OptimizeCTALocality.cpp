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

// AssignCTALayouts chooses preferred CTA layouts for Dot/Reduce ops and
// materializes the boundary with ttg.convert_layout. This pass looks for
// cross-CTA conversions that feed side-effecting users, such as stores, and
// moves those users to a layout in the same CTA group as the conversion source.
bool isCrossCTAConversion(ttg::ConvertLayoutOp convert) {
  auto srcTy = cast<RankedTensorType>(convert.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(convert.getType());
  LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
  auto kBlock = StringAttr::get(convert.getContext(), "block");
  return conversion.hasInDim(kBlock);
}

Attribute cloneWithCGALayout(Attribute layout, ttg::CGAEncodingAttr cgaLayout) {
  if (auto blockedLayout = dyn_cast<ttg::BlockedEncodingAttr>(layout)) {
    return ttg::BlockedEncodingAttr::get(
        layout.getContext(), blockedLayout.getSizePerThread(),
        blockedLayout.getThreadsPerWarp(), blockedLayout.getWarpsPerCTA(),
        blockedLayout.getOrder(), cgaLayout);
  }

  if (auto sliceLayout = dyn_cast<ttg::SliceEncodingAttr>(layout)) {
    Attribute parentLayout =
        cloneWithCGALayout(sliceLayout.getParent(), cgaLayout);
    return ttg::SliceEncodingAttr::get(
        layout.getContext(), sliceLayout.getDim(),
        cast<ttg::DistributedEncodingTrait>(parentLayout));
  }

  llvm::report_fatal_error("cloneWithCGALayout not implemented for layout");
}


Value convertValue(OpBuilder &builder, Location loc, Value value,
                   Attribute layout) {
  auto ty = cast<RankedTensorType>(value.getType());
  if (ty.getEncoding() == layout)
    return value;
  return ttg::ConvertLayoutOp::create(builder, loc,
                                      ty.cloneWithEncoding(layout), value);
}

// Convert the CGA layouts of tensor operands to ones the same as the
// conversion source:
//
//   %v1 = ttg.convert_layout %v0 : #planned -> #orig
//   tt.store %ptr_orig, %v1, %mask_orig : ... #orig
//
// becomes:
//
//   %v2 = ttg.convert_layout %v0 : #planned -> #target
//   %ptr_target = ttg.convert_layout %ptr_orig : #orig -> #target
//   %mask_target = ttg.convert_layout %mask_orig : #orig -> #target
//   tt.store %ptr_target, %v2, %mask_target : ... #target
//
// Will not insert conversions if any operands cannot be rematerialized in the
// target layout.
void rewriteUser(ttg::ConvertLayoutOp convert, OpOperand &use) {
  Operation *op = use.getOwner();
  if (!op->getResults().empty())
    return;

  auto srcTy = cast<RankedTensorType>(convert.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(convert.getType());

  auto getRootCGALayout = [](Attribute layout) -> ttg::CGAEncodingAttr {
    if (auto slice = dyn_cast<ttg::SliceEncodingAttr>(layout))
      return getRootCGALayout(slice.getParent());
    return ttg::getCGALayout(layout);
  };
  ttg::CGAEncodingAttr cgaLayout = getRootCGALayout(srcTy.getEncoding());
  Attribute targetLayout = cloneWithCGALayout(dstTy.getEncoding(), cgaLayout);

  for (OpOperand &operand : op->getOpOperands()) {
    if (&operand == &use)
      continue;
    if (!isa<RankedTensorType>(operand.get().getType()))
      continue;
    // Rewriting a user may require extra conversions on its other tensor
    // operands. Only do that when layout propagation can rematerialize the
    // producer slice in the target layout.
    llvm::SetVector<Value> slice;
    llvm::DenseMap<Value, Attribute> layouts;
    if (failed(getRematerializableSlice(operand, slice, targetLayout, layouts)))
      return;
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
}

void optimizeConvertLayout(ttg::ConvertLayoutOp convert) {
  if (!isCrossCTAConversion(convert))
    return;

  SmallVector<OpOperand *> uses;
  for (OpOperand &use : convert.getResult().getUses())
    uses.push_back(&use);

  for (OpOperand *use : uses)
    rewriteUser(convert, *use);
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

    for (ttg::ConvertLayoutOp convert : converts)
      optimizeConvertLayout(convert);
  }
};

} // namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
