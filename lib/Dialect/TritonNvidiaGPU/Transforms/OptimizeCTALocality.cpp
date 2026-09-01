#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
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

// AssignCGALayouts chooses preferred CGA layouts for Dot/Reduce ops and
// materializes the boundary with ttg.convert_layout. This pass looks for
// cross-CTA conversions that feed stores and moves those stores to a layout
// in the same CTA group as the conversion source.
bool isCrossCTAConversion(ttg::ConvertLayoutOp convert) {
  auto srcTy = cast<RankedTensorType>(convert.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(convert.getType());
  LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
  auto kBlock = StringAttr::get(convert.getContext(), "block");
  return conversion.hasInDim(kBlock);
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
  // Other resultless users can constrain region or function signatures.
  if (!isa<triton::StoreOp, triton::DescriptorStoreLikeOpInterface>(op))
    return;

  auto srcTy = cast<RankedTensorType>(convert.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(convert.getType());

  if (ttg::isGenericLinearEncoding(srcTy.getEncoding()) ||
      ttg::isGenericLinearEncoding(dstTy.getEncoding()))
    return;
  auto cgaLayout =
      ttg::maybeLinearToCGAEncodingAttr(ttg::toLinearLayout(srcTy));
  if (failed(cgaLayout))
    return;
  auto maybeTargetLayout = cloneWithCGALayout(dstTy, *cgaLayout);
  if (failed(maybeTargetLayout))
    return;
  Attribute targetLayout = *maybeTargetLayout;

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

  if (convert.getResult().use_empty())
    convert.erase();
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
