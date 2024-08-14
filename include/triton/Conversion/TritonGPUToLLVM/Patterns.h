#ifndef TRITONGPU_CONVERSION_TRITONGPUTOLLVM_PATTERNS_H
#define TRITONGPU_CONVERSION_TRITONGPUTOLLVM_PATTERNS_H

#include <functional>

namespace mlir {
class ModuleOp;
class RankedTensorType;

namespace triton::gpu {

/// Replaces `blocked -> dot_op` with `blocked -> shared -> dot_op` in the given
/// |module| op because the codegen doesn't handle `blocked -> dot_op` directly.
void decomposeBlockedToDotLayoutConversion(ModuleOp module);

/// Replaces `splat -> shared` with `splat -> blocked -> shared` in the given
/// |module| op.
void decomposeSplatOpToSharedLayoutConversion(ModuleOp module);

namespace {
static void addAttrs(Operation *op, ArrayRef<mlir::NamedAttribute> attrs) {
  for (const NamedAttribute attr : attrs)
    op->setAttr(attr.getName(), attr.getValue());
}
} // namespace

/// Replaces `mma/mfma -> dot_op` with `mma/mfma -> blocked -> dot_op` in the
/// given |module| op, but bypass the decomposition if |shortcutFn| returns
/// true.
using ShortcutFn = std::function<bool(RankedTensorType, RankedTensorType)>;
template <typename TensorCoreEncodingAttr>
void decomposeTensorCoreToDotLayoutConversion(ModuleOp module,
                                              ShortcutFn shortcutFn) {
  int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(module);
  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(module);
  int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(module);

  module.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
    OpBuilder builder(cvtOp);
    auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
    auto dstType = cast<RankedTensorType>(cvtOp.getType());
    auto srcMma = dyn_cast<TensorCoreEncodingAttr>(srcType.getEncoding());
    auto dstDotOp =
        dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
    if (srcMma && dstDotOp && !shortcutFn(srcType, dstType)) {
      auto tmpType = RankedTensorType::get(
          dstType.getShape(), dstType.getElementType(),
          triton::gpu::BlockedEncodingAttr::get(
              module.getContext(), srcType.getShape(), getSizePerThread(srcMma),
              getOrder(srcMma), numWarps, threadsPerWarp, numCTAs));
      auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
          cvtOp.getLoc(), tmpType, cvtOp.getSrc());
      addAttrs(tmp, cvtOp->getAttrs());
      auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
          cvtOp.getLoc(), dstType, tmp);
      addAttrs(newConvert, cvtOp->getAttrs());
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    }
  });
}

} // namespace triton::gpu

} // namespace mlir

#endif
