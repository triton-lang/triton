#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Patterns.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DECOMPOSEUNSUPPORTEDNVIDIACONVERSIONS
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

static void addAttrs(Operation *op, ArrayRef<mlir::NamedAttribute> attrs) {
  for (const NamedAttribute attr : attrs)
    op->setAttr(attr.getName(), attr.getValue());
}

struct DecomposeUnsupportedConversions
    : public mlir::triton::impl::DecomposeUnsupportedNVIDIAConversionsBase<
          DecomposeUnsupportedConversions> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    triton::gpu::decomposeSplatOpToSharedLayoutConversion(mod);

    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    /* -------------------------------- */
    // Replace `mma -> dot_op` with `mma -> blocked -> dot_op`
    // unless certain conditions are met
    /* -------------------------------- */
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getSrc().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcMma =
          srcType.getEncoding().dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcMma && dstDotOp && !isMmaToDotShortcut(srcType, dstType)) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), srcType.getShape(), getSizePerThread(srcMma),
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

    triton::gpu::decomposeBlockedToDotLayoutConversion(mod);
  }
};

} // namespace

namespace mlir {

namespace triton {

namespace NVIDIA {

std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeUnsupportedConversionsPass() {
  return std::make_unique<DecomposeUnsupportedConversions>();
}

} // namespace NVIDIA

} // namespace triton

} // namespace mlir
