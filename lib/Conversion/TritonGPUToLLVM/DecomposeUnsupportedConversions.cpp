#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DECOMPOSEUNSUPPORTEDCONVERSIONS
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// pass ws related named attrs.
static void addAttrs(Operation *op, ArrayRef<mlir::NamedAttribute> attrs) {
  for (const NamedAttribute attr : attrs)
    op->setAttr(attr.getName(), attr.getValue());
}

struct DecomposeUnsupportedConversions
    : public mlir::triton::impl::DecomposeUnsupportedConversionsBase<
          DecomposeUnsupportedConversions> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    /* ---------------- */
    // Convert Fp8E4B15
    /* ---------------- */
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      if (!getElementTypeOrSelf(cvtOp)
               .isa<mlir::Float8E4M3B11FNUZType, mlir::Float8E4M3FNType>())
        return;
      auto shape = cvtOp.getType().cast<RankedTensorType>().getShape();
      auto argEncoding =
          cvtOp.getSrc().getType().cast<RankedTensorType>().getEncoding();
      auto cvtEncoding = cvtOp.getType().cast<RankedTensorType>().getEncoding();
      if (argEncoding.isa<triton::gpu::DotOperandEncodingAttr>() ||
          cvtEncoding.isa<triton::gpu::DotOperandEncodingAttr>())
        return;
      auto F16Ty = builder.getF16Type();

      auto newArgType = RankedTensorType::get(shape, F16Ty, argEncoding);
      auto newCvtType = RankedTensorType::get(shape, F16Ty, cvtEncoding);
      auto newArg = builder.create<mlir::triton::FpToFpOp>(
          cvtOp.getLoc(), newArgType, cvtOp.getSrc());
      addAttrs(newArg, cvtOp->getAttrs());
      auto newCvt = builder.create<mlir::triton::gpu::ConvertLayoutOp>(
          cvtOp.getLoc(), newCvtType, newArg);
      addAttrs(newCvt, cvtOp->getAttrs());
      auto newRet = builder.create<mlir::triton::FpToFpOp>(
          cvtOp.getLoc(), cvtOp.getType(), newCvt.getResult());
      newRet.setRounding(
          triton::RoundingMode::RTNE); // Downcast requires rounding mode
      addAttrs(newRet, cvtOp->getAttrs());
      cvtOp.replaceAllUsesWith(newRet.getResult());
      cvtOp.erase();
    });
    /* -------------------------------- */
    // Replace `splat -> shared
    // with `splat -> blocked -> shared
    /* -------------------------------- */
    mod.walk([&](triton::SplatOp splatOp) -> void {
      auto dstType = splatOp.getType().cast<RankedTensorType>();
      auto shared =
          dstType.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
      if (shared) {
        OpBuilder builder(splatOp);
        SmallVector<unsigned, 4> sizePerThread(dstType.getRank(), 1);
        auto newType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), dstType.getShape(), sizePerThread,
                getOrder(shared), numWarps, threadsPerWarp, numCTAs));
        auto newSplat = builder.create<triton::SplatOp>(
            splatOp.getLoc(), newType, splatOp.getSrc());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            splatOp.getLoc(), dstType, newSplat.getResult());
        splatOp.replaceAllUsesWith(newConvert.getResult());
        splatOp.erase();
      }
    });
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
    /* -------------------------------- */
    // Replace `blocked -> dot_op` with `blocked -> shared -> dot_op`
    // because the codegen doesn't handle `blocked -> dot_op` directly
    /* -------------------------------- */
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getSrc().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcBlocked && dstDotOp) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::SharedEncodingAttr::get(
                mod.getContext(), dstDotOp, srcType.getShape(),
                srcBlocked.getOrder(), srcBlocked.getCTALayout(),
                srcType.getElementType()));
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
};

} // namespace

namespace mlir {

namespace triton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeUnsupportedConversionsPass() {
  return std::make_unique<DecomposeUnsupportedConversions>();
}

} // namespace gpu

} // namespace triton

} // namespace mlir
