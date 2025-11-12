#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUCHECKMATMULTWOCTAPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

class TritonNvidiaGPUCheckMatmulTwoCTAPass
    : public impl::TritonNvidiaGPUCheckMatmulTwoCTAPassBase<
          TritonNvidiaGPUCheckMatmulTwoCTAPass> {
public:
  using impl::TritonNvidiaGPUCheckMatmulTwoCTAPassBase<
      TritonNvidiaGPUCheckMatmulTwoCTAPass>::
      TritonNvidiaGPUCheckMatmulTwoCTAPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    Operation *firstMatmul = nullptr;
    bool firstTwoCTA = false;

    WalkResult result = mod.walk([&](ttng::TCGen5MMAOp op) {
      bool currentTwoCTA = op.getTwoCtas();
      if (!firstMatmul) {
        firstMatmul = op;
        firstTwoCTA = currentTwoCTA;
        return WalkResult::advance();
      }
      if (currentTwoCTA != firstTwoCTA) {
        auto diag = op.emitError()
                    << "inconsistent two_ctas setting across matmuls; "
                       "expected all matmuls to "
                    << (firstTwoCTA ? "enable" : "disable") << " two_ctas.";
        diag.attachNote(firstMatmul->getLoc())
            << "first matmul here has two_ctas="
            << (firstTwoCTA ? "true" : "false") << ".";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Also check TMEMCopyOp for two_ctas setting
    bool twoCTAFromCopy = false;
    mod.walk([&](ttng::TMEMCopyOp op) {
      auto dstTy = cast<mlir::triton::gpu::MemDescType>(op.getDst().getType());
      if (auto tmemEnc = dyn_cast<ttng::TensorMemoryEncodingAttr>(dstTy.getEncoding())) {
        if (tmemEnc.getTwoCTAs()) {
          twoCTAFromCopy = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    // Check if numCTAs==2 for any tmem copy ops (including TensorMemoryScalesLayout)
    bool hasTMemCopyWith2CTAs = false;
    unsigned numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    if (numCTAs == 2) {
      mod.walk([&](ttng::TMEMCopyOp op) {
        hasTMemCopyWith2CTAs = true;
        return WalkResult::interrupt();
      });
    }

    bool twoCTAValue = (firstMatmul && firstTwoCTA) || twoCTAFromCopy || hasTMemCopyWith2CTAs;
    mod->setAttr(AttrTwoCTAsName, BoolAttr::get(mod.getContext(), twoCTAValue));
  }
};

} // namespace

} // namespace mlir::triton::nvidia_gpu
