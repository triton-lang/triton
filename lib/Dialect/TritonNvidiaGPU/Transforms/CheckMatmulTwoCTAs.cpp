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
    Operation *firstTcGenOp = nullptr;
    bool global2CTA = false;
    unsigned numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);

    WalkResult result = mod.walk([&](Operation *op) {
      bool op2CTA = false;
      if (auto mmaOp = dyn_cast<ttng::TCGen5MMAOp>(op))
        op2CTA = mmaOp.getTwoCtas();
      else if (isa<ttng::TMEMCopyOp>(op))
        op2CTA = (numCTAs == 2);
      else
        return WalkResult::advance();

      if (!firstTcGenOp) {
        firstTcGenOp = op;
        global2CTA = op2CTA;
        return WalkResult::advance();
      }
      if (op2CTA != global2CTA) {
        auto diag = op->emitError()
                    << "inconsistent CTA mode between tcgen05 operations; "
                       "this op uses " << (op2CTA ? "2" : "1") << " CTA mode:\n"
                    << *op;
        diag.attachNote(firstTcGenOp->getLoc())
            << "but first tcgen05 op uses " << (global2CTA ? "2" : "1") << " CTA mode:\n"
            << *firstTcGenOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    mod->setAttr(AttrTwoCTAsName, BoolAttr::get(mod.getContext(), global2CTA));
  }
};

} // namespace

} // namespace mlir::triton::nvidia_gpu
