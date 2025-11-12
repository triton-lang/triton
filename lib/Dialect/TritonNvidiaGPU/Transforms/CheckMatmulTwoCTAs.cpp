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
    bool twoCTA = false;
    unsigned numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);

    // Walk all operations and check consistency across all tcgen05 ops
    WalkResult result = mod.walk([&](Operation *op) {
      std::optional<bool> currentTwoCTA;
      
      // Determine CTA mode for tcgen05 operations
      if (auto mmaOp = dyn_cast<ttng::TCGen5MMAOp>(op)) {
        currentTwoCTA = mmaOp.getTwoCtas();
      } else if (isa<ttng::TMEMCopyOp>(op)) {
        // For TMEMCopyOp, CTA mode is always determined by kernel launch numCTAs
        currentTwoCTA = (numCTAs == 2);
      } else {
        // Not a tcgen05 op, skip
        return WalkResult::advance();
      }
      
      // Check consistency across all tcgen05 ops
      if (!firstTcGenOp) {
        firstTcGenOp = op;
        twoCTA = *currentTwoCTA;
        return WalkResult::advance();
      }
      if (*currentTwoCTA != twoCTA) {
        auto diag = op->emitError()
                    << "inconsistent two_ctas setting across tcgen05 operations; "
                       "expected all tcgen05 ops to "
                    << (twoCTA ? "enable" : "disable") << " two_ctas.";
        diag.attachNote(firstTcGenOp->getLoc())
            << "first tcgen05 op here has two_ctas="
            << (twoCTA ? "true" : "false") << ".";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Set module attribute
    bool twoCTAValue = firstTcGenOp ? twoCTA : false;
    mod->setAttr(AttrTwoCTAsName, BoolAttr::get(mod.getContext(), twoCTAValue));
  }
};

} // namespace

} // namespace mlir::triton::nvidia_gpu
