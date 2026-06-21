#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUCHECKMATMULTWOCTAPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

ttg::MemDescType getOneCTABarrierType(ttg::MemDescType type, int numCTAs) {
  SmallVector<int64_t> shape(type.getShape());
  if (shape.empty())
    return type;
  shape.back() = numCTAs;

  MLIRContext *ctx = type.getContext();
  auto barrierCGALayout = ttg::CGAEncodingAttr::get1DLayout(ctx, numCTAs);
  auto barrierEncoding =
      ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, barrierCGALayout);
  return ttg::MemDescType::get(shape, type.getElementType(), barrierEncoding,
                               type.getMemorySpace(), type.getMutableMemory());
}

void normalizeOneCTABarrierTree(Value value, int numCTAs) {
  auto type = dyn_cast<ttg::MemDescType>(value.getType());
  if (!type)
    return;

  value.setType(getOneCTABarrierType(type, numCTAs));
  for (Operation *user : llvm::make_early_inc_range(value.getUsers())) {
    if (auto index = dyn_cast<ttg::MemDescIndexOp>(user))
      normalizeOneCTABarrierTree(index.getResult(), numCTAs);
  }
}

Value getBarrierRoot(Value barrier) {
  while (auto index = barrier.getDefiningOp<ttg::MemDescIndexOp>())
    barrier = index.getSrc();
  return barrier;
}

void normalizeOneCTATMABarrier(Value barrier, int numCTAs,
                               llvm::DenseSet<Value> &normalizedRoots) {
  Value root = getBarrierRoot(barrier);
  if (!normalizedRoots.insert(root).second)
    return;
  normalizeOneCTABarrierTree(root, numCTAs);
}

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

    // Walk all MMAv5 ops using the interface
    WalkResult result = mod.walk([&](ttng::MMAv5OpInterface op) -> WalkResult {
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

    // FPSAN rewrites all `tcgen05` MMA ops but sets the flag so it can be
    // propagated.
    if (!firstMatmul && mod->hasAttr(AttrTwoCTAsName))
      return;
    bool twoCTAValue = firstMatmul ? firstTwoCTA : false;
    mod->setAttr(AttrTwoCTAsName, BoolAttr::get(mod.getContext(), twoCTAValue));

    if (!twoCTAValue) {
      int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
      llvm::DenseSet<Value> normalizedBarrierRoots;
      mod.walk([&](Operation *op) {
        if (auto tmaLoad = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
          tmaLoad->removeAttr(tmaLoad.getMulticastAttrName());
          normalizeOneCTATMABarrier(tmaLoad.getBarrier(), numCTAs,
                                    normalizedBarrierRoots);
        }
        if (auto tmaGather = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
          tmaGather->removeAttr(tmaGather.getMulticastAttrName());
          normalizeOneCTATMABarrier(tmaGather.getBarrier(), numCTAs,
                                    normalizedBarrierRoots);
        }
      });
    }
  }
};

} // namespace

} // namespace mlir::triton::nvidia_gpu
