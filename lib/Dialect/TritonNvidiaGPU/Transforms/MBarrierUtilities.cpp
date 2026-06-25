#include "triton/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::nvidia_gpu {

namespace {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

} // namespace

bool isCrossCTAMBarrier(Value barrier, int numCTAs) {
  auto barrierTy = dyn_cast<ttg::MemDescType>(barrier.getType());
  return barrierTy && barrierTy.getShape()[0] != numCTAs;
}

bool isCrossCTAConsumer(Operation *op,
                        llvm::function_ref<bool(Value)> aliasesBarrier) {
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op)) {
    auto barrierOp = cast<ttg::MBarrierOpInterface>(op);
    return mma.getTwoCtas() &&
           llvm::any_of(barrierOp.getBarriers(), aliasesBarrier);
  }
  if (auto commit = dyn_cast<ttng::TCGen5CommitOp>(op))
    return ttng::getModuleTwoCTAs(op) && aliasesBarrier(commit.getBarrier());
  if (auto tma = dyn_cast<ttng::TMALoadLikeOpInterface>(op))
    return tma.getMulticast() && aliasesBarrier(tma.getBarrier());
  if (auto clc = dyn_cast<ttng::CLCTryCancelOp>(op))
    return aliasesBarrier(clc.getMbarrier());
  return false;
}

bool isCrossCTAConsumer(Operation *op, Value barrier) {
  return isCrossCTAConsumer(op,
                            [&](Value value) { return value == barrier; });
}

bool requiresCrossCTAMBarrierInitSync(
    FunctionOpInterface funcOp, Value barrier, int numCTAs,
    llvm::function_ref<bool(Value)> aliasesBarrier) {
  // Barrier init sync is needed for barriers that are themselves cross-CTA,
  // and also for per-CTA barriers consumed by multi-CTA ops that multicast or
  // otherwise fan out barrier state across the cluster.
  if (isCrossCTAMBarrier(barrier, numCTAs))
    return true;

  return funcOp
      ->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (isCrossCTAConsumer(op, aliasesBarrier))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

bool requiresCrossCTAMBarrierInitSync(FunctionOpInterface funcOp,
                                      Value barrier, int numCTAs) {
  return requiresCrossCTAMBarrierInitSync(
      funcOp, barrier, numCTAs, [&](Value value) { return value == barrier; });
}

} // namespace mlir::triton::nvidia_gpu
