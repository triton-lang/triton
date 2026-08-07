#include "triton/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::nvidia_gpu {

bool hasTCGen5CommitCrossCTA(Operation *op) {
  SmallVector<Value> descs;
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op))
    descs = mma.getCompletionDescs();
  else if (auto commit = dyn_cast<ttng::TCGen5CommitOp>(op))
    llvm::append_range(descs, commit.getDescs());
  else
    return false;
  return !ttng::getCTABroadcastMasks(ttng::getModuleTwoCTAs(op), descs).empty();
}

bool requiresCrossCTAMBarrierInitSync(
    FunctionOpInterface funcOp, Value barrier, int numCTAs,
    llvm::function_ref<bool(Value)> aliasesBarrier) {
  // Barrier init sync is needed for barriers that are themselves cross-CTA,
  // and also for per-CTA barriers consumed by multi-CTA ops that multicast or
  // otherwise fan out barrier state across the cluster.
  auto barrierTy = dyn_cast<ttg::MemDescType>(barrier.getType());
  if (barrierTy && barrierTy.getShape()[0] != numCTAs)
    return true;

  // Or if it's used by a multi-CTA consumer that broadcasts barrier state
  // across CTAs even though the barrier allocation itself looks per-CTA.
  return funcOp
      ->walk<WalkOrder::PreOrder>([&](Operation *op) {
        SmallVector<Value> consumerBarriers;
        if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op)) {
          auto barrierOp = cast<ttg::MBarrierOpInterface>(op);
          if (hasTCGen5CommitCrossCTA(op))
            consumerBarriers.append(barrierOp.getBarriers());
        } else if (auto commit = dyn_cast<ttng::TCGen5CommitOp>(op)) {
          if (hasTCGen5CommitCrossCTA(op))
            consumerBarriers.push_back(commit.getBarrier());
        } else if (auto tma = dyn_cast<ttng::TMALoadLikeOpInterface>(op)) {
          if (tma.getMulticast())
            consumerBarriers.push_back(tma.getBarrier());
        } else if (auto clc = dyn_cast<ttng::CLCTryCancelOp>(op)) {
          consumerBarriers.push_back(clc.getMbarrier());
        } else if (auto store = dyn_cast<ttng::AsyncSharedStoreOp>(op)) {
          consumerBarriers.push_back(store.getMbarrier());
        } else if (auto expect = dyn_cast<ttng::BarrierExpectOp>(op);
                   expect && expect.getFromCTA()) {
          consumerBarriers.push_back(expect.getBarrier());
        } else if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op);
                   arrive && (arrive.isMulticast() || arrive.getFromCTA())) {
          consumerBarriers.push_back(arrive.getAlloc());
        }
        if (llvm::any_of(consumerBarriers, aliasesBarrier))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

} // namespace mlir::triton::nvidia_gpu
