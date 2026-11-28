#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Utility.h"

#include "Allocation.h"
#include "TargetInfo.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Membar/MBarrierArrivals.h"
#include "triton/Analysis/Membar/Pass.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONNVIDIAGPUMEMBAR
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace mlir::triton {
namespace {

namespace ttng = mlir::triton::nvidia_gpu;

struct TritonNvidiaGPUMembar
    : public impl::TritonNvidiaGPUMembarBase<TritonNvidiaGPUMembar> {
  using TritonNvidiaGPUMembarBase::TritonNvidiaGPUMembarBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    NVIDIA::TargetInfo targetInfo(computeCapability, ptxVersion);
    ModuleAllocation allocation(
        mod, ttng::getNvidiaAllocationAnalysisScratchSizeFn(targetInfo));
    auto solver = createDataFlowSolver();
    auto *regions = solver->load<triton::BufferRegionAnalysis>(
        triton::BufferRegionAnalysis::Mode::AllMemory, &allocation);
    if (failed(solver->initializeAndRun(mod))) {
      signalPassFailure();
      return;
    }

    // Synchronization insertion and arrival attributes preserve this geometry.
    if (failed(membar::placeClusterBarriers(allocation, *regions)))
      return signalPassFailure();
    if (failed(membar::placeBarriersAndArrivals(
            allocation, *regions, NVIDIA::canSkipBarSync, computeCapability)))
      signalPassFailure();
  }
};

// Preserve aliases introduced by allocator reuse. Multiple possible origins
// are safe when every physical overlap has the same allocation origin.
bool hasOnlySourceAliases(const membar::SMemAccess &lhs,
                          const membar::SMemAccess &rhs,
                          Allocation &allocation) {
  auto *before = lhs.footprint;
  auto *after = rhs.footprint;
  if (!before || !after)
    return false;
  // A kernel-entry allocation has one lifetime across all calls.
  auto kernel = cast<FunctionOpInterface>(allocation.getOperation());
  bool local = lhs.issuer->getParentOfType<FunctionOpInterface>() == kernel &&
               rhs.issuer->getParentOfType<FunctionOpInterface>() == kernel;
  bool overlaps = false;
  for (const auto &a : before->regionInfo.views) {
    for (const auto &b : after->regionInfo.views) {
      if (a.allocationFrame != b.allocationFrame)
        return false;
      if (!a.region.intersects(b.region))
        continue;
      if (!a.allocation || a.allocation != b.allocation ||
          a.allocation->getParentOfType<FunctionOpInterface>() != kernel ||
          (!local &&
           a.allocation->getBlock() != &kernel.getFunctionBody().front()))
        return false;
      overlaps = true;
    }
  }
  return overlaps;
}

} // namespace

bool NVIDIA::canSkipBarSync(const membar::SMemAccess &source,
                            const membar::SMemAccess &sink,
                            Allocation &allocation) {
  Operation *before = source.issuer;
  Operation *after = sink.issuer;
  // Init and inval execute synchronously on the same lane within a region.
  // They change the counter's lifetime and must still be ordered with waits.
  if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp>(before) &&
      isa<ttng::InitBarrierOp, ttng::InvalBarrierOp>(after))
    return true;

  auto completesTransactions = [](Operation *op) {
    return isa<ttng::AsyncLoadOpInterface, ttng::CLCTryCancelOp,
               ttng::AsyncSharedStoreOp>(op);
  };
  // Accessing these live destinations requires completion, which makes their
  // writes visible to the generic proxy.
  if (!source.isMBarrier && completesTransactions(before))
    return hasOnlySourceAliases(source, sink, allocation);

  if (source.isMBarrier && sink.isMBarrier) {
    auto signalsCompletion = [&](Operation *op) {
      if (auto arrive = dyn_cast<ttng::AsyncCopyMbarrierArriveOp>(op))
        return arrive.getNoIncrement();
      return completesTransactions(op) ||
             isa<ttng::ArriveBarrierOp, ttng::BarrierExpectOp,
                 ttng::TCGen5CommitOp, ttng::MMAv5OpInterface>(op);
    };
    // Contributions commute on a live counter. Check payloads separately.
    if (signalsCompletion(before) && signalsCompletion(after))
      return true;

    // A wait can observe a live counter concurrently with signals or waits.
    if (isa<ttng::WaitBarrierOp>(after))
      return true;

    auto wait = dyn_cast<ttng::WaitBarrierOp>(before);
    // Thread ordering covers direct contributions to the same counter. A
    // scalar issuer needs a CTA rendezvous. With per-warp issuers, each warp
    // finishes its wait before contributing to the next phase. Broadcast
    // waits and nonidentity fromCTA routing omit participating CTAs.
    if (!wait || ttng::hasCGABroadcast(wait.getAlloc().getType()))
      return false;
    auto contributesDirectly = [&](auto signal) {
      return signal && !signal.getFromCTA() &&
             signal.getAlloc() == wait.getAlloc();
    };
    return contributesDirectly(dyn_cast<ttng::BarrierExpectOp>(after)) ||
           contributesDirectly(dyn_cast<ttng::ArriveBarrierOp>(after));
  }

  // Identical commutative atomics can be reordered when their results
  // are unused.
  auto beforeAtomic = dyn_cast<triton::gpu::LocalAtomicScatterRMWOp>(before);
  auto afterAtomic = dyn_cast<triton::gpu::LocalAtomicScatterRMWOp>(after);
  return beforeAtomic && afterAtomic && beforeAtomic.isCommutative() &&
         afterAtomic.isCommutative() &&
         beforeAtomic.getAtomicRmwOp() == afterAtomic.getAtomicRmwOp() &&
         beforeAtomic.getDst().getType().getElementType() ==
             afterAtomic.getDst().getType().getElementType();
}

} // namespace mlir::triton
