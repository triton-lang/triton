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
bool hasOnlySourceAliases(const AllocationSlice &lhs,
                          const AllocationSlice &rhs, Allocation &allocation) {
  auto *before = lhs.physicalFootprint;
  auto *after = rhs.physicalFootprint;
  if (!before || !after)
    return false;
  // A kernel-entry allocation has one lifetime across all calls.
  auto kernel = cast<FunctionOpInterface>(allocation.getOperation());
  bool local = lhs.getBufferId() != Allocation::InvalidBufferId &&
               rhs.getBufferId() != Allocation::InvalidBufferId;
  bool overlaps = false;
  for (const auto &a : before->regionInfo.views) {
    for (const auto &b : after->regionInfo.views) {
      if (a.allocationFrame != b.allocationFrame)
        return false;
      if (!a.region.intersects(b.region))
        continue;
      if (!a.allocation || a.allocation != b.allocation ||
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
  Operation *before = source.op;
  Operation *after = sink.op;
  // Init and inval execute synchronously on the same lane within a region.
  if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp>(before) &&
      isa<ttng::InitBarrierOp, ttng::InvalBarrierOp>(after))
    return true;

  auto completesTransactions = [](Operation *op) {
    return isa<ttng::AsyncLoadOpInterface, ttng::CLCTryCancelOp,
               ttng::AsyncSharedStoreOp>(op);
  };
  // Accessing these live destinations requires completion, which makes their
  // writes visible to the generic proxy.
  if (completesTransactions(before) && !source.isBarrier &&
      hasOnlySourceAliases(source.slice, sink.slice, allocation))
    return true;

  if (source.isBarrier && sink.isBarrier) {
    auto signalsCompletion = [&](Operation *op) {
      if (auto arrive = dyn_cast<ttng::AsyncCopyMbarrierArriveOp>(op))
        return arrive.getNoIncrement();
      return completesTransactions(op) ||
             isa<ttng::ArriveBarrierOp, ttng::BarrierExpectOp,
                 ttng::TCGen5CommitOp, ttng::MMAv5OpInterface>(op);
    };
    // Signaling completion via incrementing or decrementing
    // expect / arrive counters commutes with each other
    // Note that this is just for mbarriers
    // Data accessed by these ops may still add barriers
    if (signalsCompletion(before) && signalsCompletion(after))
      return true;

    // A wait can observe a live counter concurrently with signals or waits.
    if (isa<ttng::WaitBarrierOp>(after))
      return true;

    auto wait = dyn_cast<ttng::WaitBarrierOp>(before);
    auto contribution = cast<gpu::MBarrierOpInterface>(after);
    // Thread ordering covers direct contributions to the same counter. A
    // scalar issuer needs a CTA rendezvous. With per-warp issuers, each warp
    // finishes its wait before contributing to the next phase. Broadcast
    // waits and nonidentity fromCTA routing omit participating CTAs.
    if (wait && wait.getAlloc() == contribution.getBarrier() &&
        !ttng::hasCGABroadcast(wait.getAlloc().getType())) {
      auto expect = dyn_cast<ttng::BarrierExpectOp>(after);
      auto arrive = dyn_cast<ttng::ArriveBarrierOp>(after);
      if ((expect && !expect.getFromCTA()) || (arrive && !arrive.getFromCTA()))
        return true;
    }
  }

  // Identical same-width commutative atomics can be freely reordered.
  auto beforeAtomic = dyn_cast<triton::gpu::LocalAtomicScatterRMWOp>(before);
  auto afterAtomic = dyn_cast<triton::gpu::LocalAtomicScatterRMWOp>(after);
  return beforeAtomic && afterAtomic && beforeAtomic.isCommutative() &&
         afterAtomic.isCommutative() &&
         beforeAtomic.getAtomicRmwOp() == afterAtomic.getAtomicRmwOp() &&
         beforeAtomic.getDst().getType().getElementType() ==
             afterAtomic.getDst().getType().getElementType();
}

} // namespace mlir::triton
