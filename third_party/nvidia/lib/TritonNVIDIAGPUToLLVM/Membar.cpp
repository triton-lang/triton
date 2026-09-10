#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Utility.h"

#include "Allocation.h"
#include "TargetInfo.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

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
    ttng::runClusterBarrierInsertion(allocation, computeCapability, *regions);
    if (failed(ttng::runCrossCTAMBarrierInitSyncInsertion(allocation,
                                                          computeCapability))) {
      signalPassFailure();
      return;
    }
    ttng::optimizeMBarrierArrivals(mod, *regions, computeCapability);

    ModuleMembarAnalysis membarPass(allocation, NVIDIA::canSkipBarSync);
    membarPass.runAnalysis<MembarAnalysis>(*regions);
  }
};

} // namespace

// Preserve aliases introduced by allocator reuse. Multiple possible origins
// are safe when every physical overlap has the same allocation origin.
static bool hasOnlySourceAliases(const AllocationSlice &lhs,
                                 const AllocationSlice &rhs) {
  auto *before = lhs.physicalFootprint;
  auto *after = rhs.physicalFootprint;
  if (!before || !after)
    return false;
  for (const auto &a : before->regionInfo.views) {
    assert(isa_and_nonnull<gpu::LocalAllocOp>(a.allocation) &&
           "shared descriptor footprints must have allocation origins");
    for (const auto &b : after->regionInfo.views) {
      assert(a.allocationFrame && a.allocationFrame == b.allocationFrame &&
             "shared accesses must use the current function frame");
      if (a.region.intersects(b.region) && a.allocation != b.allocation)
        return false;
    }
  }
  return true;
}

bool NVIDIA::canSkipBarSync(Operation *before, Operation *after,
                            bool /*beforeIsRead*/, bool /*afterIsRead*/,
                            Allocation * /*allocation*/,
                            const AllocationSlice &beforeSlice,
                            const AllocationSlice &afterSlice) {
  auto completesTransactions = [](Operation *op) {
    return isa<ttng::TMALoadLikeOpInterface, ttng::CLCTryCancelOp,
               ttng::AsyncSharedStoreOp>(op);
  };
  // Accessing these live destinations requires completion, which makes their
  // writes visible to the generic proxy.
  if (completesTransactions(before) &&
      beforeSlice.sharedKind != gpu::SharedKind::Barrier &&
      hasOnlySourceAliases(beforeSlice, afterSlice))
    return true;

  // If before and after are mbarriers
  if (beforeSlice.sharedKind == gpu::SharedKind::Barrier &&
      afterSlice.sharedKind == gpu::SharedKind::Barrier) {
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
