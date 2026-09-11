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
    ttng::prepareMBarrierArrivals(mod, *regions, computeCapability);

    ModuleMembarAnalysis membarPass(allocation, NVIDIA::canSkipBarSync);
    membarPass.runAnalysis<MembarAnalysis>(*regions);
  }
};

} // namespace

bool NVIDIA::canSkipBarSync(Operation *before, Operation *after,
                            bool /*beforeIsRead*/, bool /*afterIsRead*/,
                            Allocation * /*allocation*/) {
  if (isa<ttng::WaitBarrierOp>(after)) {
    // All threads must register incrementing arrivals before any can wait.
    if (auto arrive = dyn_cast<ttng::AsyncCopyMbarrierArriveOp>(before))
      return arrive.getNoIncrement();
    // Signals and waits can access the same live barrier concurrently;
    // accesses to distinct barriers are independent.
    if (isa<ttng::TMALoadLikeOpInterface, ttng::BarrierExpectOp,
            ttng::ArriveBarrierOp, ttng::TCGen5CommitOp>(before))
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
