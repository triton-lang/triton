#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Utility.h"

#include "Allocation.h"
#include "TargetInfo.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"

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

    ttng::runClusterBarrierInsertion(allocation, computeCapability);
    if (failed(ttng::runCrossCTAMBarrierInitSyncInsertion(allocation,
                                                          computeCapability))) {
      signalPassFailure();
      return;
    }

    ModuleMembarAnalysis membarPass(allocation, NVIDIA::canSkipBarSync);
    membarPass.run();
  }
};

} // namespace

bool NVIDIA::canSkipBarSync(Operation *before, Operation *after,
                            bool /*beforeIsRead*/, bool /*afterIsRead*/,
                            Allocation * /*allocation*/) {
  // These mbarrier ops are single threaded, so are always synchronized wrt.
  // each other.
  if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::BarrierExpectOp>(
          before) &&
      isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::BarrierExpectOp>(
          after))
    return true;

  // Keep these exemptions local to an ordered pair on the same barrier;
  // do not suppress synchronization across execution regions or a backedge.
  if (auto expect = dyn_cast<ttng::BarrierExpectOp>(before)) {
    if (before->getBlock() == after->getBlock() &&
        before->isBeforeInBlock(after)) {
      // Both operations use partition-relative thread zero.
      if (auto copy = dyn_cast<ttng::AsyncBulkCopyGlobalToLocalOp>(after))
        if (expect.getAlloc() == copy.getBarrier())
          return true;
      // A matching linear bulk copy already orders the expectation and wait.
      // Keep the exemption scoped to that path so other asynchronous producers
      // retain their synchronization.
      if (auto wait = dyn_cast<ttng::WaitBarrierOp>(after)) {
        if (expect.getAlloc() == wait.getAlloc()) {
          for (Operation *op = before->getNextNode(); op != after;
               op = op->getNextNode()) {
            if (auto copy = dyn_cast<ttng::AsyncBulkCopyGlobalToLocalOp>(op))
              if (copy.getBarrier() == wait.getAlloc())
                return true;
          }
        }
      }
    }
  }

  // wait_barrier will never run ahead of the load it's waiting on
  if (isa<ttng::TMALoadLikeOpInterface, ttng::AsyncBulkCopyGlobalToLocalOp>(
          before) &&
      isa<ttng::WaitBarrierOp>(after))
    return true;

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
