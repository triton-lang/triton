#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Utility.h"

#include "Allocation.h"
#include "TargetInfo.h"

#include "mlir/Pass/PassManager.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
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
    ttng::TritonNvidiaGPUOptimizeMBarrierArrivalsPassOptions arrivalOptions;
    arrivalOptions.computeCapability = computeCapability;
    mlir::PassManager arrivalPm(mod.getContext());
    arrivalPm.addPass(ttng::createTritonNvidiaGPUOptimizeMBarrierArrivalsPass(
        arrivalOptions));
    if (failed(arrivalPm.run(mod))) {
      signalPassFailure();
      return;
    }

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
