#include "TritonAMDGPUToLLVM/Passes.h"

#include "AsyncUtility.h"
#include "ScheduleMembar.h"
#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "third_party/amd/include/Analysis/AMDGPUAllocation.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONAMDGPUMEMBAR
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace mlir::triton {
namespace {

class AMDMembarAnalysis : public MembarAnalysis {
public:
  using MembarAnalysis::MembarAnalysis;

protected:
  // Consume the earlier wait-count pass's scope proof before merging state.
  CompletionSync getCompletionSync(Operation *op) override {
    if (isa<gpu::AsyncWaitOp, amdgpu::AsyncWaitOp, amdgpu::AsyncTDMWait,
            amdgpu::AsyncTDMIntrinsicWait>(op)) {
      if (auto scope = op->getAttrOfType<StringAttr>(
              amdgpu::kAsyncCompletionScopeAttrName))
        return scope.getValue() == "shared" ? CompletionSync::Shared
                                            : CompletionSync::All;
    }
    // Preserve the default classification for IR that has not gone through
    // UpdateAsyncWaitCount, including standalone Membar analysis tests.
    return MembarAnalysis::getCompletionSync(op);
  }
};

struct TritonAMDGPUMembar
    : public impl::TritonAMDGPUMembarBase<TritonAMDGPUMembar> {
  using TritonAMDGPUMembarBase::TritonAMDGPUMembarBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    AMD::TargetInfo targetInfo(gfxArch);
    if (targetInfo.getISAFamily() == amdgpu::ISAFamily::Unknown) {
      mod.emitError("unsupported target: '") << gfxArch << "'";
      signalPassFailure();
      return;
    }

    auto allocationFn = [&targetInfo](Operation *op) {
      return AMD::AMDAllocationAnalysisScratchSizeFn(op, targetInfo);
    };
    ModuleAllocation allocation(mod, allocationFn,
                                targetInfo.getSharedMemoryPartitionSize());

    if (targetInfo.requiresAliasInfoForAsyncOps())
      AMD::annotateLocalLoadsSyncedViaAsyncWait(mod);

    ModuleMembarAnalysis membarPass(allocation, AMD::membarFilter);
    membarPass.run<AMDMembarAnalysis>();
    // Reschedule local rendezvous while waits and memory effects are explicit,
    // before ConSan and LLVM lowering.
    AMD::scheduleMembarBarriers(mod);
  }
};

} // namespace
} // namespace mlir::triton
