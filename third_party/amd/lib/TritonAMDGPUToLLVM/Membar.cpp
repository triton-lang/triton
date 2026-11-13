#include "TritonAMDGPUToLLVM/Passes.h"

#include "AsyncUtility.h"
#include "ScheduleMembar.h"
#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "third_party/amd/include/Analysis/AMDGPUAllocation.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/MembarPlacement.h"
#include "triton/Analysis/Utility.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONAMDGPUMEMBAR
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace mlir::triton {
namespace {

class AMDMembarAnalysis : public MembarAnalysis {
public:
  AMDMembarAnalysis(Allocation &allocation, MembarFilterFn filter,
                    BufferRegionAnalysis &regions)
      : MembarAnalysis(allocation, std::move(filter), regions, nullptr,
                       AccessMode::AllSharedAccesses, getCompletionSync) {}

  // Consume the earlier wait-count pass's scope proof before merging state.
  static CompletionSync getCompletionSync(Operation *op,
                                          CompletionSync fallback) {
    if (isa<gpu::AsyncWaitOp, amdgpu::AsyncWaitOp, amdgpu::AsyncTDMWait,
            amdgpu::AsyncTDMIntrinsicWait>(op)) {
      if (auto scope = op->getAttrOfType<StringAttr>(
              amdgpu::kAsyncCompletionScopeAttrName))
        return scope.getValue() == "shared" ? CompletionSync::Shared
                                            : CompletionSync::All;
    }
    // Preserve the default classification for IR that has not gone through
    // UpdateAsyncWaitCount, including standalone Membar analysis tests.
    return fallback;
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

    auto solver = createDataFlowSolver();
    auto *regions = solver->load<BufferRegionAnalysis>(
        BufferRegionAnalysis::Mode::AllMemory, &allocation);
    if (failed(solver->initializeAndRun(mod)))
      llvm::report_fatal_error("failed to analyze allocated buffer regions");

    // Phase-shifted waves rendezvous at different static pipeline stages.
    if (mod.walk([](amdgpu::CondBarrierOp) { return WalkResult::interrupt(); })
            .wasInterrupted()) {
      ModuleMembarAnalysis(allocation, AMD::membarFilter)
          .runAnalysis<AMDMembarAnalysis>(*regions);
    } else if (failed(membar::placeBarriers(
                   allocation, *regions, AMD::membarFilter,
                   membar::PlacementMode::Shared,
                   AMDMembarAnalysis::getCompletionSync))) {
      return signalPassFailure();
    }
    // Reschedule local rendezvous while waits and memory effects are explicit,
    // before ConSan and LLVM lowering.
    AMD::scheduleMembarBarriers(mod);
  }
};

} // namespace
} // namespace mlir::triton
