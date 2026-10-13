#include "TritonAMDGPUToLLVM/Passes.h"

#include "AsyncUtility.h"
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
    membarPass.run();
  }
};

} // namespace
} // namespace mlir::triton
