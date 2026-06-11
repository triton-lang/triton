#include "Analysis/AMDGPUAllocation.h"
#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::AMD;

namespace mlir::triton {
#define GEN_PASS_DEF_ALLOCATEAMDGPUSHAREDMEMORY
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

struct AllocateAMDGPUSharedMemory
    : public mlir::triton::impl::AllocateAMDGPUSharedMemoryBase<
          AllocateAMDGPUSharedMemory> {
  AllocateAMDGPUSharedMemory(StringRef arch) { this->arch = arch.str(); }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    std::string archGenerationName = this->arch;
    AMD::TargetInfo targetInfo(archGenerationName);
    // Get partition size from target info
    size_t partitionSize = targetInfo.getSharedMemoryPartitionSize();

    auto allocationFn = [&targetInfo](Operation *op) {
      return AMDAllocationAnalysisScratchSizeFn(op, targetInfo);
    };

    ModuleAllocation allocation(mod, allocationFn, partitionSize);

    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};

} // namespace

namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>>
createAllocateAMDGPUSharedMemoryPass(StringRef arch) {
  return std::make_unique<AllocateAMDGPUSharedMemory>(arch);
}
} // namespace mlir::triton
