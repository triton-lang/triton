#include "Analysis/AMDGPUAllocation.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"

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
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(mod, AMDAllocationAnalysisScratchSizeFn);

    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};

} // namespace
