#include "TargetInfo.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace {

struct AllocateSharedMemoryMetal
    : public OperationPass<ModuleOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AllocateSharedMemoryMetal)

  AllocateSharedMemoryMetal() = default;
  explicit AllocateSharedMemoryMetal(int32_t gpuFamily)
      : gpuFamily(gpuFamily) {}

  StringRef getArgument() const override {
    return "allocate-shared-memory-metal";
  }

  StringRef getDescription() const override {
    return "Allocate threadgroup (shared) memory for Metal targets";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // Metal uses the standard allocation analysis — Apple Silicon has no
    // special scratch size requirements beyond the standard layout conversions.
    ModuleAllocation allocation(mod);
    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }

private:
  int32_t gpuFamily = 8;
};

} // namespace

namespace mlir::triton::metal {

std::unique_ptr<Pass> createAllocateSharedMemoryMetalPass(int32_t gpuFamily) {
  return std::make_unique<AllocateSharedMemoryMetal>(gpuFamily);
}

} // namespace mlir::triton::metal
