#ifndef TRITON_METAL_TO_LLVM_PASSES_H
#define TRITON_METAL_TO_LLVM_PASSES_H

#include "mlir/Pass/Pass.h"
#include <cstdint>
#include <memory>

namespace mlir::triton::metal {

/// Convert TritonGPU dialect to LLVM dialect with Metal-specific target
/// information (Apple Silicon GPU). Uses MetalTargetInfo for SIMD shuffles,
/// threadgroup barriers, and simdgroup_matrix operations.
std::unique_ptr<Pass> createConvertTritonMetalToLLVMPass(int32_t gpuFamily);

/// Allocate threadgroup (shared) memory for Metal targets. Runs the standard
/// ModuleAllocation analysis and attaches size/offset attributes.
std::unique_ptr<Pass> createAllocateSharedMemoryMetalPass(int32_t gpuFamily);

} // namespace mlir::triton::metal

#endif // TRITON_METAL_TO_LLVM_PASSES_H
