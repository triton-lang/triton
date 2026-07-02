#ifndef TRITON_METAL_TO_LLVM_PASSES_H
#define TRITON_METAL_TO_LLVM_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::triton::metal {

// Create the pass that converts TritonGPU dialect to LLVM dialect
// with Metal-specific target information (Apple Silicon GPU).
std::unique_ptr<Pass> createConvertTritonMetalToLLVMPass(int32_t gpuFamily);

// Create the pass that allocates threadgroup (shared) memory for Metal.
std::unique_ptr<Pass> createAllocateSharedMemoryMetalPass(int32_t gpuFamily);

} // namespace mlir::triton::metal

#endif // TRITON_METAL_TO_LLVM_PASSES_H
