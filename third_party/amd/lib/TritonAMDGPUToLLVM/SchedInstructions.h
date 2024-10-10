#ifndef TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_SCHED_INSTRUCTIONS_H
#define TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_SCHED_INSTRUCTIONS_H

#include "mlir/IR/Types.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// The following functions are used to collect and set side-channel information
// during to LLVM conversion/lowering to facilitate instruction scheduling
// controls.
namespace mlir::triton {
void setNumGeneratedMMAs(DotOp op, size_t mmaCount, unsigned m, unsigned n,
                         unsigned k, Type elementType);
void setNumGeneratedGlobalLoads(triton::LoadOp op, size_t globalLoadsCount,
                                Type type);
void setNumGeneratedDsReads(gpu::LocalLoadOp op, size_t numDsReadsCount,
                            Type type);
void storeOpConversionCallback(triton::gpu::LocalStoreOp op, size_t llvmOpCount,
                               Type type);
} // namespace mlir::triton

#endif
