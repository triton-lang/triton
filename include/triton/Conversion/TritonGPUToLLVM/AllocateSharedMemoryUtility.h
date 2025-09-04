#ifndef TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ALLOCATE_UTILITY_H_
#define TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ALLOCATE_UTILITY_H_

#include "mlir/IR/BuiltinOps.h"
#include "triton/Analysis/Allocation.h"

namespace mlir::triton::gpu {

/// Attach shared memory related attributes to module and operations inside it.
/// This includes total shared memory consumption in module and shared memory
/// offsets of buffers associated with operations.
void attachAllocationSizeAndOffsetAttr(ModuleOp mod,
                                       ModuleAllocation &allocation);

} // namespace mlir::triton::gpu

#endif // TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ALLOCATE_UTILITY_H_
