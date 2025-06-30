#ifndef TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ALLOCATE_UTILITY_H_
#define TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ALLOCATE_UTILITY_H_

#include "mlir/IR/BuiltinOps.h"
#include "triton/Analysis/Allocation.h"

namespace mlir::triton::gpu {

void fillAllocationInfo(ModuleOp mod, ModuleAllocation &allocation);

} // namespace mlir::triton::gpu

#endif // TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ALLOCATE_UTILITY_H_
