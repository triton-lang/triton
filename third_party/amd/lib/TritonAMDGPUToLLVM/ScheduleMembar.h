#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_SCHEDULEMEMBAR_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_SCHEDULEMEMBAR_H_

#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton::AMD {

// Move eligible local barriers before global-load uses or loop-argument
// rotations that may require register copies.
void scheduleMembarBarriers(ModuleOp module);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_SCHEDULEMEMBAR_H_
