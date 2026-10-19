#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_SCHEDULEMEMBAR_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_SCHEDULEMEMBAR_H_

#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton::AMD {

// Move eligible local barriers before uses of global-load results.
void scheduleMembarBarriers(ModuleOp module);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_SCHEDULEMEMBAR_H_
