#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_TDMMERGEUTILITY_H
#define TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_TDMMERGEUTILITY_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton::AMD {

// Generate hints, compute groups, and replace each compatible run with one
// explicit `amdgpu.async_tdm_fused_copy_global_to_local` op.
void materializeTDMMergeGroups(ModuleOp mod);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_TDMMERGEUTILITY_H
