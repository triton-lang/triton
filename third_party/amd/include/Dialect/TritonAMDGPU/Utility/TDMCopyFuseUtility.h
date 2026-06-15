#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_TDMCOPYFUSEUTILITY_H
#define TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_TDMCOPYFUSEUTILITY_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton::AMD {

// Auto-fuse adjacent unhinted TDM copies into explicit
// `amdgpu.async_tdm_fused_copy_global_to_local` ops.
void autoFuseTDMCopies(ModuleOp mod);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_TDMCOPYFUSEUTILITY_H
