#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_

#include "mlir/IR/Operation.h"

namespace mlir::triton::AMD {
// Should be used to filter barriers during Membar Analysis.
// There is one filter applied:
// 1) Do not create barriers if one operand is a LocalLoad synced by an
// AsyncWait. This prevents a redundant barrier because AsyncWait will already
// create a barrier. (see amdgpu_membar.mlir for examples).
// This fitler can produce incorrect assembly for the following IR:
//   %tile_a = ttg.subview
//   %1 = AsyncCopyGlobalToLocal %ptr %tile_a
//   scf.for
//     %2 = AsyncWait %1
//     %3 = LocalLoad %tile_a
//     %4 = AsyncCopy %ptr_2 %tile_a
//     scf.yield
// Because there will be no barrier between %3 and %4 but they read/write to
// the same location in shared memory. However, the pipeliner does always use at
// least 2 buffers so this IR cannot be produced.
bool membarFilter(Operation *op1, Operation *op2);
} // namespace mlir::triton::AMD

#endif
