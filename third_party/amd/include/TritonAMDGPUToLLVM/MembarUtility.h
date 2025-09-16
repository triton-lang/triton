#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_

#include "mlir/IR/Operation.h"

namespace mlir::triton::AMD {

// Filter function used in the AMDGPU backend to filter unnecessary barriers
// during Membar Analysis. Filters applied by this function:
// 1) Do not create barriers between AsyncCopyGlobalToLocal and LocalLoad if the
// LocalLoad is synced by AsyncWait. This prevents a redundant barrier between
// LocalLoad and prefetches because membar cannot see that subviews from the
// same shared allocation do not alias when pipelining loads. See
// amdgpu_membar.mlir for examples. This filter can produce wrong IR/assembly if
// we pipeline with a single buffer in lds because it filters out a required
// gpu.barrier between the LocalLoad and the prefetches. However the pipeliner
// will always use at least 2 buffers so this IR cannot be produced. Example
// membar input IR to produce incorrect results:
//   %tile_a = ttg.memdesc_index
//   %1 = AsyncCopyGlobalToLocal %ptr %tile_a
//   scf.for
//     %2 = AsyncWait %1
//      # Membar will add a required gpu.barrier here
//     %3 = LocalLoad %tile_a
//      # Requires gpu.barrier but filter will prevent it
//     %4 = AsyncCopyGlobalToLocal %ptr_2 %tile_a
//     scf.yield
bool membarFilter(Operation *op1, Operation *op2);
} // namespace mlir::triton::AMD

#endif
