#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_

#include "mlir/IR/Operation.h"
#include "triton/Analysis/Allocation.h"

namespace mlir::triton::AMD {

// Filter function used in the AMDGPU backend to filter unnecessary barriers
// during Membar Analysis. Membar calls the filter with the pending (earlier)
// access as op1 and the current access as op2; the filters below rely on that
// ordering to tell RAW from WAR. Filters applied by this function:
// 1) Do not create a barrier between an AsyncCopyGlobalToLocal and a LocalLoad
// that follows it into the same buffer, if the LocalLoad is synced by
// AsyncWait. Such a LocalLoad is ordered after the AsyncLoads its token waits
// for, so the barrier membar would add is redundant: membar cannot see that
// subviews from the same shared allocation do not alias when pipelining loads.
// See amdgpu_membar.mlir for examples.
// This filter is direction-aware and applies to the RAW direction only. The
// wait says nothing about what follows the read, so an AsyncCopyGlobalToLocal
// that refills the same buffer after a synced LocalLoad must still wait for
// every thread to finish reading it (WAR), and only a barrier orders that.
// Example membar input IR, where both barriers below are required and emitted:
//   %tile_a = ttg.memdesc_index
//   %1 = AsyncCopyGlobalToLocal %ptr %tile_a
//   scf.for
//     %2 = AsyncWait %1
//      # Membar will add a required ttg.barrier here
//     %3 = LocalLoad %tile_a
//      # WAR: membar will add the required ttg.barrier here
//     %4 = AsyncCopyGlobalToLocal %ptr_2 %tile_a
//     scf.yield
// 2) Do not create barriers between two async loads. The synchronization
// between them do not make sense.
bool membarFilter(Operation *op1, Operation *op2, bool op1IsRead,
                  bool op2IsRead, Allocation *allocation);
} // namespace mlir::triton::AMD

#endif
