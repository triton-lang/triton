#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_

#include "mlir/IR/Operation.h"
#include "triton/Analysis/Allocation.h"

namespace mlir::triton::AMD {

// Filter function used in the AMDGPU backend to suppress unnecessary barriers
// during membar analysis.  Two independent filters are applied:
//
// 1) Async-write / local_load filter
//    Suppresses barriers between an async write (any op carrying
//    MemAsyncLocalWriteOpTrait) and a local_load on the same shared-memory
//    allocation, provided the local_load's token chains back to an async_wait.
//
//    Membar analysis conservatively assumes a hazard because it cannot
//    distinguish dynamic offsets (sub-views) within the same allocation.
//    In pipelined loops, the producer and consumer operate on different buffer
//    slots, so the RAW/WAR hazard in the same iteration is a false positive.
//    The token chain to async_wait serves as a proxy for pipeliner-generated
//    stages with multi-buffering:
//      - RAW: the local_load reads data already made visible by a prior wait.
//      - WAR: the token proves the load belongs to a pipeline stage that uses
//             a different buffer slot from the subsequent async write.
//
// 2) LDS barrier op filter
//    Suppresses barriers between pairs of LDS memory barrier operations
//    (init_barrier, arrive_barrier, async_copy_mbarrier_arrive, wait_barrier)
//    which manage their own synchronization.
bool membarFilter(Operation *op1, Operation *op2, bool op1IsRead,
                  bool op2IsRead, Allocation *allocation);
} // namespace mlir::triton::AMD

#endif
