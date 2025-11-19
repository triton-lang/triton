#ifndef TRITON_GPU_DIALECT_INTERFACES_H
#define TRITON_GPU_DIALECT_INTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// clang-format off
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/OpInterfaces.h.inc"
#include "triton/Dialect/TritonGPU/IR/AttrInterfaces.h.inc"
// clang-format on

namespace mlir::MemoryEffects {
// An atomic read or write on mbarrier:
// - atomic rmw:
//   * mbarrier.arrive
//   * mbarrier.expect_tx
//   * cp.async.bulk.tensor
// - atomic cas: mbarrier.try_wait
// We don'y need to insert a `__syncthreads()` between atomic effects, but we
// need if they were write effects.
struct MBarAtomic : public Effect::Base<MBarAtomic> {};
} // namespace mlir::MemoryEffects
#endif // TRITON_GPU_DIALECT_INTERFACES_H
