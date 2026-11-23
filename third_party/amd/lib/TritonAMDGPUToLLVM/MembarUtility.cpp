#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Membar/SmemModel.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {

bool membarFilter(const membar::SMemAccess &source,
                  const membar::SMemAccess &sink, Allocation &allocation) {
  // A workgroup barrier between async copies cannot order their completion.
  bool copy = source.issuer->hasTrait<OpTrait::GlobalToLocalCopyTrait>();
  if (copy && sink.issuer->hasTrait<OpTrait::GlobalToLocalCopyTrait>())
    return true;

  // A copy carrying an mbarrier arrives on it once complete, so the mbarrier
  // already orders it workgroup-wide. Copies without one are ordered by an
  // async wait, whose counter is per wave.
  auto usesCounter = [](Operation *op) {
    auto counter = dyn_cast<gpu::MBarrierOpInterface>(op);
    return counter && counter.getBarrier();
  };
  if (usesCounter(source.issuer) && usesCounter(sink.issuer))
    return true;

  // Filter only the RAW direction. A later refill of the same buffer
  // must still wait for the load to finish (WAR).
  auto load = dyn_cast<gpu::LocalLoadOp>(sink.issuer);
  if (!copy || !load)
    return false;

  // A fused copy's token covers every destination. Look up both sets of
  // aliases in the same allocation frame, including for call summaries.
  auto sources = allocation.getAllBufferIdsWithAliases(load.getSrc());
  for (const auto &payload :
       getMemoryAccesses(source.issuer, gpu::SharedKind::Generic))
    for (auto id : allocation.getAllBufferIdsWithAliases(payload.value))
      if (sources.contains(id))
        return isSyncedViaAsyncWait(load);
  return false;
}

} // namespace mlir::triton::AMD
