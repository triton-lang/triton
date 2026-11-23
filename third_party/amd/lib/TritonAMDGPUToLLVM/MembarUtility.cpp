#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::AMD {

using membar::OrderingRule;

membar::AsyncGroup SMemOrdering::getAsyncGroup(Operation *op) const {
  using membar::AsyncGroup;
  if (isa<amdgpu::BufferLoadToLocalOp, amdgpu::AsyncCopyLocalToGlobalOp>(op))
    return {AsyncGroup::Issue, AsyncGroup::Copy};
  // Keep the original deadline when a queue also signals an LDS counter.
  if (isa<amdgpu::AsyncCopyMbarrierArriveOp>(op))
    return {AsyncGroup::Opaque, AsyncGroup::Copy};
  if (isa<amdgpu::TDMOpInterface, amdgpu::AsyncTDMFusedCopyGlobalToLocalOp>(
          op)) {
    if (auto counter = dyn_cast<gpu::MBarrierOpInterface>(op);
        counter && counter.getBarrier())
      return {AsyncGroup::Opaque, AsyncGroup::TDM};
    return {AsyncGroup::Issue, AsyncGroup::TDM};
  }
  // A single transfer can lower to several instructions. Only a zero
  // hardware count proves that every prior transfer on this queue completed.
  if (auto wait = dyn_cast<amdgpu::AsyncWaitOp>(op);
      wait && wait.getNumInst() == 0)
    return {AsyncGroup::Drain, AsyncGroup::Copy};
  if (auto wait = dyn_cast<amdgpu::AsyncTDMIntrinsicWait>(op);
      wait && wait.getCount() == 0)
    return {AsyncGroup::Drain, AsyncGroup::TDM};
  return membar::SMemOrdering::getAsyncGroup(op);
}

void SMemOrdering::classify(membar::SMemAccess &access, Operation *op,
                            const MemoryAccess &, membar::ThreadEffects,
                            Allocation &allocation) {
  auto counter = dyn_cast<gpu::MBarrierOpInterface>(op);
  bool ordered =
      isa<amdgpu::InitBarrierOp, amdgpu::ArriveBarrierOp,
          amdgpu::AsyncCopyMbarrierArriveOp, amdgpu::WaitBarrierOp>(op) ||
      (counter && counter.getBarrier());
  Operation *unannotated = nullptr;
  if (op->hasTrait<OpTrait::GlobalToLocalCopyTrait>())
    access.rule = ordered ? OrderingRule::OrderedCopy : OrderingRule::AsyncCopy;
  else if (auto load = dyn_cast<gpu::LocalLoadOp>(op)) {
    auto annotation = getAsyncWaitAnnotation(load);
    if (!annotation)
      unannotated = load;
    else if (annotation.getValue())
      access.rule = OrderingRule::OrderedLoad;
    else
      return;
  } else {
    if (ordered)
      access.rule = OrderingRule::OrderedBarrier;
    return;
  }

  // A fused copy's token covers every destination.
  Buffers ids;
  for (const auto &payload : getMemoryAccesses(op, gpu::SharedKind::Generic))
    llvm::append_range(ids,
                       allocation.getAllBufferIdsWithAliases(payload.value));
  llvm::sort(ids);
  ids.erase(llvm::unique(ids), ids.end());
  access.key = payloads.insert({&allocation, std::move(ids), unannotated});
}

bool SMemOrdering::canSkip(const membar::SMemAccess &source,
                           const membar::SMemAccess &sink,
                           Allocation &allocation, bool) const {
  auto isCopy = [](OrderingRule rule) {
    return rule == OrderingRule::AsyncCopy || rule == OrderingRule::OrderedCopy;
  };
  auto usesCounter = [](OrderingRule rule) {
    return rule == OrderingRule::OrderedBarrier ||
           rule == OrderingRule::OrderedCopy;
  };
  if ((isCopy(source.rule) && isCopy(sink.rule)) ||
      (usesCounter(source.rule) && usesCounter(sink.rule)))
    return true;

  // A token orders a load after a copy. A refill must still wait for readers.
  // Besides copies, only ordered or unannotated loads have payload keys.
  if (!isCopy(source.rule) || !sink.key)
    return false;
  auto &[copyFrame, dests, unused] = payloads[source.key];
  auto &[loadFrame, srcs, unannotated] = payloads[sink.key];
  if (copyFrame != &allocation || loadFrame != &allocation ||
      !llvm::any_of(dests,
                    [&](auto id) { return llvm::is_contained(srcs, id); }))
    return false;
  if (unannotated)
    reportMissingAsyncWaitAnnotation(unannotated);
  return sink.rule == OrderingRule::OrderedLoad;
}

} // namespace mlir::triton::AMD
