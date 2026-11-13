#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Membar/TmemModel.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Interfaces/CallInterfaces.h"

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {

bool MembarAnalysis::requiresThreadSync(const BlockInfo &pending,
                                        const BlockInfo &effects) {
  const auto &sync = pending.threadSync;
  const auto &demands = effects.threadDemands;
  if (sync.effectIssuers.empty() || !demands.hasDemand)
    return false;
  // The maps also include implicit scratch and translated callee accesses.
  bool hasSharedAccesses =
      !effects.syncReadSlices.empty() || !effects.syncWriteSlices.empty();
  return sync.completion == CompletionSync::All ||
         (sync.completion == CompletionSync::Shared &&
          (hasSharedAccesses || demands.hasDemandBeyondGlobalReads)) ||
         (!demands.publicationIssuers.empty() &&
          !sync.effectIssuers.hasSameKnownIssuer(demands.publicationIssuers)) ||
         // A notification can let a peer advance a pending wait's phase.
         // Private counter updates are already ordered by the slice hazards.
         (sync.peerWaitNeedsSync && demands.mayNotifyPeer);
}

Operation *MembarAnalysis::syncIfNeeded(Operation *op, const BlockInfo &effects,
                                        MembarInfo *membarInfo,
                                        OpBuilder *builder, bool cluster) {
  if (!builder)
    return nullptr;
  auto &pending = membarInfo->pending;
  if (!requiresThreadSync(pending, effects) &&
      !pending.isIntersected(effects, filter, &allocation, sliceFilter))
    return nullptr;
  // The barrier clears incoming state. The operation's own effects still
  // follow it, including a scratch write that conflicts with a pending read.
  builder->setInsertionPoint(op);
  Operation *barrier = insertBarrier(op, builder, cluster);
  membarInfo->sync();
  return barrier;
}

Operation *MembarAnalysis::insertBarrier(Operation *op, OpBuilder *builder,
                                         bool cluster) {
  OpBuilder::InsertionGuard g(*builder);
  if (cluster)
    return ttng::ClusterBarrierOp::create(*builder, op->getLoc());
  return triton::gpu::BarrierOp::create(*builder, op->getLoc(),
                                        triton::gpu::AddrSpace::Local);
}

void MembarAnalysis::updateSuccessor(Operation *terminator, Block *successor,
                                     MembarInfo *membarInfo) {
  if (shared.getBufferIndexAnalysis().isBackedgeSuccessor(terminator,
                                                          successor)) {
    membarInfo->pending.invalidateIterationInfo();
    membarInfo->entryBlockInfo.invalidateIterationInfo();
  }
}

void MembarAnalysis::updateExitState(MembarInfo *membarInfo) {
  // Function summaries are reused at every call site, so per-function SSA
  // index identity is no longer meaningful.
  membarInfo->pending.invalidateIterationInfo();
  membarInfo->entryBlockInfo.invalidateIterationInfo();
}

void MembarAnalysis::update(Operation *op, MembarInfo *membarInfo,
                            FuncMapT *funcMap, OpBuilder *builder) {
  // Finish completions that block every memory demand once before a loop.
  // Shared-copy completions can still overlap independent global reads.
  if (builder && shared.getBufferIndexAnalysis().entersLoop(op) &&
      membarInfo->pending.threadSync.completion == CompletionSync::All) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    membarInfo->sync();
  }
  updateMemoryEffects(op, membarInfo, funcMap, builder);
}

void MembarAnalysis::updateMemoryEffects(
    Operation *op, MembarInfo *membarInfo, FuncMapT *funcMap,
    OpBuilder *builder, std::optional<triton::BarrierStages> clusterStages) {
  bool cluster = clusterStages.has_value();
  // A call's allocation belongs to the callee and is translated separately.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    auto callee = cast<FunctionOpInterface>(call.resolveCallable());
    auto summary =
        shared.translateCallSummary(call, callee, funcMap->at(callee));
    syncIfNeeded(op, summary.entryBlockInfo, membarInfo, builder, cluster);
    membarInfo->applyCallSummary(summary);
    return;
  }

  namespace mb = triton::membar;
  auto applyFixed = [&](unsigned mask) {
    if (mask & mb::bit(mb::Need::RegionOrder))
      membarInfo->sync();
    else if (mask & mb::bit(mb::Need::WarpOrder))
      membarInfo->syncWarps();
  };
  mb::OpModel model;
  shared.describe(op, model, triton::getMemoryAccesses(op), clusterStages);
  // Model a leading barrier before handling the operation's effects.
  applyFixed(model.fixedBefore);
  syncIfNeeded(op, model.sharedIn, membarInfo, builder, cluster);
  if (builder && model.needsWarp && !membarInfo->warpsSynced) {
    builder->setInsertionPoint(op);
    triton::gpu::BarrierOp::create(*builder, op->getLoc(),
                                   triton::gpu::AddrSpace::Local,
                                   triton::gpu::BarrierScope::Warp);
    membarInfo->syncWarps();
  }
  if (model.fixedInside) {
    // Preserve effects reachable from entry before the scratch rendezvous.
    if (!membarInfo->allPathsFromEntrySynced)
      membarInfo->entryBlockInfo.join(model.sharedIn);
    applyFixed(model.fixedInside);
  }
  // Even if a barrier was inserted, retain the operation's outgoing effects.
  membarInfo->addBlockInfo(model.sharedOut);
  // Model a trailing barrier after handling the operation's effects.
  applyFixed(model.fixedAfter);
}

void ModuleMembarAnalysis::run() { run<MembarAnalysis>(); }

} // namespace mlir
