#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Membar/CFGBuilder.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {

void MembarAnalysis::syncIfNeeded(Operation *op, const BlockInfo &effects,
                                  MembarInfo *membarInfo, OpBuilder *builder,
                                  bool cluster) {
  if (!builder)
    return;
  auto &pending = membarInfo->pending;
  if (!BlockInfo::requiresThreadSync(pending, effects) &&
      !pending.isIntersected(effects, filter, &allocation, sliceFilter))
    return;
  // The barrier clears incoming state. The operation's own effects still
  // follow it, including a scratch write that conflicts with a pending read.
  builder->setInsertionPoint(op);
  insertBarrier(op, builder, cluster);
  membarInfo->sync();
}

void MembarAnalysis::insertBarrier(Operation *op, OpBuilder *builder,
                                   bool cluster) {
  OpBuilder::InsertionGuard g(*builder);
  if (cluster)
    ttng::ClusterBarrierOp::create(*builder, op->getLoc());
  else
    triton::gpu::BarrierOp::create(*builder, op->getLoc(),
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
  mb::FixedSync fixed;
  shared.describe(op, model, fixed, triton::getMemoryAccesses(op),
                  clusterStages);
  // The flow visits guarded operations separately. Join pending completions
  // and peer-visible waits before the lanes split.
  if (!cluster && isa<cf::CondBranchOp, scf::IfOp>(op) &&
      op->getOperand(0).getDefiningOp<triton::gpu::ThreadPredicateOp>()) {
    model.sharedIn.threadDemands.kind = ThreadDemandKind::General;
    model.sharedIn.threadDemands.mayNotifyPeer = true;
  }
  // Model a leading barrier before handling the operation's effects.
  applyFixed(fixed.before);
  syncIfNeeded(op, model.sharedIn, membarInfo, builder, cluster);
  if (builder && model.needsWarp && !membarInfo->warpsSynced) {
    builder->setInsertionPoint(op);
    triton::gpu::BarrierOp::create(*builder, op->getLoc(),
                                   triton::gpu::AddrSpace::Local,
                                   triton::gpu::BarrierScope::Warp);
    membarInfo->syncWarps();
  }
  if (fixed.inside) {
    // Preserve effects reachable from entry before the scratch rendezvous.
    if (!membarInfo->allPathsFromEntrySynced)
      membarInfo->entryBlockInfo.join(model.sharedIn);
    applyFixed(fixed.inside);
  }
  // Even if a barrier was inserted, retain the operation's outgoing effects.
  membarInfo->addBlockInfo(model.sharedOut);
  // Model a trailing barrier after handling the operation's effects.
  applyFixed(fixed.after);
}

void ModuleMembarAnalysis::run() { run<MembarAnalysis>(); }

} // namespace mlir
