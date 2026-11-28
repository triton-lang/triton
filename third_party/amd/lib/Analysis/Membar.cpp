#include "third_party/amd/include/Analysis/Membar.h"
#include "triton/Analysis/Membar/CFGBuilder.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"

namespace mlir::triton::AMD {

// Physical and relative addresses are part of the ordering of the access
// set. Rebuild both sets when translating storage or discarding SSA facts.
template <typename Transform>
static void transformAccesses(MembarInfo &info, Transform transform) {
  for (auto *block : {&info.pending, &info.entryBlockInfo}) {
    decltype(block->accesses) transformed;
    for (auto access : block->accesses) {
      transform(access);
      transformed.insert(access);
    }
    block->accesses = std::move(transformed);
  }
}

// A loop backedge or a new call can bind the same SSA value to a different
// slot or subslice. The physical origins and issuers still apply.
static void invalidateIterationInfo(MembarInfo &info) {
  transformAccesses(info,
                    [](membar::SMemAccess &access) { access.relative = {}; });
}

static void insertBarrier(Operation *op, OpBuilder &builder, MembarInfo &info) {
  builder.setInsertionPoint(op);
  triton::gpu::BarrierOp::create(builder, op->getLoc(),
                                 triton::gpu::AddrSpace::Local);
  info.sync();
}

void MembarAnalysis::updateSuccessor(Operation *terminator, Block *successor,
                                     MembarInfo *membarInfo) {
  if (shared.getBufferIndexAnalysis().isBackedgeSuccessor(terminator,
                                                          successor))
    invalidateIterationInfo(*membarInfo);
}

void MembarAnalysis::updateExitState(MembarInfo *membarInfo) {
  // Function summaries are reused at every call site, so per-function SSA
  // index identity is no longer meaningful.
  invalidateIterationInfo(*membarInfo);
}

void MembarAnalysis::update(Operation *op, MembarInfo *membarInfo,
                            FuncMapT *funcMap, OpBuilder *builder) {
  namespace mb = triton::membar;
  using mb::ThreadEffects;
  auto &pending = membarInfo->pending;
  auto syncIfNeeded = [&](const auto &effects) {
    auto joins = effects.threads.joins;
    // The flow visits guarded operations separately. Join pending completions
    // and peer-visible waits before the lanes split.
    if (isa<cf::CondBranchOp, scf::IfOp>(op) &&
        op->getOperand(0).getDefiningOp<triton::gpu::ThreadPredicateOp>())
      joins |= ThreadEffects::Completions | ThreadEffects::PeerWait;
    if ((pending.threads.pending & joins) ||
        pending.isIntersected(effects, filter, allocation))
      insertBarrier(op, *builder, *membarInfo);
  };
  // Finish completions that block every memory demand once before a loop.
  // Shared-copy completions can still overlap independent global reads.
  if (shared.getBufferIndexAnalysis().entersLoop(op) &&
      (pending.threads.pending & ThreadEffects::FullCompletion))
    insertBarrier(op, *builder, *membarInfo);

  // A call's allocation belongs to the callee and is translated separately.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    auto callee = cast<FunctionOpInterface>(call.resolveCallable());
    MembarInfo summary = funcMap->at(callee);
    transformAccesses(summary, [&](mb::SMemAccess &access) {
      access.footprint =
          regions.translateToCallsite(access.footprint, call, callee);
    });
    syncIfNeeded(summary.entryBlockInfo);
    membarInfo->applyCallSummary(summary);
    return;
  }

  mb::FixedSync fixed;
  auto data = shared.describe(op, fixed, triton::getMemoryAccesses(op));
  // Include the return deadline when a callee can return before its first
  // rendezvous. Callers must finish their pending completions on that path.
  if (op->hasTrait<OpTrait::ReturnLike>() &&
      isa<FunctionOpInterface>(op->getParentOp()))
    data.threads.joins |= ThreadEffects::Completions;
  // Model a leading barrier before handling the operation's effects.
  if (fixed.before == mb::fullBarrier)
    membarInfo->sync();
  syncIfNeeded(data);
  // Preserve effects reachable from entry before the first rendezvous,
  // including the write before an internal scratch barrier.
  if (!membarInfo->allPathsFromEntrySynced) {
    auto &prefix = membarInfo->entryBlockInfo;
    prefix.accesses.insert(data.accesses.begin(), data.accesses.end());
    prefix.threads.joins |= data.threads.joins;
  }
  if (fixed.inside == mb::fullBarrier)
    membarInfo->sync();
  // Even if a barrier was inserted, retain the operation's outgoing effects.
  // An internal scratch barrier leaves the final read pending.
  pending.threads.pending |= data.threads.pending;
  for (const auto &access : data.accesses)
    if (access.exit != mb::SMemAccess::None)
      pending.accesses.insert(access);
  // Model a trailing barrier after handling the operation's effects.
  if (fixed.after == mb::fullBarrier)
    membarInfo->sync();
}

} // namespace mlir::triton::AMD
