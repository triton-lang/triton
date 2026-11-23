#include "third_party/amd/include/Analysis/Membar.h"
#include "triton/Analysis/Membar/CFGBuilder.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"

namespace mlir::triton::AMD {

void BlockInfo::invalidateIterationInfo() {
  transformSlices([](AllocationSlice slice) {
    slice.invalidateIterationInfo();
    return slice;
  });
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
  namespace mb = triton::membar;
  using mb::ThreadEffects;
  auto &pending = membarInfo->pending;
  auto syncIfNeeded = [&](const auto &effects) {
    auto joins = effects.threads.joins;
    // The flow visits guarded operations separately. Finish completions
    // and peer-visible waits before the lanes split.
    if (isa<cf::CondBranchOp, scf::IfOp>(op) &&
        op->getOperand(0).getDefiningOp<triton::gpu::ThreadPredicateOp>())
      joins |= ThreadEffects::Completions | ThreadEffects::PeerWait;
    if ((pending.threads.pending & joins) ||
        pending.isIntersected(effects, ordering, allocation))
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
    summary.transformSlices([&](const AllocationSlice &slice) {
      return slice.translateToCallsite(call, callee, regions);
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
  if (fixed.before == mb::fullBarrier)
    membarInfo->sync();
  syncIfNeeded(data);
  // Callers check the entry side of accesses up to the first rendezvous.
  if (!membarInfo->allPathsFromEntrySynced) {
    auto &prefix = membarInfo->entryBlockInfo;
    prefix.accesses.insert(data.accesses.begin(), data.accesses.end());
    prefix.threads.merge({.joins = data.threads.joins},
                         !prefix.accesses.empty());
  }
  if (fixed.inside == mb::fullBarrier)
    membarInfo->sync();
  // Only effects exposed at exit remain pending after an internal barrier.
  pending.threads.pending |= data.threads.pending;
  for (const auto &access : data.accesses)
    if (access.exit != mb::SMemAccess::None)
      pending.accesses.insert(access);
  if (fixed.after == mb::fullBarrier)
    membarInfo->sync();
}

} // namespace mlir::triton::AMD
