#include "lib/Analysis/MembarCFG.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Interfaces/CallInterfaces.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUTMEMBARRIERINSERTIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {
using namespace membar;

class TMemSyncAnalysis : public MembarAnalysis {
public:
  TMemSyncAnalysis(Allocation &allocation, MembarFilterFn filter,
                   BufferRegionAnalysis &regions, TMemModel &tmem)
      : MembarAnalysis(allocation, std::move(filter), regions), tmem(tmem) {}

  void run(FunctionOpInterface function, FuncMapT &funcMap) {
    MembarAnalysis::run(function, funcMap);
    // Place waits after the fixed point so revisits can replace earlier plans.
    OpBuilder builder(function.getContext());
    for (auto &[op, kinds] : waitsBefore) {
      builder.setInsertionPoint(op);
      for (TMEMWaitKind kind : kinds)
        TMEMWaitOp::create(builder, op->getLoc(), kind);
    }
  }

private:
  static BlockInfo::SliceMapT &accesses(BlockInfo &info, TMEMWaitKind kind) {
    return kind == TMEMWaitKind::LOAD ? info.syncReadSlices
                                      : info.syncWriteSlices;
  }

  bool hasHazard(const BlockInfo &incomplete, const BlockInfo &effects,
                 TMEMWaitKind kind, bool needsBarrier = true) {
    bool isRead = kind == TMEMWaitKind::LOAD;
    const auto &slices =
        isRead ? incomplete.syncReadSlices : incomplete.syncWriteSlices;
    auto intersects = [&](const BlockInfo::SliceMapT &other, bool rhsIsRead) {
      return BlockInfo::isIntersected(slices, other, isRead, rhsIsRead,
                                      needsBarrier ? filter : MembarFilterFn{},
                                      nullptr, &allocation);
    };
    return (!isRead && intersects(effects.syncReadSlices, true)) ||
           intersects(effects.syncWriteSlices, false);
  }

  void waitBeforeBarrier(TMEMWaitKind kind) {
    auto &before = accesses(beforeBarrier, kind);
    if (before.empty())
      return;
    assert(lastBarrier && "completion needs a remembered rendezvous");
    waitsBefore[lastBarrier].push_back(kind);
    before.clear();
  }

  void waitNow(Operation *op, TMEMWaitKind kind, BlockInfo &pending) {
    if (accesses(beforeBarrier, kind).empty() &&
        accesses(afterBarrier, kind).empty())
      return;
    waitsBefore[op].push_back(kind);
    // A wait after lastBarrier restores its preceding accesses for publication.
    // Later accesses are already in pending.
    auto &before = accesses(beforeBarrier, kind);
    for (auto &[slice, ops] : before)
      accesses(pending, kind)[slice].insert(ops.begin(), ops.end());
    before.clear();
    accesses(afterBarrier, kind).clear();
  }

  void rememberBarrier(Operation *op, MembarInfo &info) {
    lastBarrier = op;
    info.sync();
    beforeBarrier.join(afterBarrier);
    afterBarrier.sync();
  }

  // Add waits before the saved barrier + now
  void flush(Operation *op, BlockInfo &pending) {
    for (TMEMWaitKind kind : {TMEMWaitKind::LOAD, TMEMWaitKind::STORE}) {
      waitBeforeBarrier(kind);
      waitNow(op, kind, pending);
    }
    lastBarrier = nullptr;
  }

  bool requiresThreadSync(const BlockInfo &pending,
                          const BlockInfo &effects) override {
    // Pending accesses require a rendezvous before publication.
    return effects.threadDemands.hasDemand &&
           (!pending.syncReadSlices.empty() ||
            !pending.syncWriteSlices.empty());
  }

  void update(Operation *op, MembarInfo *info, FuncMapT *funcMap,
              OpBuilder *builder) override {
    waitsBefore.erase(op);
    BlockInfo &pending = info->pending;

    if (auto barrier = dyn_cast<gpu::BarrierOp>(op);
        barrier && barrier.isWarp()) {
      // Complete each warp's accesses before its rendezvous, retaining
      // dependencies on other warps for later hazards and publications.
      flush(op, pending);
      return;
    }

    // Choose the barrier before placing waits.
    auto syncBefore = [&](const BlockInfo &effects) {
      if (Operation *barrier = syncIfNeeded(op, effects, info, builder))
        rememberBarrier(barrier, *info);
    };

    if (auto call = dyn_cast<CallOpInterface>(op)) {
      auto summary = getCallSummary(call, *funcMap);
      assert(summary && "expected callee summary");
      syncBefore(summary->entryBlockInfo);
      flush(op, pending);
      // TMEM addresses are assigned across the module; no frame offset.
      info->applyCallSummary(*summary);
      return;
    }

    OpModel model;
    tmem.describe(op, model, getMemoryAccesses(op));
    BlockInfo effects;
    for (const TMemAccess &access : model.tmemAccesses) {
      // Footprints preserve TMEM rows and columns; a full interval makes
      // unknown footprints alias all TMEM.
      AllocationSlice slice(Interval<size_t>{});
      slice.physicalFootprint = access.footprint;
      auto &slices =
          access.isWrite ? effects.syncWriteSlices : effects.syncReadSlices;
      slices[slice].insert(op);
    }
    if (model.boundary == Boundary::Publication)
      effects.threadDemands.hasDemand = true;
    bool hasScratch = allocation.getBufferId(op) != Allocation::InvalidBufferId;
    if (getLocalBarrierStages(op, hasScratch).hasBarrier()) {
      assert(effects.syncReadSlices.empty() &&
             effects.syncWriteSlices.empty() &&
             "barrier stages must not access tensor memory");
      rememberBarrier(op, *info);
    }

    syncBefore(effects);

    if (model.boundary != Boundary::None) {
      flush(op, pending);
    } else {
      for (TMEMWaitKind kind : {TMEMWaitKind::LOAD, TMEMWaitKind::STORE}) {
        if (hasHazard(beforeBarrier, effects, kind))
          waitBeforeBarrier(kind);
        // Same-warp hazards still need completion before the next access.
        if (hasHazard(beforeBarrier, effects, kind, /*needsBarrier=*/false) ||
            hasHazard(afterBarrier, effects, kind, /*needsBarrier=*/false))
          waitNow(op, kind, pending);
      }
    }

    if (!info->allPathsFromEntrySynced)
      info->entryBlockInfo.join(effects);

    if (model.tmemProducerNeed) {
      pending.join(effects);
      if (*model.tmemProducerNeed != Need::RegionOrder)
        afterBarrier.join(effects);
    }
  }

  TMemModel &tmem;

  // Planned tmem_wait insertions
  DenseMap<Operation *, SmallVector<TMEMWaitKind, 2>> waitsBefore;
  Operation *lastBarrier = nullptr;
  // Incomplete TMEM accesses before `lastBarrier`
  BlockInfo beforeBarrier;
  // Incomplete TMEM accesses after `lastBarrier`
  BlockInfo afterBarrier;
  // pending: Accesses that may still need cross-warp ordering
  // Invariant: afterBarrier \subset pending at the end of update
};

static LogicalResult runTMemAnalysis(ModuleOp mod) {
  auto solver = createDataFlowSolver();
  auto *regions = solver->load<BufferRegionAnalysis>(
      BufferRegionAnalysis::Mode::TensorMemoryOnly);
  if (failed(solver->initializeAndRun(mod)))
    return failure();
  TMemModel tmem(*regions);
  DenseMap<std::pair<Operation *, Operation *>, bool> sameWarps;
  auto filter = [&](Operation *lhs, Operation *rhs, bool /*lhsIsRead*/,
                    bool /*rhsIsRead*/, Allocation *allocation,
                    const AllocationSlice &, const AllocationSlice &) {
    // Function summaries and different WS regions may have different issuers.
    if (lhs->getParentRegion() != rhs->getParentRegion() ||
        lhs->getParentOfType<FunctionOpInterface>() !=
            allocation->getOperation())
      return false;
    auto [it, inserted] = sameWarps.try_emplace({lhs, rhs}, false);
    // A completion wait orders same-warp RAW, WAR, and WAW dependencies.
    if (inserted)
      it->second = tmem.isSameWarp(lhs, rhs);
    return it->second;
  };
  ModuleAllocation moduleAllocation(mod);
  TMemSyncAnalysis::runModule(
      moduleAllocation.getModuleOp(), [&](FunctionOpInterface function) {
        Allocation &allocation = *moduleAllocation.getFuncData(function);
        return TMemSyncAnalysis(allocation, filter, *regions, tmem);
      });
  return success();
}

} // namespace

struct TMemBarrierInsertionPass
    : public impl::TritonNvidiaGPUTMemBarrierInsertionPassBase<
          TMemBarrierInsertionPass> {
  using impl::TritonNvidiaGPUTMemBarrierInsertionPassBase<
      TMemBarrierInsertionPass>::TritonNvidiaGPUTMemBarrierInsertionPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!membar::usesTensorMemory(mod))
      return;
    if (failed(runTMemAnalysis(mod)))
      return signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
