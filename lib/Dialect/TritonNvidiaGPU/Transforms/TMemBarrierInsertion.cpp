#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseMap.h"

#include <optional>

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUTMEMBARRIERINSERTIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

enum class TMemAccessKind { None, Load, Store, MMA };

// Fine grain modeling of TMEM ops as pipelining behavior is not fully
// represented in ops attributes.
static bool isWritingAlloc(Operation *op) {
  auto alloc = dyn_cast<TMEMAllocOp>(op);
  return alloc && alloc.getSrc();
}

static bool isMMALikeOp(Operation *op) {
  return isa<TCGen5MMAOp, TCGen5MMAScaledOp, TMEMCopyOp>(op);
}

static TMemAccessKind getTMemAccessKind(Operation *op) {
  if (isa<TMEMLoadOp>(op))
    return TMemAccessKind::Load;
  if (isa<TMEMStoreOp>(op) || isWritingAlloc(op))
    return TMemAccessKind::Store;
  if (isMMALikeOp(op))
    return TMemAccessKind::MMA;
  return TMemAccessKind::None;
}

static bool filterFn(Operation *lhs, Operation *rhs, bool /*lhsIsRead*/,
                     bool /*rhsIsRead*/, Allocation * /*allocation*/) {
  TMemAccessKind lhsKind = getTMemAccessKind(lhs);
  TMemAccessKind rhsKind = getTMemAccessKind(rhs);

  bool war =
      lhsKind == TMemAccessKind::Load && rhsKind == TMemAccessKind::Store;
  bool raw =
      lhsKind == TMemAccessKind::Store && rhsKind == TMemAccessKind::Load;
  bool waw =
      lhsKind == TMemAccessKind::Store && rhsKind == TMemAccessKind::Store;

  // MMAv5 ops and tmem_copy are special cases, we care about load->mma and
  // store->mma dependencies but mma -> load/store doesn't require a barrier
  // since it would need a mbarrier wait that will ensure the op is finished
  // before any thread can reach the load/store.
  bool loadToMma =
      lhsKind == TMemAccessKind::Load && rhsKind == TMemAccessKind::MMA;
  bool storeToMma =
      lhsKind == TMemAccessKind::Store && rhsKind == TMemAccessKind::MMA;

  bool requiresBarrier = war || raw || waw || loadToMma || storeToMma;
  return !requiresBarrier;
}

static bool isTensorMemory(Value value) {
  auto memDescType = dyn_cast<gpu::MemDescType>(value.getType());
  return memDescType &&
         isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace());
}

static BlockInfo getTMemAccesses(Operation *op, BufferRegionAnalysis &regions) {
  BlockInfo accesses;
  for (const MemoryAccess &access : getMemoryAccesses(op)) {
    if (!isTensorMemory(access.value))
      continue;
    // Footprints preserve TMEM rows and columns; a full interval makes unknown
    // footprints alias all TMEM.
    AllocationSlice slice(Interval<size_t>{});
    slice.physicalFootprint = regions.getFootprint(access.value);
    if (access.isRead)
      accesses.syncReadSlices[slice].insert(op);
    if (access.isWrite)
      accesses.syncWriteSlices[slice].insert(op);
  }
  return accesses;
}

enum class TMemBoundary { None, Wait };

static TMemBoundary getTMemBoundary(Operation *op) {
  // Defer completion at CTA barriers until a hazard or boundary needs it.
  if (isa<gpu::BarrierOp>(op))
    return TMemBoundary::None;
  // SCF has been lowered. Complete accesses before every CFG and WS edge.
  if (isa<BranchOpInterface, CallOpInterface, gpu::WarpSpecializeOp,
          gpu::WarpSpecializePartitionsOp, gpu::WarpYieldOp, gpu::WarpReturnOp>(
          op) ||
      op->hasTrait<OpTrait::ReturnLike>())
    return TMemBoundary::Wait;
  // A following 2CTA MMA needs completion before the cluster rendezvous.
  if (auto barrier = dyn_cast<ClusterBarrierOp>(op))
    return barrier.getRelaxed() ? TMemBoundary::None : TMemBoundary::Wait;
  if (auto atomic = dyn_cast<AtomicOpInterface>(op))
    if (atomic.getMemSemantic() == MemSemantic::RELEASE ||
        atomic.getMemSemantic() == MemSemantic::ACQUIRE_RELEASE)
      return TMemBoundary::Wait;
  if (auto barrier = dyn_cast<gpu::MBarrierOpInterface>(op))
    if (!barrier.getBarriers().empty())
      return TMemBoundary::Wait;
  // Volatile loads still access only global memory.
  if (isa<triton::LoadOp>(op))
    return TMemBoundary::None;

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory)
    return isMemoryEffectFree(op) ? TMemBoundary::None : TMemBoundary::Wait;
  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  auto boundary = TMemBoundary::None;
  for (const auto &effect : effects) {
    Value value = effect.getValue();
    // Opaque effects can hide synchronization in inline assembly or externs.
    if (!value) {
      if (isa<SideEffects::DefaultResource>(effect.getResource()))
        return TMemBoundary::Wait;
    } else if (isTensorMemory(value) &&
               isa<MemoryEffects::Free>(effect.getEffect())) {
      boundary = TMemBoundary::Wait;
    }
  }
  return boundary;
}

class TMemSyncAnalysis : public MembarAnalysis {
public:
  using MembarAnalysis::MembarAnalysis;

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
                 TMEMWaitKind kind) {
    return incomplete.isIntersected(
        effects,
        [&](Operation *lhs, Operation *rhs, bool isRead, bool rhsIsRead,
            Allocation *allocation) {
          return isRead != (kind == TMEMWaitKind::LOAD) ||
                 filter(lhs, rhs, isRead, rhsIsRead, allocation);
        },
        &allocation);
  }

  void waitBeforeBarrier(TMEMWaitKind kind) {
    auto &before = accesses(beforeBarrier, kind);
    if (before.empty())
      return;
    assert(lastBarrier && "completion needs a remembered rendezvous");
    waitsBefore[lastBarrier].push_back(kind);
    // The unconditional wait completes all preceding accesses of this kind.
    // Preserve later accesses, even if the same operation also has pending
    // accesses from an earlier loop iteration.
    before.clear();
    accesses(incomplete, kind).clear();
    incomplete.join(afterBarrier);
  }

  void complete(TMEMWaitKind kind) {
    accesses(incomplete, kind).clear();
    accesses(afterBarrier, kind).clear();
  }

  void waitNow(Operation *op, TMEMWaitKind kind) {
    if (accesses(incomplete, kind).empty())
      return;
    waitsBefore[op].push_back(kind);
    complete(kind);
  }

  void rememberBarrier(Operation *op, MembarInfo &info) {
    lastBarrier = op;
    info.sync();
    beforeBarrier = incomplete;
    afterBarrier.sync();
  }

  void flush(Operation *op) {
    for (TMEMWaitKind kind : {TMEMWaitKind::LOAD, TMEMWaitKind::STORE}) {
      waitBeforeBarrier(kind);
      waitNow(op, kind);
    }
    lastBarrier = nullptr;
  }

  void update(Operation *op, MembarInfo *info, FuncMapT *funcMap,
              OpBuilder *builder) override {
    waitsBefore.erase(op);
    if (auto wait = dyn_cast<TMEMWaitOp>(op)) {
      complete(wait.getKind());
      return;
    }

    BlockInfo effects = getTMemAccesses(op, regions);
    auto stages = getBarrierStages(op);
    // A barrier inside this operation can cover incoming accesses, but waits
    // inserted before the operation cannot complete its own TMEM effects.
    bool beforeEffects =
        stages.beforeMemoryEffects ||
        (effects.syncReadSlices.empty() && effects.syncWriteSlices.empty());
    if (stages.hasBarrier() && beforeEffects)
      rememberBarrier(op, *info);

    auto boundary = getTMemBoundary(op);
    auto call = dyn_cast<CallOpInterface>(op);
    auto summary = call ? getCallSummary(call, *funcMap) : std::nullopt;
    if (summary)
      effects = summary->entryBlockInfo;

    // Choose the barrier before placing waits, including at calls.
    Operation *previous = op->getPrevNode();
    syncIfNeeded(op, effects, info, builder);
    if (op->getPrevNode() != previous)
      rememberBarrier(op->getPrevNode(), *info);

    if (boundary != TMemBoundary::None)
      flush(op);
    for (TMEMWaitKind kind : {TMEMWaitKind::LOAD, TMEMWaitKind::STORE})
      if (hasHazard(beforeBarrier, effects, kind))
        waitBeforeBarrier(kind);

    if (summary) {
      // TMEM addresses are assigned across the module; no frame offset.
      info->applyCallSummary(*summary);
      return;
    }

    if (!info->allPathsFromEntrySynced)
      info->entryBlockInfo.join(effects);
    if (stages.hasBarrier() && !beforeEffects)
      rememberBarrier(op, *info);

    // Only loads and stores can require a barrier before later TMEM accesses.
    auto kind = getTMemAccessKind(op);
    if (kind == TMemAccessKind::Load || kind == TMemAccessKind::Store) {
      info->pending.join(effects);
      incomplete.join(effects);
      afterBarrier.join(effects);
    }
  }

  // Track accesses before and after lastBarrier within one block.
  DenseMap<Operation *, SmallVector<TMEMWaitKind, 2>> waitsBefore;
  BlockInfo incomplete;
  Operation *lastBarrier = nullptr;
  BlockInfo beforeBarrier;
  BlockInfo afterBarrier;
};

static LogicalResult runTMemAnalysis(ModuleOp mod) {
  auto solver = createDataFlowSolver();
  auto *regions = solver->load<BufferRegionAnalysis>(
      BufferRegionAnalysis::Mode::TensorMemoryOnly);
  if (failed(solver->initializeAndRun(mod)))
    return failure();
  ModuleAllocation allocation(mod);
  ModuleMembarAnalysis analysis(allocation, filterFn);
  analysis.runAnalysis<TMemSyncAnalysis>(*regions);
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
    if (!mod.walk([](Operation *op) {
              return getTMemAccessKind(op) == TMemAccessKind::None
                         ? WalkResult::advance()
                         : WalkResult::interrupt();
            })
             .wasInterrupted())
      return;
    if (failed(runTMemAnalysis(mod)))
      return signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
