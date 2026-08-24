#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUTMEMBARRIERINSERTIONPASS
#define GEN_PASS_DEF_TRITONNVIDIAGPUTMEMWAITINSERTIONPASS
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

static BlockInfo getTMemAccesses(Operation *op, BufferRegionAnalysis *regions) {
  BlockInfo accesses;
  for (const MemoryAccess &access : getMemoryAccesses(op)) {
    if (!isTensorMemory(access.value))
      continue;
    // Footprints preserve TMEM rows and columns; a full interval makes unknown
    // footprints alias all TMEM.
    AllocationSlice slice(Interval<size_t>{});
    slice.physicalFootprint = regions->getFootprint(access.value);
    if (access.isRead)
      accesses.syncReadSlices[slice].insert(op);
    if (access.isWrite)
      accesses.syncWriteSlices[slice].insert(op);
  }
  return accesses;
}

class TMemBarrierAnalysis : public MembarAnalysis {
public:
  using MembarAnalysis::MembarAnalysis;

private:
  void update(Operation *operation, MembarInfo *membarInfo, FuncMapT *funcMap,
              OpBuilder *builder) override;
};

void TMemBarrierAnalysis::update(Operation *op, MembarInfo *membarInfo,
                                 FuncMapT *funcMap, OpBuilder *builder) {
  auto stages = mlir::getLocalBarrierStages(op, &allocation);
  if (stages.beforeMemoryEffects)
    membarInfo->sync();

  if (isa<triton::CallOp>(op)) {
    auto call = cast<CallOpInterface>(op);
    if (auto callee = dyn_cast<FunctionOpInterface>(call.resolveCallable())) {
      // Tensor-memory allocation attributes are physical addresses assigned
      // across the whole module, so callee slices need no call-frame offset.
      MembarInfo calleeMembarInfo = funcMap->lookup(callee);
      if (membarInfo->pending.isIntersected(calleeMembarInfo.entryBlockInfo,
                                            filter, &allocation)) {
        builder->setInsertionPoint(op);
        insertBarrier(op, builder);
        membarInfo->sync();
      }
      membarInfo->applyCallSummary(calleeMembarInfo);
    }
    return;
  }

  BlockInfo curBlockInfo = getTMemAccesses(op, regions);
  if (membarInfo->pending.isIntersected(curBlockInfo, filter, &allocation)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    membarInfo->sync();
  }

  membarInfo->addBlockInfo(curBlockInfo);
  // A leading or interior rendezvous must preserve the operation's own effects.
  if (stages.afterMemoryEffects ||
      (stages.betweenMemoryEffects && curBlockInfo.syncReadSlices.empty() &&
       curBlockInfo.syncWriteSlices.empty()))
    membarInfo->sync();
}

// Ops before which we 100% need to flush reads and writes
bool isWaitBoundary(Operation *op) {
  // Defer waits at CTA barriers until a hazard or boundary requires them.
  if (isa<gpu::BarrierOp>(op))
    return false;

  // Flush before starting or ending WS
  if (isa<gpu::WarpSpecializeOp, gpu::WarpSpecializePartitionsOp,
          gpu::WarpYieldOp, gpu::WarpReturnOp, CallOpInterface>(op))
    return true;
  // A non-relaxed cluster barrier could be followed by a 2CTA MMA, in which
  // case we need to put the wait before the cluster barrier.
  // We could track barriers and cluster barriers separately
  // and place the wait before the previous relevant op, but
  // I don't think there are many use cases for that.
  if (auto barrier = dyn_cast<ClusterBarrierOp>(op))
    return !barrier.getRelaxed();
  // Atomic ops may synchronise as well
  if (auto atomic = dyn_cast<AtomicOpInterface>(op))
    if (atomic.getMemSemantic() == MemSemantic::RELEASE ||
        atomic.getMemSemantic() == MemSemantic::ACQUIRE_RELEASE)
      return true;
  // Mbarriers may signal other WS partitions.
  if (auto barrier = dyn_cast<gpu::MBarrierOpInterface>(op))
    if (!barrier.getBarriers().empty())
      return true;

  // Carry pending accesses through control-flow joins and backedges.
  if (isa<BranchOpInterface, RegionBranchOpInterface>(op) ||
      (isa<RegionBranchTerminatorOpInterface>(op) &&
       isa<RegionBranchOpInterface>(op->getParentOp())))
    return false;
  if (op->hasTrait<OpTrait::ReturnLike>())
    return true;

  // Volatile loads add an opaque write effect to prevent optimization, but
  // still access only global memory.
  if (isa<triton::LoadOp>(op))
    return false;

  // Wait if we can't prove that it's memory-effect free
  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory)
    return !isMemoryEffectFree(op);
  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  return llvm::any_of(effects, [](const auto &effect) {
    Value value = effect.getValue();
    // - An unbound DefaultResource effect may hide synchronization
    //   in impure inline asm or extern calls.
    if (!value)
      return isa<SideEffects::DefaultResource>(effect.getResource());
    return isTensorMemory(value) &&
           isa<MemoryEffects::Free>(effect.getEffect());
  });
}

class TMemWaitAnalysis : public MembarAnalysis {
public:
  using MembarAnalysis::MembarAnalysis;

private:
  static BlockInfo::SliceMapT &accesses(BlockInfo &info, TMEMWaitKind kind) {
    return kind == TMEMWaitKind::LOAD ? info.syncReadSlices
                                      : info.syncWriteSlices;
  }

  bool hasHazard(const BlockInfo &pending, const BlockInfo &effects,
                 TMEMWaitKind kind) {
    return pending.isIntersected(
        effects,
        [kind](Operation *, Operation *, bool isRead, bool, Allocation *) {
          return isRead != (kind == TMEMWaitKind::LOAD);
        },
        &allocation);
  }

  void waitBeforeBarrier(TMEMWaitKind kind, BlockInfo &pending,
                         OpBuilder *builder) {
    if (accesses(beforeBarrier, kind).empty())
      return;
    assert(lastBarrier && "publication needs a remembered rendezvous");
    builder->setInsertionPoint(lastBarrier);
    TMEMWaitOp::create(*builder, lastBarrier->getLoc(), kind);
    // The unconditional wait completes all preceding accesses of this kind.
    // Preserve later accesses, even if the same operation also has pending
    // accesses from an earlier loop iteration.
    accesses(beforeBarrier, kind).clear();
    accesses(pending, kind).clear();
    pending.join(afterBarrier);
  }

  void waitNow(Operation *op, TMEMWaitKind kind, BlockInfo &pending,
               OpBuilder *builder) {
    if (accesses(pending, kind).empty())
      return;
    builder->setInsertionPoint(op);
    TMEMWaitOp::create(*builder, op->getLoc(), kind);
    accesses(pending, kind).clear();
    accesses(afterBarrier, kind).clear();
  }

  void rememberBarrier(Operation *op, const BlockInfo &pending) {
    lastBarrier = op;
    beforeBarrier = pending;
    afterBarrier.sync();
  }

  void finishBlock(BlockInfo &pending, OpBuilder *builder) {
    // Complete accesses preceding lastBarrier before forgetting it.
    // Leave later accesses in pending for joins and backedges.
    for (TMEMWaitKind kind : {TMEMWaitKind::LOAD, TMEMWaitKind::STORE})
      waitBeforeBarrier(kind, pending, builder);
    lastBarrier = nullptr;
    afterBarrier.sync();
  }

  void update(Operation *op, MembarInfo *membarInfo, FuncMapT *,
              OpBuilder *builder) override {
    BlockInfo &pending = membarInfo->pending;
    if (auto wait = dyn_cast<TMEMWaitOp>(op)) {
      accesses(pending, wait.getKind()).clear();
      accesses(afterBarrier, wait.getKind()).clear();
      return;
    }

    BlockInfo effects = getTMemAccesses(op, regions);
    // Include internal scratch barriers even without descriptor memory effects.
    auto stages = getLocalBarrierStages(op, &allocation);
    // Reuse this barrier for incoming hazards only if it precedes the op's TMEM
    // effects, or the op has no TMEM effects.
    bool beforeEffects =
        stages.beforeMemoryEffects ||
        (effects.syncReadSlices.empty() && effects.syncWriteSlices.empty());
    if (stages.hasBarrier() && beforeEffects)
      rememberBarrier(op, pending);

    if (isWaitBoundary(op)) {
      finishBlock(pending, builder);
      for (TMEMWaitKind kind : {TMEMWaitKind::LOAD, TMEMWaitKind::STORE})
        waitNow(op, kind, pending, builder);
    }

    bool loadHazard = hasHazard(pending, effects, TMEMWaitKind::LOAD) ||
                      hasHazard(beforeBarrier, effects, TMEMWaitKind::LOAD);
    bool storeHazard = hasHazard(pending, effects, TMEMWaitKind::STORE) ||
                       hasHazard(beforeBarrier, effects, TMEMWaitKind::STORE);
    if (loadHazard || storeHazard) {
      // Reuse the last barrier if no later access conflicts with this
      // operation. Otherwise, insert a wait and a new barrier before the
      // operation.
      if (lastBarrier &&
          !afterBarrier.isIntersected(effects, nullptr, &allocation)) {
        if (loadHazard)
          waitBeforeBarrier(TMEMWaitKind::LOAD, pending, builder);
        if (storeHazard)
          waitBeforeBarrier(TMEMWaitKind::STORE, pending, builder);
      } else {
        if (loadHazard)
          waitNow(op, TMEMWaitKind::LOAD, pending, builder);
        if (storeHazard)
          waitNow(op, TMEMWaitKind::STORE, pending, builder);
        builder->setInsertionPoint(op);
        insertBarrier(op, builder);
        rememberBarrier(op->getPrevNode(), pending);
      }
    }

    // Handle this operation's hazards before recording its internal barrier.
    // Waits go before the operation, so its own accesses remain in
    // afterBarrier.
    if (stages.hasBarrier() && !beforeEffects)
      rememberBarrier(op, pending);

    // Track loads and stores, including allocation initializers. An allocation
    // without an initializer preserves pending accesses to reused storage.
    auto kind = getTMemAccessKind(op);
    if (kind == TMemAccessKind::Load || kind == TMemAccessKind::Store) {
      pending.join(effects);
      afterBarrier.join(effects);
    }

    if (op->hasTrait<OpTrait::IsTerminator>() ||
        isa<RegionBranchOpInterface>(op))
      finishBlock(pending, builder);
  }

  // These snapshots track accesses before and after lastBarrier within one
  // virtual block. Only MembarInfo::pending flows through joins and backedges.
  Operation *lastBarrier = nullptr;
  BlockInfo beforeBarrier;
  BlockInfo afterBarrier;
};

template <typename AnalysisT>
LogicalResult runTMemAnalysis(ModuleOp mod, MembarFilterFn filter = nullptr) {
  auto solver = createDataFlowSolver();
  auto *regions = solver->load<BufferRegionAnalysis>(
      BufferRegionAnalysis::Mode::TensorMemoryOnly);
  if (failed(solver->initializeAndRun(mod)))
    return failure();
  ModuleAllocation allocation(mod);
  ModuleMembarAnalysis analysis(allocation, std::move(filter));
  analysis.runAnalysis<AnalysisT>(regions);
  return success();
}

} // namespace

struct TMemWaitInsertionPass
    : public impl::TritonNvidiaGPUTMemWaitInsertionPassBase<
          TMemWaitInsertionPass> {
  using impl::TritonNvidiaGPUTMemWaitInsertionPassBase<
      TMemWaitInsertionPass>::TritonNvidiaGPUTMemWaitInsertionPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod.walk([](Operation *op) {
              auto kind = getTMemAccessKind(op);
              return kind == TMemAccessKind::Load ||
                             kind == TMemAccessKind::Store
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            })
             .wasInterrupted())
      return;
    // Run after TMEM and shared-memory barrier insertion. Calls and returns
    // complete pending accesses, so this pass needs no call summaries.
    if (failed(runTMemAnalysis<TMemWaitAnalysis>(mod)))
      return signalPassFailure();
  }
};

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
    if (failed(runTMemAnalysis<TMemBarrierAnalysis>(mod, filterFn)))
      return signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
