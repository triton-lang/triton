#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseMap.h"

#include <memory>
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

class TMemThreadOwnership {
public:
  explicit TMemThreadOwnership(BufferRegionAnalysis &regions)
      : regions(regions) {}

  bool isSameThread(Operation *load, Operation *store, Allocation *allocation) {
    // Function summaries and different WS regions may have different issuers.
    if (load->getParentRegion() != store->getParentRegion() ||
        load->getParentOfType<FunctionOpInterface>() !=
            allocation->getOperation())
      return false;
    const Info *lhs = getInfo(load);
    const Info *rhs = getInfo(store);
    if (!lhs || !rhs || lhs->threadRows != rhs->threadRows)
      return false;
    for (const auto &a : lhs->footprint->regionInfo.views)
      for (const auto &b : rhs->footprint->regionInfo.views) {
        if (!a.allocationFrame || a.allocationFrame != b.allocationFrame)
          return false;
        if (((a.region.baseOffset >> 16) != (b.region.baseOffset >> 16) ||
             a.affineCTAOffset != b.affineCTAOffset) &&
            a.region.intersects(b.region))
          return false;
      }
    return true;
  }

private:
  struct Info {
    LinearLayout threadRows;
    const BufferRegionFootprint *footprint;
  };

  const Info *getInfo(Operation *op) {
    auto [it, inserted] = info.try_emplace(op);
    if (!inserted)
      return it->second.get();
    RankedTensorType regTy;
    Value mem;
    if (auto load = dyn_cast<TMEMLoadOp>(op)) {
      regTy = load.getType();
      mem = load.getSrc();
    } else if (auto store = dyn_cast<TMEMStoreOp>(op)) {
      regTy = store.getSrc().getType();
      mem = store.getDst();
    } else if (auto alloc = dyn_cast<TMEMAllocOp>(op);
               alloc && alloc.getSrc()) {
      regTy = alloc.getSrc().getType();
      mem = alloc.getResult();
    } else {
      return nullptr;
    }
    auto *footprint = regions.getFootprint(mem);
    if (!footprint || footprint->regionInfo.views.empty() ||
        gpu::lookupNumWarps(op) != 4)
      return nullptr;
    auto encoding = computeTMemLdStEncodingInfo(
        regTy, cast<gpu::MemDescType>(mem.getType()), getContextualMaxNReg(op));
    if (failed(encoding))
      return nullptr;
    auto *ctx = op->getContext();
    auto reg = StringAttr::get(ctx, "register");
    auto lane = StringAttr::get(ctx, "lane");
    auto warp = StringAttr::get(ctx, "warp");
    auto row = StringAttr::get(ctx, "row");
    auto tile = getTileLayout(ctx, encoding->atom, encoding->unpacked,
                              /*withWarp=*/true);
    auto threadRows = tile.sublayout({lane, warp}, {row});
    // Each physical row must have one thread owner, independent of packing,
    // register permutation, and message width. Repetitions must stay in that
    // row.
    if (!threadRows.isInvertible() || !tile.sublayoutIsZero({reg}, {row}) ||
        encoding->reps.getInDimSize(warp) != 4 ||
        !encoding->reps.sublayoutIsZero(
            llvm::to_vector(encoding->reps.getInDimNames()), {row}))
      return nullptr;
    it->second = std::make_unique<Info>(Info{std::move(threadRows), footprint});
    return it->second.get();
  }

  BufferRegionAnalysis &regions;
  DenseMap<Operation *, std::unique_ptr<Info>> info;
};

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

enum class TMemBoundary { None, Wait, Publication };

static TMemBoundary getTMemBoundary(Operation *op) {
  if (requiresThreadSyncBefore(op))
    return TMemBoundary::Publication;
  if (auto arrive = dyn_cast<ArriveBarrierOp>(op);
      arrive && arrive.getArrivalWarps())
    return TMemBoundary::Wait;
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
      return TMemBoundary::Publication;
  if (auto barrier = dyn_cast<gpu::MBarrierOpInterface>(op))
    if (!barrier.getBarriers().empty()) {
      bool publishes = !isa<WaitBarrierOp>(op) &&
                       hasSharedAccess(op, gpu::SharedKind::Barrier, RW::Write);
      return publishes ? TMemBoundary::Publication : TMemBoundary::Wait;
    }
  // Volatile loads still access only global memory.
  if (isa<triton::LoadOp>(op))
    return TMemBoundary::None;
  if (isa<FenceAsyncSharedOp, FenceMBarrierInitReleaseClusterOp,
          gpu::AsyncCommitGroupOp, gpu::AsyncWaitOp, TMAStoreWaitOp,
          WarpGroupDotWaitOp>(op))
    return TMemBoundary::Wait;

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory)
    return isMemoryEffectFree(op) ? TMemBoundary::None
                                  : TMemBoundary::Publication;
  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  auto boundary = TMemBoundary::None;
  for (const auto &effect : effects) {
    Value value = effect.getValue();
    // Opaque effects can hide synchronization in inline assembly or externs.
    if (!value) {
      if (isa<SideEffects::DefaultResource>(effect.getResource()))
        return TMemBoundary::Publication;
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
                 TMEMWaitKind kind, bool needsBarrier = true) {
    return incomplete.isIntersected(
        effects,
        [&](Operation *lhs, Operation *rhs, bool isRead, bool rhsIsRead,
            Allocation *allocation) {
          return isRead != (kind == TMEMWaitKind::LOAD) ||
                 (needsBarrier &&
                  filter(lhs, rhs, isRead, rhsIsRead, allocation));
        },
        &allocation);
  }

  void waitBeforeBarrier(TMEMWaitKind kind) {
    auto &before = accesses(beforeBarrier, kind);
    if (before.empty())
      return;
    assert(lastBarrier && "completion needs a remembered rendezvous");
    waitsBefore[lastBarrier].push_back(kind);
    before.clear();
  }

  void complete(TMEMWaitKind kind, BlockInfo &pending) {
    // A wait after lastBarrier restores its preceding accesses for publication.
    // Later accesses are already in pending.
    auto &before = accesses(beforeBarrier, kind);
    for (auto &[slice, ops] : before)
      accesses(pending, kind)[slice].insert(ops.begin(), ops.end());
    before.clear();
    accesses(afterBarrier, kind).clear();
  }

  void waitNow(Operation *op, TMEMWaitKind kind, BlockInfo &pending) {
    if (accesses(beforeBarrier, kind).empty() &&
        accesses(afterBarrier, kind).empty())
      return;
    waitsBefore[op].push_back(kind);
    complete(kind, pending);
  }

  void rememberBarrier(Operation *op, MembarInfo &info) {
    lastBarrier = op;
    info.sync();
    beforeBarrier.join(afterBarrier);
    afterBarrier.sync();
  }

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
    return !effects.threadDemands.empty() && (!pending.syncReadSlices.empty() ||
                                              !pending.syncWriteSlices.empty());
  }

  void update(Operation *op, MembarInfo *info, FuncMapT *funcMap,
              OpBuilder *builder) override {
    waitsBefore.erase(op);
    BlockInfo &pending = info->pending;
    if (auto wait = dyn_cast<TMEMWaitOp>(op)) {
      complete(wait.getKind(), pending);
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
    if (boundary == TMemBoundary::Publication || (call && !summary))
      effects.threadDemands.insert(op);

    // Choose the barrier before placing waits, including at calls.
    Operation *previous = op->getPrevNode();
    syncIfNeeded(op, effects, info, builder);
    if (op->getPrevNode() != previous)
      rememberBarrier(op->getPrevNode(), *info);

    if (boundary != TMemBoundary::None)
      flush(op, pending);
    for (TMEMWaitKind kind : {TMEMWaitKind::LOAD, TMEMWaitKind::STORE})
      if (hasHazard(beforeBarrier, effects, kind))
        waitBeforeBarrier(kind);
    // Same-thread WAR still needs completion immediately before the store.
    if (hasHazard(beforeBarrier, effects, TMEMWaitKind::LOAD,
                  /*needsBarrier=*/false) ||
        hasHazard(afterBarrier, effects, TMEMWaitKind::LOAD,
                  /*needsBarrier=*/false))
      waitNow(op, TMEMWaitKind::LOAD, pending);

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
      pending.join(effects);
      afterBarrier.join(effects);
    }
  }

  // Every CFG/WS edge and call completes these accesses. Only ordinary
  // unpublished effects propagate through MembarInfo and function summaries.
  DenseMap<Operation *, SmallVector<TMEMWaitKind, 2>> waitsBefore;
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
  TMemThreadOwnership ownership(*regions);
  auto filter = [&](Operation *lhs, Operation *rhs, bool lhsIsRead,
                    bool rhsIsRead, Allocation *allocation) {
    return filterFn(lhs, rhs, lhsIsRead, rhsIsRead, allocation) ||
           (lhsIsRead && !rhsIsRead &&
            ownership.isSameThread(lhs, rhs, allocation));
  };
  ModuleAllocation allocation(mod);
  ModuleMembarAnalysis analysis(allocation, std::move(filter));
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
