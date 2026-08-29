#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <memory>
#include <utility>

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

class TMemWarpOwnership {
public:
  explicit TMemWarpOwnership(BufferRegionAnalysis &regions)
      : regions(regions) {}

  bool isSameWarp(Operation *load, Operation *store, Allocation *allocation) {
    // Function summaries and different WS regions may have different issuers.
    if (load->getParentRegion() != store->getParentRegion() ||
        load->getParentOfType<FunctionOpInterface>() !=
            allocation->getOperation())
      return false;
    auto [it, inserted] = sameWarps.try_emplace({load, store}, false);
    if (!inserted)
      return it->second;
    const Info *lhs = getInfo(load);
    const Info *rhs = getInfo(store);
    if (!lhs || !rhs)
      return false;
    for (const auto &a : lhs->footprint->regionInfo.views)
      for (const auto &b : rhs->footprint->regionInfo.views) {
        if (!a.allocationFrame || a.allocationFrame != b.allocationFrame)
          return false;
        if (a.affineCTAOffset == b.affineCTAOffset &&
            a.region.baseOffset == b.region.baseOffset &&
            lhs->warps == rhs->warps)
          continue;
        for (const auto &[cta, addresses] : a.region.ctaAddresses)
          for (const auto &[otherCTA, otherAddresses] : b.region.ctaAddresses) {
            if (cta != otherCTA)
              continue;
            auto overlap = addresses.intersection(otherAddresses);
            if (overlap.empty())
              continue;
            if (a.affineCTAOffset != b.affineCTAOffset)
              return false;
            // Descriptor offsets use integer addition, not layout XOR.
            for (uint32_t address : overlap) {
              if (lhs->getWarp(address - a.region.baseOffset) !=
                  rhs->getWarp(address - b.region.baseOffset))
                return false;
            }
          }
      }
    return it->second = true;
  }

private:
  struct Info {
    // (row, word column) -> warp.
    LinearLayout warps;
    const BufferRegionFootprint *footprint;

    uint32_t getWarp(uint32_t address) const {
      auto dims = llvm::to_vector<2>(warps.getInDimNames());
      return warps
          .apply({{dims[0], int32_t(address >> 16)},
                  {dims[1], int32_t(address & 0xffff)}})
          .front()
          .second;
    }
  };

  static LinearLayout getWarpLayout(RankedTensorType regTy,
                                    gpu::MemDescType memTy) {
    auto layout = computeTMemLdStLayout(regTy, memTy);
    assert(succeeded(layout) && "TMEM layout must have been verified");
    auto *ctx = regTy.getContext();
    auto warp = StringAttr::get(ctx, "warp");
    auto row = StringAttr::get(ctx, "row");
    auto col = StringAttr::get(ctx, "col");

    // Match the footprint's 32-bit words. Packed elements may share a word.
    unsigned bitwidth = memTy.getElementTypeBitWidth();
    unsigned elementsPerWord = 32 / bitwidth;
    auto rows = layout->getOutDimSize(row);
    auto cols =
        std::max<int32_t>(1, layout->getOutDimSize(col) / elementsPerWord);
    auto toWords = LinearLayout::identity1D(rows, row, row) *
                   LinearLayout::zeros1D(elementsPerWord, col, col) *
                   LinearLayout::identity1D(cols, col, col);
    auto accesses = layout->compose(toWords);

    // Complete the image after the real inputs so the inverse prefers them.
    auto padding = StringAttr::get(ctx, "padding");
    auto completion = LinearLayout::identity1D(rows, padding, row) *
                      LinearLayout::identity1D(cols, padding, col);
    return accesses.concatIns(completion)
        .pseudoinvert()
        .sublayout({row, col}, {warp});
  }

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
    if (!footprint)
      return nullptr;
    auto warps = getWarpLayout(regTy, cast<gpu::MemDescType>(mem.getType()));
    // A non-surjective inverse hides other warps accessing the same word.
    // The load wait only orders accesses within its issuing warp.
    if (!warps.isSurjective())
      return nullptr;
    it->second = std::make_unique<Info>(Info{std::move(warps), footprint});
    return it->second.get();
  }

  BufferRegionAnalysis &regions;
  DenseMap<Operation *, std::unique_ptr<Info>> info;
  DenseMap<std::pair<Operation *, Operation *>, bool> sameWarps;
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

// Wait: Complete outstanding TMEM accesses with tmem_wait.
// Publication: Wait + require a rendezvous between issuing warps if needed.
enum class TMemBoundary { None, Wait, Publication };

static TMemBoundary getTMemBoundary(Operation *op) {
  if (requiresThreadSyncBefore(op))
    return TMemBoundary::Publication;
  if (isa<ArriveBarrierOp>(op))
    return TMemBoundary::Wait;
  // Defer completion at CTA barriers until a hazard or boundary needs it.
  if (isa<gpu::BarrierOp>(op))
    return TMemBoundary::None;
  // SCF has been lowered. Complete accesses before every CFG and WS edge.
  if (isa<BranchOpInterface, gpu::WarpSpecializeOp,
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

  // Update the state when we add or find a tmem_wait
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

    // Choose the barrier before placing waits.
    auto syncBefore = [&](const BlockInfo &effects) {
      Operation *previous = op->getPrevNode();
      syncIfNeeded(op, effects, info, builder);
      if (op->getPrevNode() != previous)
        rememberBarrier(op->getPrevNode(), *info);
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
    if (boundary == TMemBoundary::Publication)
      effects.threadDemands.insert(op);

    syncBefore(effects);

    if (boundary != TMemBoundary::None) {
      flush(op, pending);
    } else {
      for (TMEMWaitKind kind : {TMEMWaitKind::LOAD, TMEMWaitKind::STORE})
        if (hasHazard(beforeBarrier, effects, kind))
          waitBeforeBarrier(kind);
      // Same-warp WAR still needs completion immediately before the store.
      if (hasHazard(beforeBarrier, effects, TMEMWaitKind::LOAD,
                    /*needsBarrier=*/false) ||
          hasHazard(afterBarrier, effects, TMEMWaitKind::LOAD,
                    /*needsBarrier=*/false))
        waitNow(op, TMEMWaitKind::LOAD, pending);
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

  // Planed tmem_wait insertions
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
  TMemWarpOwnership ownership(*regions);
  auto filter = [&](Operation *lhs, Operation *rhs, bool lhsIsRead,
                    bool rhsIsRead, Allocation *allocation) {
    return filterFn(lhs, rhs, lhsIsRead, rhsIsRead, allocation) ||
           (lhsIsRead && !rhsIsRead &&
            ownership.isSameWarp(lhs, rhs, allocation));
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
