#include "MembarCFG.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {

using triton::membar::SMemModel;

namespace {

enum class ThreadSyncKind {
  Ordinary,
  Publication,
  // Each warp publishes its completed effects without a CTA rendezvous.
  WarpPublication,
  // Completions only establish completion or acquire state; they do
  // not access payload or consume another completion's unpublished effects.
  Completion,
  CompletionNeedsSync,
  // These completions do not publish writes to global memory.
  SharedCompletionNeedsSync,
  // Finish waits before a notification can let a peer advance their phase.
  MBarrierWait,
};

struct ThreadSyncInfo {
  // Rendezvous requirements, not barriers provided by the operation itself.
  ThreadSyncKind kind = ThreadSyncKind::Ordinary;
  ThreadIssuer::Kind issuer = ThreadIssuer::Unknown;

  bool requiresBefore() const { return kind == ThreadSyncKind::Publication; }

  bool requiresWarpBefore() const {
    return kind == ThreadSyncKind::WarpPublication;
  }

  bool requiresAfter() const {
    return kind == ThreadSyncKind::CompletionNeedsSync ||
           kind == ThreadSyncKind::SharedCompletionNeedsSync;
  }

  bool hasThreadDemand() const {
    return kind == ThreadSyncKind::Ordinary ||
           kind == ThreadSyncKind::Publication;
  }
};

ThreadSyncInfo getThreadSyncInfo(Operation *op) {
  // Acquires and proxy fences do not consume payload. Wait deps only keep
  // allocations live; each thread fences its own completed accesses.
  if (isa<ttng::WaitBarrierOp>(op))
    return {ThreadSyncKind::MBarrierWait};
  if (isa<ttng::FenceAsyncSharedOp>(op))
    return {ThreadSyncKind::Completion};
  if (isa<triton::gpu::AsyncWaitOp>(op))
    return {ThreadSyncKind::SharedCompletionNeedsSync};
  if (auto wait = dyn_cast<ttng::TMAStoreWaitOp>(op);
      wait && wait.getReadOnly())
    return {ThreadSyncKind::SharedCompletionNeedsSync};
  if (op->hasTrait<OpTrait::MemWaitOpTrait>() ||
      isa<ttng::TensormapFenceproxyAcquireOp>(op))
    return {ThreadSyncKind::CompletionNeedsSync};
  if (auto wait = dyn_cast<ttng::WarpGroupDotWaitOp>(op)) {
    // WGMMA waits only synchronize the issuing warp group.
    return {!wait.getWarpGroupLocal() && triton::gpu::lookupNumWarps(op) > 4
                ? ThreadSyncKind::CompletionNeedsSync
                : ThreadSyncKind::Completion};
  }
  if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::TMEMCopyOp,
          ttng::CLCTryCancelOp, ttng::FenceMBarrierInitReleaseClusterOp>(op)) {
    // Each issuing CTA elects a lane in its partition's warp zero.
    // Init and inval use thread zero before SM90.
    return {ThreadSyncKind::Ordinary, ThreadIssuer::Warp0Leader};
  }
  if (auto barrier = dyn_cast<triton::gpu::MBarrierOpInterface>(op);
      barrier && barrier.isPerWarp())
    return {ThreadSyncKind::WarpPublication};
  if (isa<ttng::ArriveBarrierOp, ttng::BarrierExpectOp>(op)) {
    auto fromCTA = isa<ttng::ArriveBarrierOp>(op)
                       ? cast<ttng::ArriveBarrierOp>(op).getFromCTA()
                       : cast<ttng::BarrierExpectOp>(op).getFromCTA();
    // Nonidentity routing may use several lanes to signal the peer CTAs.
    return {ThreadSyncKind::Publication,
            !fromCTA ? ThreadIssuer::Thread0 : ThreadIssuer::Unknown};
  }
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op))
    return {mma.getCompletionBarriers().empty() ? ThreadSyncKind::Ordinary
                                                : ThreadSyncKind::Publication,
            ThreadIssuer::Warp0Leader};
  if (isa<ttng::TCGen5CommitOp>(op))
    return {ThreadSyncKind::Publication, ThreadIssuer::Warp0Leader};
  return {};
}

} // namespace

// Every observer stays in this region and CTA. Follow views and selects;
// calls, captures and raw-address escapes reject the proof.
bool triton::membar::hasRegionLocalUses(Value root) {
  if (ttng::hasCrossCTASharedAccess(root.getDefiningOp()).value_or(false))
    return false;
  SmallVector<Value> worklist{root};
  llvm::SmallPtrSet<Value, 8> seen;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (user->hasTrait<OpTrait::MemDescViewTrait>() ||
          isa<arith::SelectOp>(user)) {
        llvm::append_range(worklist, user->getResults());
        continue;
      }
      if (user->getParentRegion() != root.getParentRegion())
        return false;
      // TMEM loads and stores only access the issuing CTA's tensor memory.
      if (isa<triton::gpu::LocalDeallocOp, ttng::TMEMLoadOp, ttng::TMEMStoreOp>(
              user))
        continue;
      // Wait dependencies only keep allocations live.
      if (auto wait = dyn_cast<ttng::WaitBarrierOp>(user);
          wait && wait.getAlloc() != value)
        continue;
      auto barrier = dyn_cast<triton::gpu::MBarrierOpInterface>(user);
      if (barrier && llvm::is_contained(barrier.getBarriers(), value)) {
        if (ttng::hasCrossCTAMBarrierUse(barrier) ||
            ttng::hasCGABroadcast(
                cast<triton::gpu::MemDescType>(value.getType())))
          return false;
        if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp>(user))
          continue;
        for (const auto &access : triton::getMemoryAccesses(user))
          if (access.value == value &&
              access.sharedKind != triton::gpu::SharedKind::Barrier)
            return false;
        continue;
      }
      auto crossCTA = ttng::hasCrossCTASharedAccess(user);
      if (!crossCTA || *crossCTA)
        return false;
    }
  }
  return true;
}

namespace {

bool hasOnlyGlobalReads(Operation *op) {
  auto effects = getEffectsRecursively(op);
  return effects && !effects->empty() &&
         llvm::all_of(*effects, [](const auto &effect) {
           return isa<MemoryEffects::Read>(effect.getEffect()) &&
                  isa<triton::GlobalMemory>(effect.getResource());
         });
}

ThreadIssuer getThreadIssuer(Operation *op) {
  auto kind = getThreadSyncInfo(op).issuer;
  return kind == ThreadIssuer::Unknown
             ? ThreadIssuer{}
             : ThreadIssuer{kind, op->getParentRegion()};
}

} // namespace

bool SMemModel::isRegionLocal(Value value) {
  auto *footprint = regions.getFootprint(value);
  return footprint &&
         llvm::all_of(footprint->regionInfo.views, [&](auto &view) {
           Value root = view.allocation->getResult(0);
           auto [it, inserted] =
               regionLocalAllocations.try_emplace(root, false);
           if (inserted)
             it->second = triton::membar::hasRegionLocalUses(root);
           return it->second;
         });
}

bool SMemModel::mayNotifyPeer(Operation *op, ArrayRef<MemoryAccess> accesses) {
  namespace tt = triton;
  namespace ttg = triton::gpu;
  namespace tti = triton::instrument;

  // We may just notify a different WS partition with barriers or global atomics
  return llvm::TypeSwitch<Operation *, bool>(op)
      // A guarded notification must join pending waits before lanes split.
      .Case<ttg::ThreadPredicateOp>([](auto) { return true; })
      // Atomic loads and polls model a write to prevent elimination.
      .Case<tt::AtomicLoadOp, tt::AtomicPollOp,
            tti::ExperimentalGSanAtomicLoadOp>([](auto) { return false; })
      .Case<tt::AtomicOpInterface, tt::GridDependencyLaunchDependentsOp>(
          [](auto) { return true; })
      .Case<ttg::LocalAtomicScatterRMWOp>(
          [&](auto atomic) { return !isRegionLocal(atomic.getDst()); })
      .Case<ttg::MBarrierOpInterface>([&](auto) {
        return llvm::any_of(accesses, [&](const MemoryAccess &access) {
          return access.isWrite && access.isShared(ttg::SharedKind::Barrier) &&
                 !isRegionLocal(access.value);
        });
      })
      .Case<tt::ElementwiseInlineAsmOp, ttg::InlineAsmOp,
            tt::ExternElementwiseOp>(
          [](auto opaque) { return !opaque.getPure(); })
      .Default(false);
}

static bool scratchBufferUsesWarpSync(Operation *op) {
  if (isa<ttng::TensormapCreateOp>(op))
    return true;
  auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
  if (!cvt)
    return false;

  auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(cvt.getType());
  auto srcLayout = triton::gpu::toLinearLayout(srcTy);
  auto dstLayout = triton::gpu::toLinearLayout(dstTy);
  auto kWarp = StringAttr::get(op->getContext(), "warp");
  return mlir::isCvtDimSync(srcLayout, dstLayout, kWarp);
}

triton::BarrierStages getLocalBarrierStages(Operation *op, bool hasScratch) {
  triton::BarrierStages stages;
  // The local-memory mask guarantees ordering of local memory accesses.
  if (auto barrier = dyn_cast<triton::gpu::BarrierOp>(op)) {
    stages.beforeMemoryEffects = stages.afterMemoryEffects = barrier.hasLocal();
    return stages;
  }
  // Explicit barriers have no accesses between their leading and trailing
  // stages. A relaxed cluster rendezvous does not order memory accesses.
  if (auto barrier = dyn_cast<ttng::ClusterBarrierOp>(op)) {
    stages.beforeMemoryEffects = stages.afterMemoryEffects =
        !barrier.getRelaxed();
    return stages;
  }
  // Pure layout conversions and reductions can still use shared scratch and
  // internal barriers even without descriptor memory effects.
  bool hasScratchBarrier = hasScratch && !scratchBufferUsesWarpSync(op);

  // Atomic polls always end in a rendezvous. With scratch, the rendezvous is
  // between the scratch write and read; otherwise it follows all effects.
  if (isa<triton::AtomicPollOp>(op)) {
    stages.betweenMemoryEffects = hasScratchBarrier;
    stages.afterMemoryEffects = !hasScratchBarrier;
    return stages;
  }

  if (auto atomic = dyn_cast<triton::AtomicOpInterface>(op)) {
    if (op->getParentOfType<triton::gpu::ThreadScopeOp>())
      return stages;
    // Atomic result broadcast uses a scratch write, rendezvous, and read for
    // every memory semantic, including relaxed.
    return triton::getAtomicBarrierStages(atomic.getMemSemantic(),
                                          hasScratchBarrier);
  }

  // Scratch-backed operations contain a rendezvous between their scratch
  // write and read phases. Warp-specialization boundaries synchronize before
  // the operation.
  stages.betweenMemoryEffects = hasScratchBarrier;
  stages.beforeMemoryEffects =
      isa<triton::gpu::WarpYieldOp, triton::gpu::WarpReturnOp>(op);

  // The first launch rendezvous publishes captures. The second finishes their
  // reads before any partition starts its body and reuses capture storage.
  if (isa<triton::gpu::WarpSpecializeOp>(op)) {
    stages.beforeMemoryEffects = !hasScratchBarrier;
    stages.afterMemoryEffects = true;
  }
  return stages;
}

bool SMemModel::hasThreadEffects(Operation *op) {
  // A handoff only accesses its private scratch, modeled separately below.
  if (isa<triton::gpu::ThreadHandoffOp>(op))
    return false;
  // ThreadPredicateOp retains unknown effects so pending waits finish before
  // entering its restricted region, while every enclosing thread participates.
  auto effects = getEffectsRecursively(op);
  return !effects || llvm::any_of(*effects, [](const auto &effect) {
    return !isa<triton::gpu::SharedMemory>(effect.getResource()) ||
           !isa<MemoryEffects::Allocate, MemoryEffects::Free>(
               effect.getEffect());
  });
}

static CompletionSync getCompletionSync(ThreadSyncInfo sync) {
  if (sync.kind == ThreadSyncKind::CompletionNeedsSync)
    return CompletionSync::All;
  if (sync.kind == ThreadSyncKind::SharedCompletionNeedsSync)
    return CompletionSync::Shared;
  return CompletionSync::None;
}

BlockInfo SMemModel::getThreadEffects(Operation *op) {
  BlockInfo effects;
  effects.threadSync.effectIssuers = IssuerSummary{getThreadIssuer(op)};
  auto sync = getThreadSyncInfo(op);
  auto completion = getCompletionSync(sync);
  effects.threadSync.completion =
      completionScope && completion != CompletionSync::None
          ? completionScope(op, completion)
          : completion;
  effects.threadSync.peerWaitNeedsSync =
      sync.kind == ThreadSyncKind::MBarrierWait &&
      !isRegionLocal(cast<ttng::WaitBarrierOp>(op).getAlloc());
  return effects;
}

void SMemModel::addThreadDemand(BlockInfo &effects, Operation *op,
                                ArrayRef<MemoryAccess> accesses) {
  effects.threadDemands.hasDemand = true;
  effects.threadDemands.mayNotifyPeer = mayNotifyPeer(op, accesses);
  effects.threadDemands.hasDemandBeyondGlobalReads = !hasOnlyGlobalReads(op);
  if (getThreadSyncInfo(op).requiresBefore())
    effects.threadDemands.publicationIssuers =
        IssuerSummary{getThreadIssuer(op)};
}

SmallVector<AllocationSlice> SMemModel::getAllocationSlices(
    Value value, std::optional<triton::gpu::SharedKind> sharedKind) {
  auto function = cast<FunctionOpInterface>(allocation.getOperation());
  // Device-function views use their own frame until imported at a call;
  // foreign or mixed frames, including returned descriptors, stay unknown.
  auto *footprint = regions.getFootprint(value, function);
  SmallVector<AllocationSlice> slices;
  auto addSlice = [&](Interval<size_t> interval,
                      Allocation::BufferId bufferId) {
    auto slice = bufferIndexAnalysis.makeSlice(value, interval, bufferId);
    slice.physicalFootprint = footprint;
    slice.sharedKind = sharedKind;
    slices.push_back(std::move(slice));
  };
  Allocation::BufferIdSetT bufferIds;
  if (accessMode == AccessMode::AllocatorAliasesOnly) {
    // Allocation identity survives unknown geometry. Argument effects
    // have no local IDs; retain them until a caller binds their allocation.
    bufferIds = allocation.getAllBufferIdsWithAliases(value);
    for (unsigned argument : allocation.getAliasedArgumentIndices(value)) {
      AllocationSlice slice(Interval<size_t>{});
      slice.argumentIndex = argument;
      slice.sharedKind = sharedKind;
      slices.push_back(std::move(slice));
    }
  } else if (footprint) {
    // Use every lattice origin: function-local alias analysis can miss roots
    // returned through calls or joined with descriptor arguments.
    for (const auto &view : footprint->regionInfo.views) {
      auto ids = view.allocation
                     ? allocation.getBufferIds(view.allocation->getResult(0))
                     : SmallVector<Allocation::BufferId>{};
      if (ids.empty()) {
        bufferIds.clear();
        break;
      }
      bufferIds.insert(ids.begin(), ids.end());
    }
  }
  // Unknown allocations use an unbounded interval. View disjointness and
  // RAW/WAR/WAW checks still determine whether a barrier is needed.
  if (accessMode == AccessMode::AllSharedAccesses && bufferIds.empty())
    addSlice(Interval<size_t>{}, Allocation::InvalidBufferId);
  for (auto bufferId : bufferIds)
    addSlice(allocation.getAllocatedInterval(bufferId), bufferId);
  return slices;
}

MembarInfo SMemModel::translateCallSummary(CallOpInterface call,
                                           FunctionOpInterface callee,
                                           MembarInfo summary) {
  summary.transformSlices([&](const AllocationSlice &slice) {
    if (!slice.argumentIndex)
      return SmallVector<AllocationSlice>{
          slice.translateToCallsite(call, callee, regions)};
    Value actual = call.getArgOperands()[*slice.argumentIndex];
    auto slices = getAllocationSlices(actual, slice.sharedKind);
    // A callee can reinterpret the argument's view. Retain the caller's
    // allocation IDs, but cover each whole allocation without shifting.
    for (AllocationSlice &bound : slices) {
      bound.physicalFootprint = nullptr;
      bound.invalidateIterationInfo();
    }
    return slices;
  });
  return summary;
}

void SMemModel::describe(Operation *op, OpModel &model,
                         ArrayRef<MemoryAccess> accesses,
                         std::optional<BarrierStages> clusterStages) {
  if (auto barrier = dyn_cast<triton::gpu::BarrierOp>(op);
      !clusterStages && barrier && barrier.isWarp()) {
    model.fixedAfter = warpBarrier;
    return;
  }

  auto scratchId = allocation.getBufferId(op);
  bool hasScratch = scratchId != Allocation::InvalidBufferId;
  auto barriers =
      clusterStages ? *clusterStages : getLocalBarrierStages(op, hasScratch);
  model.fixedBefore = barriers.beforeMemoryEffects ? fullBarrier : 0;
  model.fixedInside =
      hasScratch && barriers.betweenMemoryEffects ? fullBarrier : 0;
  model.fixedAfter = barriers.afterMemoryEffects ? fullBarrier : 0;
  if (!clusterStages) {
    auto sync = getThreadSyncInfo(op);
    // Region control flow is visited by the dataflow engine. Its children
    // contribute their own effects; implicit scratch belongs to this op.
    bool controlFlow = isa<BranchOpInterface, RegionBranchOpInterface,
                           RegionBranchTerminatorOpInterface>(op);
    bool isReturn = op->hasTrait<OpTrait::ReturnLike>();
    // Cluster barriers contribute synchronization stages, not payload effects.
    bool hasEffects =
        sync.requiresBefore() || sync.requiresAfter() || hasScratch ||
        (!controlFlow && !isReturn && !isa<ttng::ClusterBarrierOp>(op) &&
         hasThreadEffects(op));

    model.needsWarp = sync.requiresWarpBefore();
    BlockInfo &effects = model.sharedIn;
    if (hasEffects) {
      effects = getThreadEffects(op);
      if (sync.hasThreadDemand())
        addThreadDemand(effects, op, accesses);
    }
    if (isReturn && isa<FunctionOpInterface>(op->getParentOp()))
      addThreadDemand(effects, op, accesses);
  }
  // Intra-function dependencies
  // Explicit buffer
  for (const auto &access : accesses) {
    Value value = access.value;
    auto memory = cast<triton::gpu::MemDescType>(value.getType());
    if (!isa<triton::gpu::SharedMemorySpaceAttr>(memory.getMemorySpace()))
      continue;
    // Footprints include allocation padding and retain subviews, partitions
    // and CTA identity.
    for (const AllocationSlice &slice :
         getAllocationSlices(value, access.sharedKind)) {
      if (access.isWrite)
        model.sharedIn.syncWriteSlices[slice].insert(op);
      if (access.isRead)
        model.sharedIn.syncReadSlices[slice].insert(op);
    }
  }

  if (!hasScratch) {
    model.sharedOut = model.sharedIn;
  } else {
    // Scratch buffer operations consist of a series of shared memory operations
    // starting from a shared memory write, followed by a series of shared
    // memory read/write operations, and ending with a shared memory read, i.e.,
    // shared memory write -> ... -> shared memory read.
    AllocationSlice scratch(allocation.getAllocatedInterval(scratchId));
    scratch.physicalFootprint = regions.getScratchFootprint(op);
    model.sharedIn.syncWriteSlices[scratch].insert(op);
    if (model.fixedInside) {
      // The internal barrier synchronizes all incoming effects. Do not carry
      // them past the operation; only effects after the barrier are outgoing.
      model.sharedOut.threadSync = model.sharedIn.threadSync;
    } else {
      model.sharedOut = model.sharedIn;
    }
    // Final scratch reads still need to finish before a later publication.
    model.sharedOut.syncReadSlices[scratch].insert(op);
  }
}

} // namespace mlir
