#include "triton/Analysis/Membar/SmemModel.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Membar/CFGBuilder.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {

using triton::membar::SMemModel;

namespace triton::membar {

static bool mayCrossCTA(Operation *op, const MemoryAccess &access) {
  using namespace nvidia_gpu;
  // Init, inval and wait execute on the CTA that owns the mbarrier.
  if (gpu::lookupNumCTAs(op) == 1 ||
      isa<InitBarrierOp, InvalBarrierOp, WaitBarrierOp, CLCLoadResultOp,
          gpu::AsyncCopyGlobalToLocalOp>(op))
    return false;
  if (access.isShared(gpu::SharedKind::Barrier))
    return hasCrossCTAMBarrierUse(cast<gpu::MBarrierOpInterface>(op)) ||
           hasCGABroadcast(cast<gpu::MemDescType>(access.value.getType()));
  if (access.isShared(gpu::SharedKind::Async))
    return hasCrossCTAAsyncSharedAccess(op);
  if (auto store = dyn_cast<AsyncSharedStoreOp>(op))
    return isCrossCTALoadStore(store.getDst().getType(),
                               store.getSrc().getType());
  return hasCrossCTASharedAccess(op).value_or(true);
}

const BufferIndexAnalysis &
SharedAccessModel::getIndices(FunctionOpInterface function) {
  return indices.try_emplace(function, function).first->second;
}

const BufferRegionFootprint *SharedAccessModel::getFootprint(Value value) {
  return regions.getKernelFootprint(regions.getFootprint(value));
}

const BufferRegionFootprint *
SharedAccessModel::getScratchFootprint(Operation *op) {
  return regions.getKernelFootprint(regions.getScratchFootprint(op));
}

SharedAccess SharedAccessModel::getSharedAccess(Operation *op,
                                                const MemoryAccess &access) {
  return {getFootprint(access.value), access.isWrite, mayCrossCTA(op, access)};
}

} // namespace triton::membar

AllocationSlice::AllocationSlice(Value value,
                                 Interval<size_t> allocationInterval,
                                 Allocation::BufferId bufferId)
    : allocationInterval(allocationInterval),
      accessTy(cast<triton::gpu::MemDescType>(value.getType())),
      bufferId(bufferId) {
  // Get the memdesc_subslice information if present. If no subslice is
  // present the whole interval is accessed
  if (auto subslice = value.getDefiningOp<triton::gpu::MemDescSubsliceOp>()) {
    // The source supplies coordinates even if a preceding subslice has not
    // folded, or the descriptor is carried through control flow or a loop.
    subsliceOffsets = subslice.getOffsets();
    subsliceSource = subslice.getSrc();
  }
}

bool AllocationSlice::intersects(const AllocationSlice &other) const {
  // Disjoint intervals don't overlap
  if (!allocationInterval.intersects(other.allocationInterval))
    return false;

  // Physical footprints include every possible origin across loop iterations,
  // retaining CTA and memory-space identity.
  if (!triton::mayOverlap(physicalFootprint, other.physicalFootprint))
    return false;

  // Compare indices of the same source descriptor to prove disjoint slots,
  // including arguments without allocator IDs.
  if (bufferId == other.bufferId &&
      areBufferIndicesProvablyDifferent(*this, other))
    return false;

  // Compare logical offsets only for subslices of the same source descriptor.
  if (!subsliceSource || subsliceSource != other.subsliceSource)
    return true;

  auto shapeA = accessTy.getShape();
  auto shapeB = other.accessTy.getShape();
  // Chek if all subslice region dimensions have some intersection
  // [offsetA, offsetA + shape) and [offsetB, offsetB + other.shape)
  // If any dimension doesn't intersect, we are looking at disjoint subslices
  for (size_t i = 0; i < subsliceOffsets.size(); ++i) {
    int64_t startA = subsliceOffsets[i];
    int64_t endA = startA + shapeA[i];
    int64_t startB = other.subsliceOffsets[i];
    int64_t endB = startB + shapeB[i];

    // Is A completely before B? Is B completely before A? If so, disjoint
    if (endA <= startB || endB <= startA)
      return false;
  }

  // All dimensions of subslices have some intersection
  return true;
}

void AllocationSlice::print(raw_ostream &os) const {
  os << "interval=[" << allocationInterval.start() << ","
     << allocationInterval.end() << ")";

  if (bufferId != Allocation::InvalidBufferId)
    os << " buffer=" << bufferId;
  if (argumentIndex)
    os << " argument=" << *argumentIndex;

  os << " offsets=[";
  if (!subsliceOffsets.empty()) {
    llvm::interleaveComma(subsliceOffsets, os);
  } else {
    os << "unknown";
  }
  os << "]";

  os << " shape=";
  if (accessTy) {
    llvm::interleave(accessTy.getShape(), os, "x");
    os << " layout=" << accessTy.getEncoding();
  } else {
    os << "? layout=unknown";
  }
}

void BlockInfo::invalidateIterationInfo() {
  transformSlices([](AllocationSlice slice) {
    slice.invalidateIterationInfo();
    return SmallVector<AllocationSlice>{std::move(slice)};
  });
}

AllocationSlice AllocationSlice::translateToCallsite(
    CallOpInterface call, FunctionOpInterface callee,
    triton::BufferRegionAnalysis &regions) const {
  assert(!argumentIndex && "argument effects must be bound to call operands");
  AllocationSlice shifted = *this;
  // Unknown accesses span the whole frame; shifting their endpoint overflows.
  if (allocationInterval != Interval<size_t>{}) {
    size_t offset = regions.getCallOffset(call);
    shifted.allocationInterval = Interval<size_t>(
        allocationInterval.start() + offset, allocationInterval.end() + offset);
  }
  shifted.physicalFootprint =
      regions.translateToCallsite(physicalFootprint, call, callee);
  shifted.bufferId = Allocation::InvalidBufferId;
  shifted.invalidateIterationInfo();
  return shifted;
}

static ThreadIssuer joinIssuers(ThreadIssuer a, ThreadIssuer b) {
  if (a == ThreadIssuer::None)
    return b;
  if (b == ThreadIssuer::None || a == b)
    return a;
  return ThreadIssuer::Unknown;
}

void BlockInfo::joinThreadState(const BlockInfo &other) {
  const auto &sync = other.threadSync;
  threadSync.effectIssuer =
      joinIssuers(threadSync.effectIssuer, sync.effectIssuer);
  threadSync.completion = std::max(threadSync.completion, sync.completion);
  threadSync.peerWaitNeedsSync |= sync.peerWaitNeedsSync;
  const auto &demands = other.threadDemands;
  threadDemands.kind = std::max(threadDemands.kind, demands.kind);
  threadDemands.publicationIssuer =
      joinIssuers(threadDemands.publicationIssuer, demands.publicationIssuer);
  threadDemands.mayNotifyPeer |= demands.mayNotifyPeer;
}

bool BlockInfo::requiresThreadSync(const BlockInfo &pending,
                                   const BlockInfo &effects) {
  const auto &sync = pending.threadSync;
  const auto &demands = effects.threadDemands;
  if (sync.effectIssuer == ThreadIssuer::None ||
      demands.kind == ThreadDemandKind::None)
    return false;
  // The maps also include implicit scratch and translated callee accesses.
  bool hasSharedAccesses =
      !effects.syncReadSlices.empty() || !effects.syncWriteSlices.empty();
  return sync.completion == CompletionSync::All ||
         (sync.completion == CompletionSync::Shared &&
          (hasSharedAccesses || demands.kind == ThreadDemandKind::General)) ||
         (demands.publicationIssuer != ThreadIssuer::None &&
          (sync.effectIssuer == ThreadIssuer::Unknown ||
           sync.effectIssuer != demands.publicationIssuer)) ||
         // A notification can let a peer advance a pending wait's phase.
         // Private counter updates are already ordered by the slice hazards.
         (sync.peerWaitNeedsSync && demands.mayNotifyPeer);
}

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
  ThreadIssuer issuer = ThreadIssuer::Unknown;

  bool requiresBefore() const { return kind == ThreadSyncKind::Publication; }

  bool requiresWarpBefore() const {
    return kind == ThreadSyncKind::WarpPublication;
  }

  CompletionSync getCompletionSync() const {
    if (kind == ThreadSyncKind::CompletionNeedsSync)
      return CompletionSync::All;
    if (kind == ThreadSyncKind::SharedCompletionNeedsSync)
      return CompletionSync::Shared;
    return CompletionSync::None;
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

// Follow views and selects within this region. For calls, record which results
// carry each descriptor argument back to the caller.
// Only the legacy flow passes a null cache.
// TODO: Make it a reference when followCalls is removed.
static bool
usesStayLocal(Value root,
              DenseMap<Value, SmallVector<unsigned>> *calleeReturns) {
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
      if (auto call = dyn_cast<triton::CallOp>(user); calleeReturns && call) {
        auto callee = cast<triton::FuncOp>(call.resolveCallable());
        BlockArgument arg = callee.getArgument(use.getOperandNumber());
        if (calleeReturns->try_emplace(arg).second &&
            !usesStayLocal(arg, calleeReturns))
          return false;
        for (unsigned index : calleeReturns->at(arg))
          worklist.push_back(call.getResult(index));
        continue;
      }
      if (calleeReturns && isa<BlockArgument>(root) &&
          isa<triton::ReturnOp>(user)) {
        calleeReturns->at(root).push_back(use.getOperandNumber());
        continue;
      }
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

bool triton::membar::hasRegionLocalUses(Value root, bool followCalls) {
  if (ttng::hasCrossCTASharedAccess(root.getDefiningOp()).value_or(false))
    return false;
  DenseMap<Value, SmallVector<unsigned>> calleeReturns;
  return usesStayLocal(root, followCalls ? &calleeReturns : nullptr);
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

} // namespace

bool SMemModel::isRegionLocal(Value value) {
  auto *footprint = regions.getFootprint(value);
  return footprint &&
         llvm::all_of(footprint->regionInfo.views, [&](auto &view) {
           Value root = view.allocation->getResult(0);
           auto [it, inserted] =
               regionLocalAllocations.try_emplace(root, false);
           if (inserted)
             it->second = triton::membar::hasRegionLocalUses(root, followCalls);
           return it->second;
         });
}

bool SMemModel::mayNotifyPeer(Operation *op, ArrayRef<MemoryAccess> accesses) {
  namespace tt = triton;
  namespace ttg = triton::gpu;
  namespace tti = triton::instrument;

  // We may just notify a different WS partition with barriers or global atomics
  return llvm::TypeSwitch<Operation *, bool>(op)
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

static triton::BarrierStages getLocalBarrierStages(Operation *op,
                                                   bool hasScratch) {
  triton::BarrierStages stages;
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
  // Assumptions issue no memory access.
  if (isa<triton::gpu::ThreadHandoffOp, LLVM::AssumeOp>(op))
    return false;
  auto effects = getEffectsRecursively(op);
  return !effects || llvm::any_of(*effects, [](const auto &effect) {
    return !isa<triton::gpu::SharedMemory, ttng::TensorMemory>(
               effect.getResource()) ||
           !isa<MemoryEffects::Allocate, MemoryEffects::Free>(
               effect.getEffect());
  });
}

BlockInfo SMemModel::getThreadEffects(Operation *op) {
  BlockInfo effects;
  auto sync = getThreadSyncInfo(op);
  effects.threadSync.effectIssuer = sync.issuer;
  auto completion = sync.getCompletionSync();
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
  effects.threadDemands.mayNotifyPeer = mayNotifyPeer(op, accesses);
  effects.threadDemands.kind = hasOnlyGlobalReads(op)
                                   ? ThreadDemandKind::GlobalReadsOnly
                                   : ThreadDemandKind::General;
  if (getThreadSyncInfo(op).requiresBefore())
    effects.threadDemands.publicationIssuer = effects.threadSync.effectIssuer;
}

SmallVector<AllocationSlice> SMemModel::getAllocationSlices(
    Value value, std::optional<triton::gpu::SharedKind> sharedKind) {
  auto function = cast<FunctionOpInterface>(allocation.getOperation());
  // Only views in the current frame have local allocator IDs.
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
  } else if (footprint && (!followCalls || isKernel(function))) {
    // Use every lattice origin: function-local alias analysis can miss roots
    // returned through calls or joined with descriptor arguments.
    for (const auto &view : footprint->regionInfo.views)
      for (auto id : allocation.getBufferIds(view.allocation->getResult(0)))
        bufferIds.insert(id);
  } else if (followCalls) {
    footprint = regions.getKernelFootprint(regions.getFootprint(value));
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

void SMemModel::describe(Operation *op, OpModel &model, FixedSync &fixed,
                         ArrayRef<MemoryAccess> accesses,
                         std::optional<BarrierStages> clusterStages) {
  bool includeShared = accessMode != AccessMode::SynchronizationOnly;
  if (!clusterStages) {
    // A verified callback has no scratch or payload accesses to publish.
    if (isa<triton::MapElementwiseOp>(op))
      return;
    // The local-memory mask guarantees ordering of local memory accesses.
    if (auto barrier = dyn_cast<triton::gpu::BarrierOp>(op)) {
      fixed.after = barrier.hasLocal()
                        ? (barrier.isWarp() ? warpBarrier : fullBarrier)
                        : bit(Need::WarpConvergence);
      return;
    }
    if (isa<ttng::TMEMWaitOp>(op)) {
      // A scalar notification must observe completion in every issuing warp.
      // The wait itself converges each warp and adds no memory-order demand.
      if (includeShared)
        model.sharedOut.threadSync.effectIssuer = ThreadIssuer::Unknown;
      fixed.after = bit(Need::WarpConvergence);
      return;
    }
  }

  auto scratchId = allocation.getBufferId(op);
  bool hasScratch = scratchId != Allocation::InvalidBufferId;
  auto barriers =
      clusterStages ? *clusterStages : getLocalBarrierStages(op, hasScratch);
  fixed.before = barriers.beforeMemoryEffects ? fullBarrier : 0;
  fixed.inside = hasScratch && barriers.betweenMemoryEffects ? fullBarrier : 0;
  fixed.after = barriers.afterMemoryEffects ? fullBarrier : 0;
  if (!includeShared)
    return;
  if (!clusterStages) {
    auto sync = getThreadSyncInfo(op);
    // Region control flow is visited by the dataflow engine. Its children
    // contribute their own effects; implicit scratch belongs to this op.
    bool controlFlow = isa<BranchOpInterface, RegionBranchOpInterface,
                           RegionBranchTerminatorOpInterface>(op);
    bool isReturn = op->hasTrait<OpTrait::ReturnLike>();
    // Cluster barriers contribute synchronization stages, not payload effects.
    bool hasEffects =
        sync.requiresBefore() ||
        sync.getCompletionSync() != CompletionSync::None || hasScratch ||
        (!controlFlow && !isReturn && !isa<ttng::ClusterBarrierOp>(op) &&
         hasThreadEffects(op));

    model.needsWarp = sync.requiresWarpBefore();
    BlockInfo &effects = model.sharedIn;
    if (hasEffects) {
      effects = getThreadEffects(op);
      if (sync.hasThreadDemand())
        addThreadDemand(effects, op, accesses);
    }
    // TODO: Retire this legacy return demand with MembarAnalysis. The CFG
    // models function returns in MembarModel::describe().
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
    auto interval = allocation.getAllocatedInterval(scratchId);
    auto *footprint = regions.getScratchFootprint(op);
    if (followCalls &&
        !isKernel(cast<FunctionOpInterface>(allocation.getOperation()))) {
      interval = {};
      footprint = regions.getKernelFootprint(footprint);
    }
    AllocationSlice scratch(interval);
    scratch.physicalFootprint = footprint;
    model.sharedIn.syncWriteSlices[scratch].insert(op);
    if (fixed.inside) {
      // The internal barrier synchronizes all incoming effects. Do not carry
      // them past the operation; only effects after the barrier are outgoing.
      model.sharedOut.threadSync = model.sharedIn.threadSync;
    } else {
      model.sharedOut = model.sharedIn;
    }
    // Final scratch reads still need to finish before a later publication.
    model.sharedOut.syncReadSlices[scratch].insert(op);
  }
  // A diagnostic can consume a pending completion without publishing payload.
  if (isa<triton::PrintOp, triton::AssertOp>(op))
    model.sharedOut.threadSync.effectIssuer = ThreadIssuer::None;
}

} // namespace mlir
