#include "triton/Analysis/Membar/SmemModel.h"
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

#include <tuple>

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {

using triton::membar::SMemData;
using triton::membar::SMemModel;
using triton::membar::ThreadEffects;

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

AllocationSlice::AllocationSlice(Value value, Interval<size_t> interval,
                                 Allocation::BufferId bufferId)
    : allocationInterval(interval), bufferId(bufferId),
      subslice(value.getDefiningOp<triton::gpu::MemDescSubsliceOp>()) {}

bool AllocationSlice::operator<(const AllocationSlice &other) const {
  auto key = [](const AllocationSlice &slice) {
    return std::tie(slice.allocationInterval, slice.bufferId, slice.subslice,
                    slice.bufferIndexExpr, slice.physicalFootprint);
  };
  return key(*this) < key(other);
}

bool AllocationSlice::mayOverlapPhysically(const AllocationSlice &other) const {
  return allocationInterval.intersects(other.allocationInterval) &&
         triton::mayOverlap(physicalFootprint, other.physicalFootprint);
}

bool AllocationSlice::disjointWithinBinding(
    const AllocationSlice &other) const {
  // Indices of the same descriptor can prove slots disjoint even for
  // arguments without a local allocator ID.
  if (bufferId == other.bufferId &&
      areBufferIndicesProvablyDifferent(*this, other))
    return true;
  if (!subslice || !other.subslice)
    return false;
  auto a = cast<triton::gpu::MemDescSubsliceOp>(subslice);
  auto b = cast<triton::gpu::MemDescSubsliceOp>(other.subslice);
  // Only a common source supplies comparable coordinates, even if it
  // is itself a subslice or is carried through control flow or a loop.
  if (a.getSrc() != b.getSrc())
    return false;
  // Disjoint half-open [offset, offset + size) intervals in any one
  // dimension prove that the subslices cannot intersect.
  for (auto [startA, sizeA, startB, sizeB] :
       llvm::zip_equal(a.getOffsets(), a.getType().getShape(), b.getOffsets(),
                       b.getType().getShape()))
    if (startA + sizeA <= startB || startB + sizeB <= startA)
      return true;
  return false;
}

AllocationSlice AllocationSlice::translateToCallsite(
    CallOpInterface call, FunctionOpInterface callee,
    triton::BufferRegionAnalysis &regions) const {
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

namespace triton::membar {

void SMemData::merge(const SMemData &other) {
  llvm::append_range(accesses, other.accesses);
  threads.merge(other.threads, !accesses.empty());
}

Conflict SMemData::conflict(const SMemData &sink, const MembarFilterFn &filter,
                            Allocation &allocation) const {
  auto result = Conflict::None;
  for (const SMemAccess &before : accesses)
    for (const SMemAccess &after : sink.accesses) {
      auto pair = before.conflict(after, filter, allocation);
      if (pair == Conflict::Always)
        return pair;
      if (pair == Conflict::AfterRebind)
        result = pair;
    }
  return result;
}

void ThreadEffects::merge(ThreadEffects other, bool usesShared) {
  pending |= other.pending;
  joins |= other.joins;
  // In a guard or call summary, a global read combined with shared storage
  // also consumes pending shared completions.
  if (usesShared && (joins & FullCompletion))
    joins |= SharedCompletion;
}

Conflict SMemAccess::conflict(const SMemAccess &sink,
                              const MembarFilterFn &filter,
                              Allocation &allocation) const {
  // Check RAW, WAR and WAW between this operation's exit and the
  // later operation's entry.
  if (exit == None || sink.entry == None ||
      (exit == Read && sink.entry == Read))
    return Conflict::None;
  if (!slice.mayOverlapPhysically(sink.slice) ||
      (filter && issuer && sink.issuer && filter(*this, sink, allocation)))
    return Conflict::None;
  // Only relative disjointness can expire on a backedge or call entry.
  return slice.disjointWithinBinding(sink.slice) ? Conflict::AfterRebind
                                                 : Conflict::Always;
}

} // namespace triton::membar

namespace {

template <typename Op> uint8_t getNotificationIssuer(Op op) {
  // Nonidentity routing may use several lanes to signal the peer CTAs.
  if (op.getFromCTA())
    return ThreadEffects::AnyIssuer;
  return op.getPerWarp() ? ThreadEffects::WarpLeaders : ThreadEffects::Thread0;
}

// A completion establishes ordering in the issuing threads. Its scope
// determines which consumers must rendezvous with those threads.
// nullopt identifies an ordinary operation that may consume prior effects.
std::optional<CompletionSync> getCompletion(Operation *op) {
  // Acquires and proxy fences do not consume payload. Wait dependencies
  // only keep allocations live. Each thread fences its own completed accesses.
  if (isa<ttng::WaitBarrierOp, ttng::FenceAsyncSharedOp>(op))
    return CompletionSync::None;
  // These waits establish shared-memory completion without guaranteeing
  // completion of global writes.
  if (isa<triton::gpu::AsyncWaitOp>(op))
    return CompletionSync::Shared;
  if (auto wait = dyn_cast<ttng::TMAStoreWaitOp>(op);
      wait && wait.getReadOnly())
    return CompletionSync::Shared;
  if (op->hasTrait<OpTrait::MemWaitOpTrait>() ||
      isa<ttng::TensormapFenceproxyAcquireOp>(op))
    return CompletionSync::All;
  if (auto wait = dyn_cast<ttng::WarpGroupDotWaitOp>(op)) {
    // WGMMA waits only synchronize the issuing warp group.
    return !wait.getWarpGroupLocal() && triton::gpu::lookupNumWarps(op) > 4
               ? CompletionSync::All
               : CompletionSync::None;
  }
  return std::nullopt;
}

// A publication orders earlier effects of its own known issuer.
uint8_t publicationIssuer(Operation *op) {
  if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op))
    return getNotificationIssuer(arrive);
  if (auto expect = dyn_cast<ttng::BarrierExpectOp>(op))
    return getNotificationIssuer(expect);
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op))
    return mma.getCompletionBarriers().empty() ? 0 : ThreadEffects::Warp0Leader;
  if (isa<ttng::TCGen5CommitOp>(op))
    return ThreadEffects::Warp0Leader;
  return 0;
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

// Classify the operation's own payload. A full completion must precede
// even an independent global read. A shared completion may overlap it.
static uint8_t joinedCompletions(Operation *op) {
  // Region children contribute their own effects. A thread handoff only
  // accesses private scratch, which is accounted for separately. Cluster
  // barriers provide fixed synchronization. Assumptions access no memory.
  if (isa<BranchOpInterface, RegionBranchOpInterface,
          RegionBranchTerminatorOpInterface, ttng::ClusterBarrierOp,
          triton::gpu::ThreadHandoffOp, LLVM::AssumeOp>(op) ||
      op->hasTrait<OpTrait::ReturnLike>())
    return 0;
  // Bulk prefetch has opaque effects but only reads global memory.
  if (isa<ttng::PrefetchBulkOp>(op))
    return ThreadEffects::FullCompletion;
  auto effects = getEffectsRecursively(op);
  if (!effects)
    return ThreadEffects::Completions;
  // Shared- and tensor-memory lifetime markers do not access payload.
  if (llvm::all_of(*effects, [](const auto &effect) {
        return isa<triton::gpu::SharedMemory, ttng::TensorMemory>(
                   effect.getResource()) &&
               isa<MemoryEffects::Allocate, MemoryEffects::Free>(
                   effect.getEffect());
      }))
    return 0;
  if (llvm::all_of(*effects, [](const auto &effect) {
        return isa<MemoryEffects::Read>(effect.getEffect()) &&
               isa<triton::GlobalMemory>(effect.getResource());
      }))
    return ThreadEffects::FullCompletion;
  return ThreadEffects::Completions;
}

ThreadEffects SMemModel::getThreadEffects(Operation *op, bool hasScratch,
                                          ArrayRef<MemoryAccess> accesses) {
  ThreadEffects threads;
  if (auto completion = getCompletion(op)) {
    // Acquires and waits leave effects to publish but consume no payload.
    // A wait does not consume unpublished effects from earlier waits.
    auto scope = *completion;
    if (completionScope && scope != CompletionSync::None)
      scope = completionScope(op, scope);
    threads.pending = ThreadEffects::AnyIssuer | static_cast<uint8_t>(scope);
    // Finish waiting before a notification lets a peer advance the phase.
    // Storage hazards already order updates to private counters.
    if (auto wait = dyn_cast<ttng::WaitBarrierOp>(op);
        wait && !isRegionLocal(wait.getAlloc()))
      threads.pending |= ThreadEffects::PeerWait;
    return threads;
  }

  if (auto issuer = publicationIssuer(op)) {
    threads.pending = issuer;
    unsigned ordered = issuer == ThreadEffects::AnyIssuer ? 0 : issuer;
    threads.joins =
        ThreadEffects::Completions | (ThreadEffects::AnyIssuer & ~ordered);
  } else {
    threads.joins =
        hasScratch ? ThreadEffects::Completions : joinedCompletions(op);
    if (!threads.joins)
      return {};
    // Diagnostics consume completions without leaving payload to publish.
    if (isa<triton::PrintOp, triton::AssertOp>(op))
      return threads;
    // Each issuing CTA uses a fixed lane of the partition's first warp.
    // Init and inval use thread zero before SM90.
    threads.pending =
        isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::TMEMCopyOp,
            ttng::CLCTryCancelOp, ttng::FenceMBarrierInitReleaseClusterOp,
            ttng::MMAv5OpInterface>(op)
            ? ThreadEffects::Warp0Leader
            : ThreadEffects::AnyIssuer;
  }
  if (mayNotifyPeer(op, accesses))
    threads.joins |= ThreadEffects::PeerWait;
  return threads;
}

SmallVector<AllocationSlice> SMemModel::getAllocationSlices(Value value) {
  auto function = cast<FunctionOpInterface>(allocation.getOperation());
  // Only views in the current frame have local allocator IDs.
  auto *footprint = regions.getFootprint(value, function);
  SmallVector<AllocationSlice> slices;
  auto addSlice = [&](Interval<size_t> interval,
                      Allocation::BufferId bufferId) {
    auto slice = bufferIndexAnalysis.makeSlice(value, interval, bufferId);
    slice.physicalFootprint = footprint;
    slices.push_back(std::move(slice));
  };
  Allocation::BufferIdSetT bufferIds;
  if (footprint && (!followCalls || isKernel(function))) {
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
  if (bufferIds.empty())
    addSlice(Interval<size_t>{}, Allocation::InvalidBufferId);
  for (auto bufferId : bufferIds)
    addSlice(allocation.getAllocatedInterval(bufferId), bufferId);
  return slices;
}

SMemData SMemModel::describe(Operation *op, FixedSync &fixed,
                             ArrayRef<MemoryAccess> accesses) {
  // A verified callback has no scratch or payload accesses to publish.
  // TMEM waits order the transfers described by the TMEM model.
  if (isa<triton::MapElementwiseOp, ttng::TMEMWaitOp>(op))
    return {};
  // The local-memory mask guarantees ordering of local memory accesses.
  if (auto barrier = dyn_cast<triton::gpu::BarrierOp>(op)) {
    if (barrier.hasLocal())
      fixed.after = barrier.isWarp() ? warpBarrier : fullBarrier;
    return {};
  }

  auto scratchId = allocation.getBufferId(op);
  bool hasScratch = scratchId != Allocation::InvalidBufferId;
  fixed =
      FixedSync::fromStages(getLocalBarrierStages(op, hasScratch), fullBarrier);
  if (accessMode == AccessMode::SynchronizationOnly)
    return {};

  SMemData data;
  data.threads = getThreadEffects(op, hasScratch, accesses);
  for (const auto &access : accesses) {
    if (!access.isShared())
      continue;
    auto kind = access.isWrite ? SMemAccess::Write : SMemAccess::Read;
    // The internal rendezvous orders all incoming effects. Only accesses
    // after it can conflict with subsequent operations.
    for (const auto &slice : getAllocationSlices(access.value))
      data.accesses.push_back({op, slice, kind,
                               fixed.inside ? SMemAccess::None : kind,
                               access.isShared(gpu::SharedKind::Barrier)});
  }

  if (!hasScratch)
    return data;

  auto interval = allocation.getAllocatedInterval(scratchId);
  auto *footprint = regions.getScratchFootprint(op);
  if (followCalls &&
      !isKernel(cast<FunctionOpInterface>(allocation.getOperation()))) {
    interval = {};
    footprint = regions.getKernelFootprint(footprint);
  }
  AllocationSlice scratch(interval);
  scratch.physicalFootprint = footprint;
  // Scratch follows write -> ... -> read. An internal barrier orders the
  // initial write. The final read must still finish before publication or
  // reuse. Without the barrier, the write also conflicts with later reads.
  data.accesses.push_back(
      {nullptr, scratch, SMemAccess::Write,
       fixed.inside ? SMemAccess::Read : SMemAccess::Write});
  return data;
}

} // namespace mlir
