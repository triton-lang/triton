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
#include <optional>

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::membar {

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

void SMemData::merge(const SMemData &other) {
  llvm::append_range(accesses, other.accesses);
  threads.merge(other.threads);
}

Conflict SMemData::conflict(const SMemData &sink, const SMemOrdering *ordering,
                            Allocation &allocation, bool inKernel) const {
  auto result = Conflict::None;
  for (const SMemAccess &before : accesses)
    for (const SMemAccess &after : sink.accesses) {
      auto pair = before.conflict(after, ordering, allocation, inKernel);
      if (pair == Conflict::Always)
        return pair;
      if (pair == Conflict::AfterRebind)
        result = pair;
    }
  return result;
}

SMemAccess SMemAccess::fromDescriptor(const MemoryAccess &effect,
                                      BufferIndexAnalysis &indices,
                                      bool useIndex) {
  Kind kind = effect.isWrite ? Write : Read;
  return {.entry = kind,
          .exit = kind,
          .index = useIndex ? indices.getIndexExpr(effect.value) : nullptr,
          .subslice = indices.getSubslice(effect.value)};
}

Conflict SMemAccess::conflict(const SMemAccess &sink,
                              const SMemOrdering *ordering,
                              Allocation &allocation, bool inKernel) const {
  if (exit == None || sink.entry == None ||
      (exit == Read && sink.entry == Read))
    return Conflict::None;
  if (!mayOverlap(footprint, sink.footprint) ||
      (ordering && ordering->canSkip(*this, sink, allocation, inKernel)))
    return Conflict::None;
  // Only relative disjointness can expire on a backedge or call entry.
  if (areBufferIndicesProvablyDifferent(index, sink.index) ||
      disjointSubslices(subslice, sink.subslice))
    return Conflict::AfterRebind;
  return Conflict::Always;
}

// Follow views and selects within this region. For calls, record which results
// carry each descriptor argument back to the caller.
static bool
usesStayLocal(Value root,
              DenseMap<Value, SmallVector<unsigned>> &calleeReturns) {
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
      if (auto call = dyn_cast<triton::CallOp>(user)) {
        auto callee = cast<triton::FuncOp>(call.resolveCallable());
        BlockArgument arg = callee.getArgument(use.getOperandNumber());
        if (calleeReturns.try_emplace(arg).second &&
            !usesStayLocal(arg, calleeReturns))
          return false;
        for (unsigned index : calleeReturns.at(arg))
          worklist.push_back(call.getResult(index));
        continue;
      }
      if (isa<BlockArgument>(root) && isa<triton::ReturnOp>(user)) {
        calleeReturns.at(root).push_back(use.getOperandNumber());
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

bool hasRegionLocalUses(Value root) {
  if (ttng::hasCrossCTASharedAccess(root.getDefiningOp()).value_or(false))
    return false;
  DenseMap<Value, SmallVector<unsigned>> calleeReturns;
  return usesStayLocal(root, calleeReturns);
}

bool SMemModel::isRegionLocal(Value value) {
  auto *footprint = regions.getFootprint(value);
  return footprint &&
         llvm::all_of(footprint->regionInfo.views, [&](auto &view) {
           Value root = view.allocation->getResult(0);
           auto [it, inserted] =
               regionLocalAllocations.try_emplace(root, false);
           if (inserted)
             it->second = hasRegionLocalUses(root);
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
  // Acquires and proxy fences do not consume payload. Wait deps only keep
  // allocations live; each thread fences its own completed accesses.
  if (isa<ttng::WaitBarrierOp, ttng::FenceAsyncSharedOp>(op))
    return CompletionSync::None;
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

// A direct notification orders the work of every contributing thread.
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

uint8_t payloadIssuer(Operation *op, SMemModel::Mode mode) {
  // In one role, scalar stores and thread scopes select the same thread
  // zero across calls. A function summary can be reused by several roles.
  if (mode == SMemModel::Mode::Role) {
    auto store = dyn_cast<StoreOp>(op);
    if ((store && isa<PointerType>(store.getPtr().getType())) ||
        op->getParentOfType<gpu::ThreadScopeOp>())
      return ThreadEffects::Thread0;
  }
  if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::TMEMCopyOp,
          ttng::CLCTryCancelOp, ttng::FenceMBarrierInitReleaseClusterOp,
          ttng::MMAv5OpInterface>(op))
    return ThreadEffects::Warp0Leader;
  return ThreadEffects::AnyIssuer;
}

// Classify the operation's own payload. A full completion must precede
// even an independent global read. A shared completion may overlap it.
uint8_t joinedCompletions(Operation *op) {
  // Bulk prefetch has opaque effects but only reads global memory.
  if (isa<ttng::PrefetchBulkOp>(op))
    return ThreadEffects::FullCompletion;
  if (isa<BranchOpInterface, RegionBranchOpInterface,
          RegionBranchTerminatorOpInterface, ttng::ClusterBarrierOp,
          triton::gpu::ThreadHandoffOp, LLVM::AssumeOp>(op) ||
      op->hasTrait<OpTrait::ReturnLike>())
    return 0;
  auto effects = getEffectsRecursively(op);
  if (!effects)
    return ThreadEffects::Completions;
  uint8_t joins = 0;
  for (const auto &effect : *effects) {
    auto *resource = effect.getResource();
    // TMEM transfers have their own completion and publication protocol.
    if (isa<ttng::TensorMemory>(resource) ||
        (isa<triton::gpu::SharedMemory>(resource) &&
         isa<MemoryEffects::Allocate, MemoryEffects::Free>(effect.getEffect())))
      continue;
    // A global write can overwrite the source of an asynchronous copy.
    if (!isa<triton::GlobalMemory>(resource) ||
        !isa<MemoryEffects::Read>(effect.getEffect()))
      return ThreadEffects::Completions;
    joins = ThreadEffects::FullCompletion;
  }
  return joins;
}

} // namespace

ThreadEffects SMemModel::getThreadEffects(Operation *op, bool hasScratch,
                                          ArrayRef<MemoryAccess> accesses) {
  ThreadEffects threads;
  if (auto completion = getCompletion(op)) {
    // Acquires and waits leave effects to publish but consume no payload.
    auto scope = *completion;
    if (completionScope && scope != CompletionSync::None)
      scope = completionScope(op, scope);
    threads.pending = ThreadEffects::AnyIssuer | static_cast<uint8_t>(scope);
    if (auto wait = dyn_cast<ttng::WaitBarrierOp>(op);
        wait && !isRegionLocal(wait.getAlloc()))
      threads.pending |= ThreadEffects::PeerWait;
    return threads;
  }

  if (auto issuer = publicationIssuer(op)) {
    threads.pending = issuer;
    // Routed signals have no guaranteed participant. Direct signals join
    // only work that may have been issued outside their participating lanes.
    unsigned participants = issuer == ThreadEffects::AnyIssuer ? 0 : issuer;
    threads.joins = ThreadEffects::AnyIssuer & ~participants;
    return threads;
  }

  threads.joins =
      hasScratch ? ThreadEffects::Completions : joinedCompletions(op);
  // Diagnostics consume completions without leaving payload to publish.
  if (!threads.joins || isa<triton::PrintOp, triton::AssertOp>(op))
    return threads;
  threads.pending = payloadIssuer(op, mode);
  if (mayNotifyPeer(op, accesses))
    threads.joins |= ThreadEffects::PeerWait;
  return threads;
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

  bool hasScratch = allocation.getBufferId(op) != Allocation::InvalidBufferId;
  fixed =
      FixedSync::fromStages(getLocalBarrierStages(op, hasScratch), fullBarrier);
  if (mode == Mode::SynchronizationOnly)
    return {};

  SMemData data;
  data.threads = getThreadEffects(op, hasScratch, accesses);
  for (const auto &effect : accesses) {
    if (!effect.isShared())
      continue;
    auto access = SMemAccess::fromDescriptor(effect, bufferIndexAnalysis);
    access.footprint = regions.getFootprint(effect.value);
    // An internal rendezvous completes explicit accesses before exit.
    if (fixed.inside)
      access.exit = SMemAccess::None;
    if (ordering)
      ordering->classify(access, op, effect, data.threads, allocation);
    data.accesses.push_back(access);
  }

  // A scratch rendezvous leaves only its last read exposed. Otherwise
  // the initial write can still conflict with a subsequent reader.
  if (hasScratch)
    data.accesses.push_back(
        {.entry = SMemAccess::Write,
         .exit = fixed.inside ? SMemAccess::Read : SMemAccess::Write,
         .footprint = regions.getScratchFootprint(op)});
  if (mode == Mode::Role)
    for (auto &access : data.accesses)
      access.footprint = regions.getKernelFootprint(access.footprint);
  return data;
}

} // namespace mlir::triton::membar
