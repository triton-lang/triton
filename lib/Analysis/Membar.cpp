#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {

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

namespace {

enum class ThreadSyncKind {
  Ordinary,
  Publication,
  // Both completion kinds only establish completion or acquire state; they do
  // not access payload or consume another completion's unpublished effects.
  Completion,
  CompletionNeedsSync,
};

enum class ThreadSyncIssuer {
  Unknown,
  // All effects are issued by thread zero of the current execution partition.
  Thread0,
  // A fixed leader of partition-relative warp zero, not necessarily lane zero.
  Warp0Leader,
};

struct ThreadSyncInfo {
  // Rendezvous requirements, not barriers provided by the operation itself.
  ThreadSyncKind kind = ThreadSyncKind::Ordinary;
  ThreadSyncIssuer issuer = ThreadSyncIssuer::Unknown;

  bool requiresBefore() const { return kind == ThreadSyncKind::Publication; }

  bool requiresAfter() const {
    return kind == ThreadSyncKind::CompletionNeedsSync;
  }

  bool isCompletionOnly() const {
    return kind == ThreadSyncKind::Completion ||
           kind == ThreadSyncKind::CompletionNeedsSync;
  }
};

ThreadSyncInfo getThreadSyncInfo(Operation *op) {
  if (op->hasTrait<OpTrait::MemWaitOpTrait>() ||
      isa<ttng::TensormapFenceproxyAcquireOp>(op))
    return {ThreadSyncKind::CompletionNeedsSync};
  if (auto wait = dyn_cast<ttng::WarpGroupDotWaitOp>(op)) {
    // WGMMA waits only synchronize the issuing warp group.
    return {!wait.getWarpGroupLocal() && triton::gpu::lookupNumWarps(op) > 4
                ? ThreadSyncKind::CompletionNeedsSync
                : ThreadSyncKind::Completion};
  }
  if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp>(op)) {
    // Init and inval use the same full-mask election, or thread zero before
    // SM90.
    return {ThreadSyncKind::Ordinary, ThreadSyncIssuer::Warp0Leader};
  }
  if (isa<ttng::ArriveBarrierOp, ttng::BarrierExpectOp>(op)) {
    auto fromCTA = isa<ttng::ArriveBarrierOp>(op)
                       ? cast<ttng::ArriveBarrierOp>(op).getFromCTA()
                       : cast<ttng::BarrierExpectOp>(op).getFromCTA();
    // Nonidentity routing may use several lanes to signal the peer CTAs.
    return {ThreadSyncKind::Publication,
            !fromCTA ? ThreadSyncIssuer::Thread0 : ThreadSyncIssuer::Unknown};
  }
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op))
    return {mma.getCompletionBarriers().empty() ? ThreadSyncKind::Ordinary
                                                : ThreadSyncKind::Publication};
  if (isa<ttng::TCGen5CommitOp>(op))
    return {ThreadSyncKind::Publication};
  return {};
}

// Whether both operations use the same fixed issuer in each CTA of one region.
bool haveSameThreadSyncIssuer(Operation *lhs, Operation *rhs) {
  // Equal issuer kinds in different regions can denote different physical
  // threads, including sibling warp-specialized partitions and callees.
  if (lhs->getParentRegion() != rhs->getParentRegion())
    return false;
  auto issuer = getThreadSyncInfo(lhs).issuer;
  return issuer != ThreadSyncIssuer::Unknown &&
         issuer == getThreadSyncInfo(rhs).issuer;
}

} // namespace

bool BlockInfo::requiresThreadSync(const BlockInfo &other) const {
  // Only effect -> demand edges require a rendezvous; independent completion
  // operations can publish together.
  for (Operation *before : threadEffects)
    for (Operation *after : other.threadDemands)
      if (getThreadSyncInfo(before).requiresAfter() ||
          (getThreadSyncInfo(after).requiresBefore() &&
           !haveSameThreadSyncIssuer(before, after)))
        return true;
  return false;
}

void MembarAnalysis::syncIfNeeded(Operation *op, const BlockInfo &effects,
                                  MembarInfo *membarInfo, OpBuilder *builder,
                                  bool cluster) {
  auto &pending = membarInfo->pending;
  auto canSkip = [&](Operation *before, Operation *after, bool beforeIsRead,
                     bool afterIsRead, Allocation *allocation) {
    // Effects issued by the same thread are ordered with respect to each other.
    // A fixed thread-zero issuer and an elected warp-zero leader may differ.
    return (pending.threadEffects.count(before) &&
            effects.threadEffects.count(after) &&
            haveSameThreadSyncIssuer(before, after)) ||
           (filter &&
            filter(before, after, beforeIsRead, afterIsRead, allocation));
  };
  if (!pending.requiresThreadSync(effects) &&
      !pending.isIntersected(effects, canSkip, &allocation, sliceFilter))
    return;
  // The barrier clears incoming state. The operation's own effects still
  // follow it, including a scratch write that conflicts with a pending read.
  builder->setInsertionPoint(op);
  insertBarrier(op, builder, cluster);
  membarInfo->sync();
}

void MembarAnalysis::insertBarrier(Operation *op, OpBuilder *builder,
                                   bool cluster) {
  OpBuilder::InsertionGuard g(*builder);
  if (cluster)
    ttng::ClusterBarrierOp::create(*builder, op->getLoc());
  else
    triton::gpu::BarrierOp::create(*builder, op->getLoc(),
                                   triton::gpu::AddrSpace::Local);
}

static Allocation::BufferId getScratchBufferId(Operation *op,
                                               Allocation *allocation) {
  // A call's allocation belongs to the callee and is translated separately.
  if (isa<CallOpInterface>(op))
    return Allocation::InvalidBufferId;
  return allocation->getBufferId(op);
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

triton::BarrierStages getLocalBarrierStages(Operation *op,
                                            Allocation *allocation) {
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
  auto scratchBufferId = getScratchBufferId(op, allocation);
  bool hasScratchBarrier = scratchBufferId != Allocation::InvalidBufferId &&
                           !scratchBufferUsesWarpSync(op);

  // Atomic polls always end in a rendezvous. With scratch, the rendezvous is
  // between the scratch write and read; otherwise it follows all effects.
  if (isa<triton::AtomicPollOp>(op)) {
    stages.betweenMemoryEffects = hasScratchBarrier;
    stages.afterMemoryEffects = !hasScratchBarrier;
    return stages;
  }

  if (auto atomic = dyn_cast<triton::AtomicOpInterface>(op)) {
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
      isa<triton::gpu::WarpSpecializePartitionsOp, triton::gpu::WarpYieldOp,
          triton::gpu::WarpReturnOp>(op);

  // Warp specialization writes its captures before the launch rendezvous.
  if (isa<triton::gpu::WarpSpecializeOp>(op))
    stages.beforeMemoryEffects = !hasScratchBarrier;
  return stages;
}

void MembarAnalysis::updateSuccessor(Operation *terminator, Block *successor,
                                     MembarInfo *membarInfo) {
  if (bufferIndexAnalysis.isBackedgeSuccessor(terminator, successor)) {
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

triton::BarrierStages MembarAnalysis::getBarrierStages(Operation *op) {
  return getLocalBarrierStages(op, &allocation);
}

void MembarAnalysis::update(Operation *op, MembarInfo *membarInfo,
                            FuncMapT *funcMap, OpBuilder *builder) {
  auto sync = getThreadSyncInfo(op);
  // Region control flow is visited by the dataflow engine. Its children
  // contribute their own effects; implicit scratch belongs to this op.
  bool controlFlow = isa<BranchOpInterface, RegionBranchOpInterface,
                         RegionBranchTerminatorOpInterface>(op);
  bool isReturn = op->hasTrait<OpTrait::ReturnLike>();
  // Cluster barriers contribute synchronization stages, not payload effects.
  bool hasEffects =
      sync.requiresBefore() || sync.requiresAfter() ||
      getScratchBufferId(op, &allocation) != Allocation::InvalidBufferId ||
      (!controlFlow && !isReturn && !isa<ttng::ClusterBarrierOp>(op) &&
       !isMemoryEffectFree(op));

  BlockInfo effects;
  if (hasEffects) {
    effects.threadEffects.insert(op);
    if (!sync.isCompletionOnly())
      effects.threadDemands.insert(op);
  }
  if (isReturn && isa<FunctionOpInterface>(op->getParentOp()))
    effects.threadDemands.insert(op);
  updateMemoryEffects(op, membarInfo, funcMap, builder, /*cluster=*/false,
                      std::move(effects));
}

SmallVector<AllocationSlice> MembarAnalysis::getAllocationSlices(Value value) {
  auto function = cast<FunctionOpInterface>(allocation.getOperation());
  // Device-function views use their own frame until imported at a call;
  // foreign or mixed frames, including returned descriptors, stay unknown.
  auto *footprint = regions.getFootprint(value, function);
  SmallVector<AllocationSlice> slices;
  auto addSlice = [&](Interval<size_t> interval,
                      Allocation::BufferId bufferId) {
    auto slice = bufferIndexAnalysis.makeSlice(value, interval, bufferId);
    slice.physicalFootprint = footprint;
    slices.push_back(std::move(slice));
  };
  Allocation::BufferIdSetT bufferIds;
  if (accessMode == AccessMode::AllocatorAliasesOnly) {
    // Allocation identity survives unknown geometry. Direct argument effects
    // have no local IDs; retain them until a caller binds their allocation.
    bufferIds = allocation.getAllBufferIdsWithAliases(value);
    auto argument = dyn_cast<BlockArgument>(value);
    if (argument && argument.getOwner() == &function.getBlocks().front()) {
      AllocationSlice slice(Interval<size_t>{});
      slice.argumentIndex = argument.getArgNumber();
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

void MembarAnalysis::updateMemoryEffects(Operation *op, MembarInfo *membarInfo,
                                         FuncMapT *funcMap, OpBuilder *builder,
                                         bool cluster, BlockInfo curBlockInfo) {
  auto barrierStages = getBarrierStages(op);
  if (barrierStages.beforeMemoryEffects) {
    // Model a leading barrier before handling the operation's effects.
    membarInfo->sync();
  }

  if (auto call = dyn_cast<CallOpInterface>(op)) {
    auto summary = getCallSummary(
        call, *funcMap, [&](MembarInfo &summary, FunctionOpInterface callee) {
          summary.transformSlices([&](const AllocationSlice &slice) {
            if (!slice.argumentIndex)
              return SmallVector<AllocationSlice>{
                  slice.translateToCallsite(call, callee, regions)};
            Value actual = call.getArgOperands()[*slice.argumentIndex];
            auto slices = getAllocationSlices(actual);
            if (!actual.getDefiningOp<triton::gpu::LocalAllocOp>() &&
                llvm::none_of(slices, [](const AllocationSlice &bound) {
                  return bound.argumentIndex.has_value();
                }))
              return SmallVector<AllocationSlice>{};
            // A callee can reinterpret the argument's view. Retain the caller's
            // allocation IDs, but cover each whole allocation without shifting.
            for (AllocationSlice &bound : slices) {
              bound.physicalFootprint = nullptr;
              bound.invalidateIterationInfo();
            }
            return slices;
          });
        });
    assert(summary && "expected callee summary");
    syncIfNeeded(op, summary->entryBlockInfo, membarInfo, builder, cluster);
    membarInfo->applyCallSummary(*summary);
    return;
  }

  // Intra-function dependencies
  // Explicit buffer
  for (const auto &access : triton::getMemoryAccesses(op)) {
    Value value = access.value;
    auto memory = cast<triton::gpu::MemDescType>(value.getType());
    if (!isa<triton::gpu::SharedMemorySpaceAttr>(memory.getMemorySpace()))
      continue;
    // Shared effects cover only descriptor elements, including gather/scatter
    // and async copies. Footprints retain padding, partitions and CTA identity.
    for (const AllocationSlice &slice : getAllocationSlices(value)) {
      if (access.isWrite)
        curBlockInfo.syncWriteSlices[slice].insert(op);
      if (access.isRead)
        curBlockInfo.syncReadSlices[slice].insert(op);
    }
  }

  // Scratch buffer operations consist of a series of shared memory operations
  // starting from a shared memory write, followed by a series of shared memory
  // read/write operations, and ending with a shared memory read, i.e., shared
  // memory write -> ... -> shared memory read.
  auto scratchBufferId = getScratchBufferId(op, &allocation);
  std::optional<AllocationSlice> scratchSlice;
  if (scratchBufferId != Allocation::InvalidBufferId) {
    bool hasExplicitSharedDeps = !curBlockInfo.syncReadSlices.empty() ||
                                 !curBlockInfo.syncWriteSlices.empty();
    if (hasExplicitSharedDeps &&
        !isa<triton::gpu::LocalAtomicScatterRMWOp>(op)) {
      llvm::report_fatal_error(
          "scratch buffer operations should not have any shared memory "
          "dependencies");
    }
    auto interval = allocation.getAllocatedInterval(scratchBufferId);
    scratchSlice.emplace(interval);
    scratchSlice->physicalFootprint = regions.getScratchFootprint(op);
    curBlockInfo.syncWriteSlices[*scratchSlice].insert(op);
  }

  syncIfNeeded(op, curBlockInfo, membarInfo, builder, cluster);

  if (scratchSlice) {
    if (barrierStages.betweenMemoryEffects) {
      // The internal barrier synchronizes all incoming effects. Do not carry
      // them past the operation; only effects after the barrier are outgoing.
      bool hasThreadEffects = curBlockInfo.threadEffects.count(op);
      membarInfo->addBlockInfo(curBlockInfo);
      membarInfo->sync();
      curBlockInfo.sync();
      // Final scratch reads still need to finish before a later publication.
      if (hasThreadEffects)
        curBlockInfo.threadEffects.insert(op);
    }
    curBlockInfo.syncReadSlices[*scratchSlice].insert(op);
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  membarInfo->addBlockInfo(curBlockInfo);

  if (barrierStages.afterMemoryEffects) {
    // Model a trailing barrier after handling the operation's effects.
    membarInfo->sync();
  }
}

void ModuleMembarAnalysis::run() { run<MembarAnalysis>(); }

} // namespace mlir
