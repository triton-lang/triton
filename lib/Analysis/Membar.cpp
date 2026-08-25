#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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

  // Physical footprints include every possible origin across loop iterations.
  if (physicalFootprint && other.physicalFootprint &&
      !physicalFootprint->intersects(*other.physicalFootprint))
    return false;

  // For slices of the same allocation, compare dynamic buffer indices to prove
  // that different slots do not overlap.
  if (bufferId == other.bufferId && bufferId != Allocation::InvalidBufferId &&
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
  auto rebuild = [](SliceMapT &slices) {
    SliceMapT rebuilt;
    for (const auto &[slice, ops] : slices) {
      AllocationSlice key = slice;
      key.invalidateIterationInfo();
      rebuilt[key].insert(ops.begin(), ops.end());
    }
    slices = std::move(rebuilt);
  };
  rebuild(syncReadSlices);
  rebuild(syncWriteSlices);
}

void MembarAnalysis::insertBarrier(Operation *op, OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
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
    stages.beforeMemoryEffects = barrier.hasLocal();
    return stages;
  }

  // Pure layout conversions and reductions can still use shared scratch and
  // internal barriers even without descriptor memory effects. Call frames do
  // not imply a rendezvous, and tensor-map creation and warp-only conversions
  // only synchronize a warp.
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
  // write and read phases. Other barrier-like operations behave as a barrier
  // immediately before the operation.
  stages.betweenMemoryEffects = hasScratchBarrier;
  stages.beforeMemoryEffects =
      isa<gpu::BarrierOp, ttng::ClusterBarrierOp,
          triton::gpu::WarpSpecializePartitionsOp, triton::gpu::WarpYieldOp,
          triton::gpu::WarpReturnOp, ttng::ArriveBarrierOp,
          ttng::BarrierExpectOp, ttng::TCGen5CommitOp>(op);

  // Warp specialization writes its captures before the launch rendezvous.
  if (isa<triton::gpu::WarpSpecializeOp>(op))
    stages.beforeMemoryEffects = !hasScratchBarrier;
  // Fused MMA completion synchronizes the partition before issuing the MMA.
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op))
    stages.beforeMemoryEffects = !mma.getCompletionBarriers().empty();
  if (auto wgWait = dyn_cast<ttng::WarpGroupDotWaitOp>(op))
    stages.beforeMemoryEffects =
        !wgWait.getWarpGroupLocal() && triton::gpu::lookupNumWarps(op) > 4;
  return stages;
}

// Returns true if the same block has a later wait or local barrier before any
// memory effect or nested control flow.
static bool hasSyncPointBeforeMemoryEffect(Operation *op,
                                           Allocation *allocation) {
  for (Operation *next = op->getNextNode(); next; next = next->getNextNode()) {
    auto stages = getLocalBarrierStages(next, allocation);
    if (stages.beforeMemoryEffects ||
        next->hasTrait<mlir::OpTrait::MemWaitOpTrait>())
      return true;

    // A contained barrier follows the operation's incoming shared-memory
    // effects, so it cannot protect those effects from the preceding wait.
    if (stages.betweenMemoryEffects)
      return false;

    // Barriers classified as "after" have no shared-memory effects before
    // them. Currently these are non-scratch atomics and polls.
    if (stages.afterMemoryEffects)
      return true;

    if (isa<RegionBranchOpInterface>(next) || !isMemoryEffectFree(next))
      return false;
  }
  return false;
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

void MembarAnalysis::update(Operation *op, MembarInfo *membarInfo,
                            FuncMapT *funcMap, OpBuilder *builder) {
  // A later CTA-wide synchronization can also synchronize this wait, provided
  // no memory is accessed before reaching it.
  if (auto wgWait = dyn_cast<ttng::WarpGroupDotWaitOp>(op)) {
    if (!wgWait.getWarpGroupLocal() &&
        triton::gpu::lookupNumWarps(wgWait) > 4 &&
        hasSyncPointBeforeMemoryEffect(wgWait, &allocation)) {
      wgWait->setAttr("warpGroupLocal", builder->getUnitAttr());
    }
  }

  auto barrierStages = getLocalBarrierStages(op, &allocation);
  if (barrierStages.beforeMemoryEffects) {
    // Model a leading local barrier before handling the operation's effects.
    membarInfo->sync();
  }

  // If the current op is an (async) memory wait and there is no later sync
  // point before memory is accessed, insert a barrier op and sync. This avoids
  // redundant barriers by deferring the barrier to the later sync point.
  if (op->hasTrait<mlir::OpTrait::MemWaitOpTrait>() &&
      !hasSyncPointBeforeMemoryEffect(op, &allocation)) {
    builder->setInsertionPointAfter(op);
    insertBarrier(op, builder);
    membarInfo->sync();
    return;
  }

  BlockInfo curBlockInfo;
  auto scratchBufferId = getScratchBufferId(op, &allocation);
  if (isa<triton::CallOp>(op)) {
    auto call = cast<CallOpInterface>(op);
    if (auto callee = dyn_cast<FunctionOpInterface>(call.resolveCallable())) {
      MembarInfo calleeMembarInfo = funcMap->lookup(callee);
      auto callBufferId = allocation.getBufferId(op);
      size_t callOffset = 0;
      if (callBufferId != Allocation::InvalidBufferId)
        callOffset = allocation.getAllocatedInterval(callBufferId).start();
      calleeMembarInfo.pending =
          translateBlockInfoToCallsite(calleeMembarInfo.pending, callOffset);
      calleeMembarInfo.entryBlockInfo = translateBlockInfoToCallsite(
          calleeMembarInfo.entryBlockInfo, callOffset);
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

  // Intra-function dependencies
  // Explicit buffer
  for (const auto &access : triton::getMemoryAccesses(op)) {
    Value value = access.value;
    const triton::AddressSet *footprint = nullptr;
    auto found = physicalFootprints.find(value);
    if (found != physicalFootprints.end())
      footprint = &found->second;
    for (auto bufferId : allocation.getAllBufferIdsWithAliases(value)) {
      if (bufferId == Allocation::InvalidBufferId)
        continue;
      auto interval = allocation.getAllocatedInterval(bufferId);
      auto slice = bufferIndexAnalysis.makeSlice(value, interval, bufferId);
      slice.physicalFootprint = footprint;
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
    auto scratchSlice = AllocationSlice(interval);
    curBlockInfo.syncWriteSlices[scratchSlice].insert(op);
    auto insertCTABarrier =
        membarInfo->pending.isIntersected(curBlockInfo, filter, &allocation);
    if (insertCTABarrier) {
      builder->setInsertionPoint(op);
      insertBarrier(op, builder);
    }
    if (insertCTABarrier)
      membarInfo->sync();

    if (barrierStages.betweenMemoryEffects) {
      // The internal barrier synchronizes all incoming effects. Do not carry
      // them past the operation; only effects after the barrier are outgoing.
      membarInfo->addBlockInfo(curBlockInfo);
      membarInfo->sync();
      curBlockInfo.sync();
    }
    curBlockInfo.syncReadSlices[scratchSlice].insert(op);
  } else if (membarInfo->pending.isIntersected(curBlockInfo, filter,
                                               &allocation)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    membarInfo->sync();
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  membarInfo->addBlockInfo(curBlockInfo);

  if (barrierStages.afterMemoryEffects) {
    // Model a trailing local barrier after handling the operation's effects.
    membarInfo->sync();
  }
}

SharedMemoryFootprints ModuleMembarAnalysis::getSharedMemoryFootprints() {
  ModuleOp module = moduleAllocation.getModuleOp();
  SharedMemoryFootprints footprints;

  // Physical footprints use the addresses assigned by the allocation passes.
  auto solver = createDataFlowSolver();
  auto *regions = solver->load<triton::BufferRegionAnalysis>();
  if (failed(solver->initializeAndRun(module)))
    return footprints;
  // Device-function allocations need their callsite offsets before comparison.
  for (auto function : module.getOps<FunctionOpInterface>()) {
    if (!triton::isKernel(function))
      continue;
    uint32_t frame = regions->getOperationId(function.getOperation());

    function.walk([&](Operation *op) {
      for (const auto &access : triton::getMemoryAccesses(op)) {
        Value value = access.value;
        if (!access.isShared() || footprints.contains(value))
          continue;
        const auto &info = regions->getRegionInfo(value);
        // BufferRegion joins callee arguments across call sites, so a returned
        // descriptor can include views from another caller's allocation frame.
        if (info.kind != triton::RegionInfo::Kind::Exact ||
            info.views.empty() ||
            llvm::any_of(info.views, [&](const auto &view) {
              return view.allocationFrame != frame;
            }))
          continue;
        // Union all possible origins. Merging CTA address sets only adds
        // aliases.
        auto &footprint = footprints[value];
        for (const auto &view : info.views)
          for (const auto &entry : view.region.ctaAddresses)
            footprint.insert(entry.second);
      }
    });
  }
  return footprints;
}

void ModuleMembarAnalysis::run() {
  auto physicalFootprints = getSharedMemoryFootprints();
  runAnalysis<MembarAnalysis>(physicalFootprints);
}
} // namespace mlir
