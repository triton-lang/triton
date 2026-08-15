#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
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
    : allocationInterval(allocationInterval), bufferId(bufferId) {
  auto accessTy = cast<triton::gpu::MemDescType>(value.getType());
  this->accessTy = accessTy;

  // Get the memdesc_subslice information if present. If no subslice is
  // present the whole interval is accessed
  if (auto subslice = value.getDefiningOp<triton::gpu::MemDescSubsliceOp>()) {
    // We know there aren't subslices before the one because of subslice::fold
    // Still need to check this for where a fold isn't possible (control flow)
    // and when a subslice is carried in a loop
    if (accessTy.getAllocShape() == subslice.getSrc().getType().getShape()) {
      subsliceOffsets = SmallVector<int64_t>(subslice.getOffsets());
    }
  }
}

bool AllocationSlice::intersects(const AllocationSlice &other) const {
  // Disjoint intervals don't overlap
  if (!allocationInterval.intersects(other.allocationInterval))
    return false;

  // For slices of the same allocation, compare dynamic buffer indices to prove
  // that different slots do not overlap.
  if (bufferId == other.bufferId && bufferId != Allocation::InvalidBufferId &&
      areBufferIndicesProvablyDifferent(*this, other))
    return false;

  // If access types are unknown, assume intersection
  if (!accessTy || !other.accessTy)
    return true;

  // If offsets are unknown, conservatively assume overlap
  if (subsliceOffsets.empty() || other.subsliceOffsets.empty())
    return true;

  // If layouts differ, we assume intersection as we currently only work on
  // logical elements
  if (accessTy.getEncoding() != other.accessTy.getEncoding())
    return true;

  auto shapeA = SmallVector<int64_t>(accessTy.getShape());
  auto shapeB = SmallVector<int64_t>(other.accessTy.getShape());
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

void MembarAnalysis::insertBarrier(Operation *op, OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
  triton::gpu::BarrierOp::create(*builder, op->getLoc(),
                                 triton::gpu::AddrSpace::Local);
}

bool containsLocalBarrier(Operation *op) {
  if (isa<triton::AtomicPollOp>(op))
    return true;
  if (auto atomic = dyn_cast<triton::AtomicOpInterface>(op))
    return atomic.getMemSemantic() != triton::MemSemantic::RELAXED;
  if (isa<gpu::BarrierOp>(op))
    return true;
  if (isa<ttng::ClusterBarrierOp>(op))
    return true;
  if (isa<triton::gpu::WarpSpecializePartitionsOp>(op))
    return true;
  if (isa<ttng::ArriveBarrierOp>(op))
    return true;
  if (isa<ttng::BarrierExpectOp>(op))
    return true;
  if (isa<ttng::TCGen5CommitOp>(op))
    return true;
  if (auto barrier = dyn_cast<triton::gpu::BarrierOp>(op))
    return barrier.hasLocal();
  if (auto wgWait = dyn_cast<ttng::WarpGroupDotWaitOp>(op))
    return !wgWait.getWarpGroupLocal() && triton::gpu::lookupNumWarps(op) > 4;
  return false;
}

static Allocation::BufferId getScratchBufferId(Operation *op,
                                               Allocation *allocation) {
  // A call's allocation belongs to the callee and is translated separately.
  if (isa<triton::CallOp>(op))
    return Allocation::InvalidBufferId;
  return allocation->getBufferId(op);
}

static bool scratchBufferUsesWarpSync(Operation *op) {
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
                                                   Allocation *allocation) {
  triton::BarrierStages stages;
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

  // Leaving the default region of a `ttg.warp_specialize` goes through the
  // CTA-wide barrier `lowerWarpSpecializeCommon` emits at every
  // `ttg.warp_yield`, which is the rendezvous with the barrier each partition
  // executes on `ttg.warp_return`, so shared memory effects inside the region
  // are synchronized before the code after the op runs.
  if (isa<triton::gpu::WarpYieldOp>(op))
    stages.beforeMemoryEffects = true;
  else
    stages.beforeMemoryEffects = containsLocalBarrier(op);
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
                                     BlockInfo *blockInfo) {
  if (bufferIndexAnalysis.isBackedgeSuccessor(terminator, successor))
    bufferIndexAnalysis.invalidateBufferIndices(*blockInfo);
}

void MembarAnalysis::updateExitState(BlockInfo *blockInfo) {
  // Function summaries are reused at every call site, so per-function SSA
  // index identity is no longer meaningful.
  bufferIndexAnalysis.invalidateBufferIndices(*blockInfo);
}

void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
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
    blockInfo->sync();
  }

  // If the current op is an (async) memory wait and there is no later sync
  // point before memory is accessed, insert a barrier op and sync. This avoids
  // redundant barriers by deferring the barrier to the later sync point.
  if (op->hasTrait<mlir::OpTrait::MemWaitOpTrait>() &&
      !hasSyncPointBeforeMemoryEffect(op, &allocation)) {
    builder->setInsertionPointAfter(op);
    insertBarrier(op, builder);
    blockInfo->sync();
    return;
  }

  BlockInfo curBlockInfo;
  auto scratchBufferId = getScratchBufferId(op, &allocation);
  if (isa<triton::CallOp>(op)) {
    // Inter-function dependencies
    auto callOpInterface = dyn_cast<CallOpInterface>(op);
    if (auto callee =
            dyn_cast<FunctionOpInterface>(callOpInterface.resolveCallable())) {
      auto calleeBlockInfo = funcMap->lookup(callee);
      auto callBufferId = allocation.getBufferId(op);
      size_t callOffset = 0;
      if (callBufferId != Allocation::InvalidBufferId)
        callOffset = allocation.getAllocatedInterval(callBufferId).start();
      curBlockInfo = translateBlockInfoToCallsite(calleeBlockInfo, callOffset);
    }
  } else {
    // Intra-function dependencies
    if (auto memoryEffectOpInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Explicit buffer
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
          effectInstances;
      memoryEffectOpInterface.getEffects(effectInstances);
      for (auto effectInstance : effectInstances) {
        if (auto value = effectInstance.getValue()) {
          for (auto bufferId : allocation.getAllBufferIdsWithAliases(value)) {
            if (bufferId != Allocation::InvalidBufferId) {
              auto interval = allocation.getAllocatedInterval(bufferId);
              auto slice =
                  bufferIndexAnalysis.makeSlice(value, interval, bufferId);

              if (isa<MemoryEffects::Write>(effectInstance.getEffect()))
                curBlockInfo.syncWriteSlices[slice].insert(op);
              else if (isa<MemoryEffects::Read>(effectInstance.getEffect()))
                curBlockInfo.syncReadSlices[slice].insert(op);
            }
          }
        }
      }
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
        blockInfo->isIntersected(curBlockInfo, filter, &allocation);
    if (insertCTABarrier) {
      builder->setInsertionPoint(op);
      insertBarrier(op, builder);
    }
    if (insertCTABarrier)
      blockInfo->sync();

    if (barrierStages.betweenMemoryEffects) {
      // The internal barrier synchronizes all incoming effects. Do not carry
      // them past the operation; only effects after the barrier are outgoing.
      blockInfo->join(curBlockInfo);
      blockInfo->sync();
      curBlockInfo.sync();
    }
    curBlockInfo.syncReadSlices[scratchSlice].insert(op);
  } else if (blockInfo->isIntersected(curBlockInfo, filter, &allocation)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    blockInfo->sync();
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  blockInfo->join(curBlockInfo);

  if (barrierStages.afterMemoryEffects) {
    // Model a trailing local barrier after handling the operation's effects.
    blockInfo->sync();
  }
}
} // namespace mlir
