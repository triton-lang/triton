#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include <optional>

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {

static triton::gpu::MemDescType getRootAllocationType(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (auto alloc = dyn_cast<triton::gpu::LocalAllocOp>(def))
      return alloc.getType();
    if (isa<triton::gpu::MemDescTransOp, triton::gpu::MemDescReshapeOp,
            triton::gpu::MemDescReinterpretOp>(def)) {
      value = def->getOperand(0);
      continue;
    }
    return {};
  }
  return {};
}

static std::optional<uint64_t> linearizeIndex(ArrayRef<int64_t> coordinates,
                                              ArrayRef<int64_t> shape,
                                              ArrayRef<unsigned> order) {
  if (coordinates.size() != shape.size() || order.size() != shape.size())
    return std::nullopt;
  uint64_t linear = 0;
  uint64_t stride = 1;
  for (unsigned dim : order) {
    if (coordinates[dim] < 0 || coordinates[dim] >= shape[dim])
      return std::nullopt;
    linear += coordinates[dim] * stride;
    stride *= shape[dim];
  }
  return linear;
}

static std::optional<SmallVector<int64_t>>
delinearizeIndex(uint64_t linear, ArrayRef<int64_t> shape,
                 ArrayRef<unsigned> order) {
  if (order.size() != shape.size())
    return std::nullopt;
  SmallVector<int64_t> coordinates(shape.size());
  for (unsigned dim : order) {
    coordinates[dim] = linear % shape[dim];
    linear /= shape[dim];
  }
  if (linear != 0)
    return std::nullopt;
  return coordinates;
}

static bool canonicalizeSubslice(triton::gpu::MemDescSubsliceOp subslice,
                                 SmallVectorImpl<int64_t> &canonicalBounds) {
  auto accessTy = subslice.getType();
  auto sourceTy = subslice.getSrc().getType();
  auto rootTy = getRootAllocationType(subslice.getSrc());
  if (!rootTy || sourceTy.getEncoding() != rootTy.getEncoding() ||
      triton::gpu::getPaddedEncoding(sourceTy.getEncoding()) ||
      isa<triton::gpu::PartitionedSharedEncodingAttr>(sourceTy.getEncoding()))
    return false;

  auto sourceLayout =
      dyn_cast<triton::gpu::SharedEncodingTrait>(sourceTy.getEncoding());
  auto rootLayout =
      dyn_cast<triton::gpu::SharedEncodingTrait>(rootTy.getEncoding());
  if (!sourceLayout || !rootLayout)
    return false;

  auto sourceAllocShape = sourceTy.getAllocShape();
  auto rootAllocShape = rootTy.getAllocShape();
  auto sourceOrder = triton::gpu::getOrder(sourceLayout, sourceAllocShape);
  auto rootOrder = triton::gpu::getOrder(rootLayout, rootAllocShape);
  if (sourceOrder.size() != sourceAllocShape.size() ||
      rootOrder.size() != rootAllocShape.size())
    return false;

  SmallVector<int64_t> offsets(subslice.getOffsets());
  SmallVector<int64_t> last(offsets);
  uint64_t numElements = 1;
  for (auto [i, extent] : llvm::enumerate(accessTy.getShape())) {
    last[i] += extent - 1;
    numElements *= extent;
  }
  auto firstLinear = linearizeIndex(offsets, sourceAllocShape, sourceOrder);
  auto lastLinear = linearizeIndex(last, sourceAllocShape, sourceOrder);
  if (!firstLinear || !lastLinear || *lastLinear < *firstLinear ||
      *lastLinear - *firstLinear + 1 != numElements)
    return false;

  unsigned sourceBits = sourceTy.getElementTypeBitWidth();
  unsigned rootBits = rootTy.getElementTypeBitWidth();
  uint64_t firstBit = *firstLinear * sourceBits;
  uint64_t numBits = numElements * sourceBits;
  if (firstBit % rootBits != 0 || numBits % rootBits != 0)
    return false;

  uint64_t rootFirst = firstBit / rootBits;
  uint64_t rootNumElements = numBits / rootBits;
  auto rootStart = delinearizeIndex(rootFirst, rootAllocShape, rootOrder);
  auto rootEnd = delinearizeIndex(rootFirst + rootNumElements - 1,
                                  rootAllocShape, rootOrder);
  if (!rootStart || !rootEnd)
    return false;

  SmallVector<int64_t> normalizedShape(rootAllocShape.size());
  uint64_t normalizedElements = 1;
  for (size_t i = 0; i < rootAllocShape.size(); ++i) {
    normalizedShape[i] = (*rootEnd)[i] - (*rootStart)[i] + 1;
    if (normalizedShape[i] <= 0)
      return false;
    normalizedElements *= normalizedShape[i];
  }
  if (normalizedElements != rootNumElements)
    return false;

  canonicalBounds.assign(rootStart->begin(), rootStart->end());
  for (auto [start, extent] : llvm::zip(*rootStart, normalizedShape))
    canonicalBounds.push_back(start + extent);
  return true;
}

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
      canonicalizeSubslice(subslice, canonicalBounds);
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

  if (bufferId == other.bufferId && bufferId != Allocation::InvalidBufferId) {
    ArrayRef<int64_t> offsetsA = subsliceOffsets;
    ArrayRef<int64_t> offsetsB = other.subsliceOffsets;
    ArrayRef<int64_t> shapeA = accessTy.getShape();
    ArrayRef<int64_t> shapeB = other.accessTy.getShape();

    // Prefer bounds normalized to the original allocation shape. This keeps
    // offsets comparable across reinterpret/reshape view shapes.
    if (!canonicalBounds.empty() && !other.canonicalBounds.empty() &&
        canonicalBounds.size() == other.canonicalBounds.size()) {
      size_t rank = canonicalBounds.size() / 2;
      auto startsA = ArrayRef<int64_t>(canonicalBounds).take_front(rank);
      auto endsA = ArrayRef<int64_t>(canonicalBounds).drop_front(rank);
      auto startsB = ArrayRef<int64_t>(other.canonicalBounds).take_front(rank);
      auto endsB = ArrayRef<int64_t>(other.canonicalBounds).drop_front(rank);
      for (size_t i = 0; i < rank; ++i) {
        if (endsA[i] <= startsB[i] || endsB[i] <= startsA[i])
          return false;
      }
      return true;
    } else if (accessTy.getAllocShape() != other.accessTy.getAllocShape() ||
               accessTy.getEncoding() != other.accessTy.getEncoding()) {
      return true;
    }

    // Check if all subslice region dimensions have some intersection.
    // [offsetA, offsetA + shape) and [offsetB, offsetB + other.shape)
    // If any dimension doesn't intersect, we are looking at disjoint slices.
    for (size_t i = 0; i < offsetsA.size(); ++i) {
      int64_t startA = offsetsA[i];
      int64_t endA = startA + shapeA[i];
      int64_t startB = offsetsB[i];
      int64_t endB = startB + shapeB[i];

      if (endA <= startB || endB <= startA)
        return false;
    }
    return true;
  }

  // Distinct allocations may reuse the same physical interval, but their
  // logical origins are unrelated.
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
