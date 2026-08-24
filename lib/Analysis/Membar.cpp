#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/Support/MathExtras.h"

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
      subsliceSource = subslice.getSrc();
    }
  }
}

bool AllocationSlice::intersects(const AllocationSlice &other) const {
  // Disjoint intervals don't overlap
  if (!allocationInterval.intersects(other.allocationInterval))
    return false;

  // A MAY-set covers every runtime origin, including across loop iterations.
  // Require disjointness for every pair in the same physical allocation frame.
  // Different unnormalized frames are unknown, not evidence of disjointness.
  if (physicalFootprint && other.physicalFootprint &&
      !physicalFootprint->empty() && !other.physicalFootprint->empty() &&
      llvm::all_of(*physicalFootprint, [&](const auto &lhs) {
        return llvm::all_of(*other.physicalFootprint, [&](const auto &rhs) {
          return lhs.allocationFrame == rhs.allocationFrame &&
                 !lhs.intersects(rhs);
        });
      }))
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

  // Logical offsets share coordinates only for the same source descriptor
  // in the same dynamic iteration. Matching encodings or allocation IDs
  // do not establish this after selection, indexing or reinterpretation.
  if (!subsliceSource || subsliceSource != other.subsliceSource)
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
  if (isa<triton::gpu::WarpYieldOp, triton::gpu::WarpReturnOp>(op))
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
                                     MembarInfo *membarInfo) {
  if (bufferIndexAnalysis.isBackedgeSuccessor(terminator, successor)) {
    bufferIndexAnalysis.invalidateBufferIndices(membarInfo->pending);
    bufferIndexAnalysis.invalidateBufferIndices(membarInfo->entryBlockInfo);
  }
}

void MembarAnalysis::updateExitState(MembarInfo *membarInfo) {
  // Function summaries are reused at every call site, so per-function SSA
  // index identity is no longer meaningful.
  bufferIndexAnalysis.invalidateBufferIndices(membarInfo->pending);
  bufferIndexAnalysis.invalidateBufferIndices(membarInfo->entryBlockInfo);
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
  } else {
    // Intra-function dependencies
    if (auto memoryEffectOpInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Explicit buffer
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
          effectInstances;
      memoryEffectOpInterface.getEffects(effectInstances);
      auto getPhysicalFootprint =
          [&](Value value) -> const triton::RegionInfo::ViewList * {
        auto found = physicalFootprints.find(value);
        if (found == physicalFootprints.end())
          return nullptr;

        // Only these operations have a complete access image given by the
        // existing block-local register-to-shared conversion. Other effects,
        // including asynchronous instructions, keep the allocation fallback.
        RankedTensorType registerType;
        if (auto load = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
          if (value == load.getSrc())
            registerType = load.getType();
        } else if (auto store = dyn_cast<triton::gpu::LocalStoreOp>(op)) {
          if (value == store.getDst())
            registerType = store.getSrc().getType();
        } else if (auto alloc = dyn_cast<triton::gpu::LocalAllocOp>(op)) {
          if (alloc.getSrc() && value == alloc.getResult())
            registerType = alloc.getSrc().getType();
        }
        if (!registerType ||
            !isa_and_nonnull<triton::gpu::DistributedEncodingTrait>(
                registerType.getEncoding()))
          return nullptr;

        auto type = cast<triton::gpu::MemDescType>(value.getType());
        auto encoding = type.getEncoding();
        if (!isa_and_nonnull<triton::gpu::SharedEncodingTrait>(encoding) ||
            triton::gpu::getPaddedEncoding(encoding) ||
            isa<triton::gpu::PartitionedSharedEncodingAttr>(encoding))
          return nullptr;
        unsigned bitWidth = type.getElementType().getIntOrFloatBitWidth();
        if (bitWidth < 8 || !llvm::isPowerOf2_32(bitWidth) ||
            llvm::any_of(
                type.getShape(),
                [](int64_t size) {
                  return size <= 0 ||
                         size > std::numeric_limits<int32_t>::max() ||
                         !llvm::isPowerOf2_64(size);
                }))
          return nullptr;

        // BufferRegion inverts the descriptor's storage layout and restricts
        // it to the logical shape. Prove that the complete instruction access
        // fits inside that image before using it as a physical MAY-footprint.
        auto storage = triton::gpu::toLinearLayout(type);
        auto offset = StringAttr::get(op->getContext(), "offset");
        auto block = StringAttr::get(op->getContext(), "block");
        if (storage.getNumOutDims() != type.getRank() ||
            storage.getNumInDims() != 2 || !storage.hasInDim(offset) ||
            !storage.hasInDim(block) || storage.getInDimSize(block) != 1 ||
            !storage.isSurjective() ||
            storage.getTotalInDimSizeLog2() + storage.getTotalOutDimSizeLog2() >
                64 ||
            storage.getInDimSizeLog2(offset) + llvm::Log2_32(bitWidth / 8) >=
                31)
          return nullptr;
        auto footprint = storage.pseudoinvert();
        auto logicalDims = llvm::to_vector(footprint.getInDimNames());
        for (auto [dim, size] : llvm::zip_equal(logicalDims, type.getShape()))
          footprint = footprint.resizeInDim(dim, size);
        auto byte = StringAttr::get(op->getContext(), "byte");
        footprint =
            triton::LinearLayout::identity1D(bitWidth / 8, byte, offset) *
            footprint;

        auto reg = StringAttr::get(op->getContext(), "register");
        auto registerLayout = triton::gpu::toLinearLayout(registerType)
                                  .removeZeroBasesAlongDim(reg);
        auto access =
            triton::invertAndComposeBlockLocal(storage, registerLayout);
        access = triton::LinearLayout::identity1D(bitWidth / 8, reg, offset) *
                 access;
        return triton::lstsq(footprint, access) ? &found->second : nullptr;
      };
      for (auto effectInstance : effectInstances) {
        if (auto value = effectInstance.getValue()) {
          auto *footprint = getPhysicalFootprint(value);
          for (auto bufferId : allocation.getAllBufferIdsWithAliases(value)) {
            if (bufferId != Allocation::InvalidBufferId) {
              auto interval = allocation.getAllocatedInterval(bufferId);
              auto slice =
                  bufferIndexAnalysis.makeSlice(value, interval, bufferId);
              slice.physicalFootprint = footprint;

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
  // Physical footprints need allocated addresses. Standalone analyses may lack
  // those attributes; retain the coarse allocation intervals in that case.
  bool canAnalyzeFootprints = true;
  module.walk([&](Operation *op) {
    if (isa<triton::gpu::LocalAllocOp>(op))
      canAnalyzeFootprints &= op->hasAttr("allocation.offset");
    if (isa<ttng::TMEMAllocOp>(op))
      canAnalyzeFootprints &= op->hasAttr("tensor_memory_col_offset") &&
                              op->hasAttr("tensor_memory_row_offset");
    // BufferRegion currently describes byte-addressed numeric shared
    // storage, not pointers or the widened storage of sub-byte elements.
    // Check views too: reinterpretation can change an allocation's element.
    for (Type type :
         llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
      auto memory = dyn_cast<triton::gpu::MemDescType>(type);
      if (!memory ||
          !isa<triton::gpu::SharedMemorySpaceAttr>(memory.getMemorySpace()))
        continue;
      Type element = memory.getElementType();
      canAnalyzeFootprints &= isa<IntegerType, FloatType>(element) &&
                              element.getIntOrFloatBitWidth() % 8 == 0;
    }
  });
  if (!canAnalyzeFootprints)
    return footprints;

  auto solver = createDataFlowSolver();
  auto *regions = solver->load<triton::BufferRegionAnalysis>();
  if (failed(solver->initializeAndRun(module)))
    return footprints;
  // Absolute addresses must belong to a common physical frame. Device-function
  // frames and returned descriptors need callsite translation before comparing
  // their offsets; use coarse allocation intervals until that is supported.
  module.walk([&](Operation *op) {
    auto function = op->getParentOfType<FunctionOpInterface>();
    if (!function || !triton::isKernel(function) ||
        triton::gpu::lookupNumCTAs(op) != 1)
      return;
    auto consider = [&](Value value) {
      auto type = dyn_cast<triton::gpu::MemDescType>(value.getType());
      if (!type ||
          !isa<triton::gpu::SharedMemorySpaceAttr>(type.getMemorySpace()))
        return;
      const auto &info = regions->getRegionInfo(value);
      if (info.kind != triton::RegionInfo::Kind::Exact || info.views.empty() ||
          llvm::any_of(info.views, [&](const auto &view) {
            return view.allocationFrame !=
                   regions->getOperationId(function.getOperation());
          }))
        return;
      // Preserve every possible runtime stage and origin, not only singletons.
      footprints.try_emplace(value, info.views);
    };
    for (Value value : op->getOperands())
      consider(value);
    for (Value value : op->getResults())
      consider(value);
  });
  return footprints;
}

void ModuleMembarAnalysis::run() {
  auto physicalFootprints = getSharedMemoryFootprints();
  runAnalysis<MembarAnalysis>(physicalFootprints);
}
} // namespace mlir
