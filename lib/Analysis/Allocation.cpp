#include "triton/Analysis/Allocation.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <numeric>

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Alias.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

// Debug defines
#define DEBUG_TYPE "allocate-shared-memory"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define LLDBG_TAG(DEPTH, TAG)                                                  \
  llvm::dbgs() << llvm::right_justify("", DEPTH * 4) << TAG                    \
               << llvm::left_justify(": ", 24 - (DEPTH * 4) - strlen(TAG))

#define LLDBG_L0(TAG, VAL) LLDBG_TAG(0, TAG) << VAL << "\n"
#define LLDBG_L0V(TAG, VAL) LLVM_DEBUG(LLDBG_L0(TAG, VAL))

#define LLDBG_L1(TAG) LLDBG_TAG(1, TAG) << "\n"
#define LLDBG_L1V(TAG, VAL) LLDBG_TAG(1, TAG) << VAL << "\n"
#define LLDBG_L1P(TAG) LLDBG_TAG(1, TAG)

#define LLDBG_L2(TAG, VAL) LLDBG_TAG(2, TAG) << VAL << "\n"
#define LLDBG_L2V(TAG, VAL) LLVM_DEBUG(LLDBG_L2(TAG, VAL))
#define LLDBG_L2C(TAG, CONTAINER)                                              \
  LLDBG_TAG(2, TAG);                                                           \
  llvm::interleaveComma(CONTAINER, llvm::dbgs(),                               \
                        [](const auto &r) { llvm::dbgs() << r; });             \
  llvm::dbgs() << "\n"
#define LLDBG_L2C_ID(TAG, CONTAINER)                                           \
  LLDBG_TAG(2, TAG);                                                           \
  llvm::interleaveComma(CONTAINER, llvm::dbgs(),                               \
                        [](const auto &b) { llvm::dbgs() << b->id; });         \
  llvm::dbgs() << "\n"

namespace mlir {

namespace {
// Debug utilities

void printValue(mlir::Value v) {
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Block *block = arg.getOwner();
    const auto &blks = block->getParent()->getBlocks();
    int32_t blockIdx = 0;
    auto itr = llvm::find_if(blks, [&](const Block &blk) {
      blockIdx++;
      return block == &blk;
    });
    if (itr != blks.end())
      llvm::dbgs() << "^bb" << blockIdx << ": ";
  }
  v.print(llvm::dbgs());
  llvm::dbgs() << "\n";
}
} // namespace

//===----------------------------------------------------------------------===//
// Shared Memory Allocation Analysis
//===----------------------------------------------------------------------===//
namespace triton {

// Bitwidth of pointers
constexpr int kPtrBitWidth = 64;
// Max shmem LDS/STS instruction in bits
constexpr int kMaxShmemVecBitLength = 128;

static SmallVector<unsigned> getRepShapeForCvt(RankedTensorType srcTy,
                                               RankedTensorType dstTy) {
  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();

  if (!cvtNeedsSharedMemory(srcTy, dstTy)) {
    return {};
  }

  if (shouldUseDistSmem(srcLayout, dstLayout)) {
    // TODO: padding to avoid bank conflicts
    return convertType<unsigned, int64_t>(gpu::getShapePerCTA(srcTy));
  }

  assert(srcLayout && dstLayout && "Unexpected layout in getRepShapeForCvt()");

  auto srcShapePerCTA = gpu::getShapePerCTA(srcTy);
  auto dstShapePerCTA = gpu::getShapePerCTA(dstTy);
  auto srcShapePerCTATile = gpu::getShapePerCTATile(srcTy);
  auto dstShapePerCTATile = gpu::getShapePerCTATile(dstTy);

  assert(srcTy.getRank() == dstTy.getRank() &&
         "src and dst must have the same rank");

  unsigned rank = dstTy.getRank();
  SmallVector<unsigned> repShape(rank);
  for (unsigned d = 0; d < rank; ++d) {
    repShape[d] =
        std::max(std::min<unsigned>(srcShapePerCTA[d], srcShapePerCTATile[d]),
                 std::min<unsigned>(dstShapePerCTA[d], dstShapePerCTATile[d]));
  }
  return repShape;
}

// Both `atomic_cas` and `atomic_rmw need a single scratch element if returning
// a scalar value because Triton's block-based programming model ensures that
// all threads in each block see the same return value, even those threads that
// do not participate in the atomic operation
static SmallVector<unsigned> getRepShapeForAtomic(Value result) {
  SmallVector<unsigned> smemShape;
  if (atomicNeedsSharedMemory(result)) {
    smemShape.push_back(1);
  }
  return smemShape;
}

std::pair<unsigned, unsigned>
getScratchCvtInOutVecLengths(RankedTensorType srcTy, RankedTensorType dstTy) {
  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();

  auto srcLinAttr = gpu::toLinearEncoding(srcLayout, srcTy.getShape());
  auto dstLinAttr = gpu::toLinearEncoding(dstLayout, dstTy.getShape());
  auto inOrd = srcLinAttr.getOrder();
  auto outOrd = dstLinAttr.getOrder();

  unsigned rank = srcTy.getRank();

  unsigned srcContigPerThread = srcLinAttr.getContigPerThread()[inOrd[0]];
  unsigned dstContigPerThread = dstLinAttr.getContigPerThread()[outOrd[0]];
  // TODO: Fix the legacy issue that outOrd[0] == 0 always means
  //       that we cannot do vectorization.
  unsigned innerDim = rank - 1;
  unsigned inVec = outOrd[0] != innerDim  ? 1
                   : inOrd[0] != innerDim ? 1
                                          : srcContigPerThread;
  unsigned outVec = outOrd[0] != innerDim ? 1 : dstContigPerThread;

  if (isa<gpu::NvidiaMmaEncodingAttr>(srcLayout) &&
      isa<gpu::BlockedEncodingAttr>(dstLayout)) {
    // when storing from mma layout and loading in blocked layout vectorizing
    // the load back gives better performance even if there is a
    // transposition.
    outVec = dstContigPerThread;
  }
  return {inVec, outVec};
}

ScratchConfig getScratchConfigForCvt(RankedTensorType srcTy,
                                     RankedTensorType dstTy) {
  // Initialize vector sizes and stride
  auto repShape = getRepShapeForCvt(srcTy, dstTy);
  if (repShape.empty())
    return ScratchConfig({}, {});
  ScratchConfig scratchConfig(repShape, repShape);
  auto rank = repShape.size();
  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();

  assert(cvtNeedsSharedMemory(srcTy, dstTy));
  auto outOrd = gpu::toLinearEncoding(dstLayout, dstTy.getShape()).getOrder();
  scratchConfig.order = outOrd;

  std::tie(scratchConfig.inVec, scratchConfig.outVec) =
      getScratchCvtInOutVecLengths(srcTy, dstTy);
  // We can't write a longer vector than the shape of shared memory.
  // This shape might be smaller than the tensor shape in case we decided to
  // do the conversion in multiple iterations.
  unsigned contiguousShapeDim = scratchConfig.repShape[scratchConfig.order[0]];
  scratchConfig.inVec = std::min(scratchConfig.inVec, contiguousShapeDim);
  scratchConfig.outVec = std::min(scratchConfig.outVec, contiguousShapeDim);
  // Clamp the vector length to kMaxShmemVecBitLength / element bitwidth as this
  // is the max vectorisation
  auto inBitWidth = isa<PointerType>(srcTy.getElementType())
                        ? kPtrBitWidth
                        : srcTy.getElementTypeBitWidth();
  auto outBitWidth = isa<PointerType>(dstTy.getElementType())
                         ? kPtrBitWidth
                         : dstTy.getElementTypeBitWidth();
  scratchConfig.inVec =
      std::min(scratchConfig.inVec, kMaxShmemVecBitLength / inBitWidth);
  scratchConfig.outVec =
      std::min(scratchConfig.outVec, kMaxShmemVecBitLength / outBitWidth);

  // No padding is required if the tensor is 1-D, or if all dimensions except
  // the first accessed dimension have a size of 1.
  if (rank <= 1 || product(repShape) == repShape[outOrd[0]])
    return scratchConfig;

  auto paddedSize = std::max(scratchConfig.inVec, scratchConfig.outVec);
  scratchConfig.paddedRepShape[outOrd[0]] += paddedSize;
  return scratchConfig;
}

unsigned defaultAllocationAnalysisScratchSizeFn(Operation *op) {
  if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    ReduceOpHelper helper(reduceOp);
    return helper.getScratchSizeInBytes();
  }
  if (auto scanOp = dyn_cast<ScanOp>(op)) {
    ScanLoweringHelper helper(scanOp);
    return helper.getScratchSizeInBytes();
  }
  if (auto gatherOp = dyn_cast<GatherOp>(op)) {
    GatherLoweringHelper helper(gatherOp);
    return helper.getScratchSizeInBytes();
  }
  if (auto histogram = dyn_cast<HistogramOp>(op)) {
    auto dstTy = histogram.getType();
    int threadsPerWarp = gpu::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    return std::max<int>(dstTy.getNumElements(), threadsPerWarp) *
           std::max<int>(8, dstTy.getElementTypeBitWidth()) / 8;
  }
  if (auto cvtLayout = dyn_cast<gpu::ConvertLayoutOp>(op)) {
    auto srcTy = cvtLayout.getSrc().getType();
    auto dstTy = cvtLayout.getType();
    auto srcEncoding = srcTy.getEncoding();
    auto dstEncoding = dstTy.getEncoding();
    if (mlir::isa<gpu::SharedEncodingTrait>(srcEncoding) ||
        mlir::isa<gpu::SharedEncodingTrait>(dstEncoding)) {
      // Conversions from/to shared memory do not need scratch memory.
      return 0;
    }
    // ConvertLayoutOp with both input/output non-shared_layout
    // TODO: Besides of implementing ConvertLayoutOp via shared memory, it's
    //       also possible to realize it with other approaches in restricted
    //       conditions, such as warp-shuffle
    auto scratchConfig = getScratchConfigForCvt(srcTy, dstTy);
    auto elems = getNumScratchElements(scratchConfig.paddedRepShape);
    return isa<PointerType>(srcTy.getElementType())
               ? elems * kPtrBitWidth / 8
               : elems * std::max<int>(8, srcTy.getElementTypeBitWidth()) / 8;
  }
  if (isa<AtomicRMWOp, AtomicCASOp>(op)) {
    auto value = op->getOperand(0);
    // only scalar requires scratch memory
    // make it explicit for readability
    if (dyn_cast<RankedTensorType>(value.getType())) {
      return 0;
    }
    auto smemShape = getRepShapeForAtomic(op->getResult(0));
    auto elems = getNumScratchElements(smemShape);
    auto elemTy = cast<PointerType>(value.getType()).getPointeeType();
    assert(!isa<PointerType>(elemTy) && "unexpected pointer type");
    return elems * std::max<int>(8, elemTy.getIntOrFloatBitWidth()) / 8;
  }
  if (isa<ExperimentalTensormapCreateOp>(op)) {
    constexpr int32_t kTMASize = 128;
    return kTMASize;
  }
  return 0;
}

class AllocationAnalysis {
public:
  AllocationAnalysis(Operation *operation,
                     Allocation::FuncAllocMapT *funcAllocMap,
                     Allocation *allocation,
                     AllocationAnalysisScratchSizeFn scratchSizeGetter)
      : operation(operation), funcAllocMap(funcAllocMap),
        allocation(allocation), scratchSizeGetter(scratchSizeGetter) {
    run();
  }

private:
  using BufferT = Allocation::BufferT;

  /// Value -> Liveness Range
  using IntervalT = Interval<size_t>;
  /// Use MapVector to ensure determinism.
  using BufferRangeMapT = llvm::MapVector<BufferT *, IntervalT>;
  /// Nodes -> Nodes
  using GraphT = DenseMap<BufferT *, DenseSet<BufferT *>>;

  /// Set of Liveness Intervals
  /// - Helper to capture and track disjoint intervals
  class LiveIntervals {
  public:
    LiveIntervals() = default;
    LiveIntervals(const LiveIntervals &) = default;

    void push_back(const IntervalT &ival) { intervals.push_back(ival); }
    auto begin() const { return intervals.begin(); }
    auto end() const { return intervals.end(); }
    auto size() const { return intervals.size(); }
    auto empty() const { return intervals.empty(); }

    void sortAndJoin() {
      if (size() <= 1)
        return;
      llvm::sort(intervals, [](const auto &lhs, const auto &rhs) {
        return lhs.start() <= rhs.start();
      });
      SmallVector<IntervalT> newIntervals;
      IntervalT interval = intervals.front();
      for (auto &I : *this) {
        if (interval.adjacent(I) || interval.intersects(I))
          interval = interval.span(I);
        else {
          newIntervals.push_back(interval);
          interval = I;
        }
      }
      newIntervals.push_back(interval);
      intervals.assign(newIntervals);
    }
    IntervalT span() const {
      assert(!empty());
      IntervalT interval = intervals.front();
      for (auto &I : intervals)
        interval = interval.span(I);
      return interval;
    }

  private:
    SmallVector<IntervalT> intervals;
  };

  typedef function_ref<LiveIntervals(Value value)> LivenessF;

  void run() {
    getValuesAndSizes();
    resolveLiveness();
    computeOffsets();
    LLVM_DEBUG(dump());
  }

  /// Initializes explicitly defined shared memory values for a given operation.
  void getExplicitValueSize(Operation *op) {
    auto alloc = dyn_cast<gpu::LocalAllocOp>(op);
    if (!alloc || !alloc.isSharedMemoryAlloc())
      return;
    // Bytes could be a different value once we support padding or other
    // allocation policies.
    auto allocType = alloc.getType();
    auto shapePerCTA = gpu::getAllocationShapePerCTA(allocType);
    auto bytes =
        product<int64_t>(shapePerCTA) * allocType.getElementTypeBitWidth() / 8;

    auto alignment = alloc.getAlignmentOrDefault();
    allocation->addBuffer<BufferT::BufferKind::Explicit>(alloc, bytes,
                                                         alignment);
  }

  template <BufferT::BufferKind T>
  void maybeAddScratchBuffer(Operation *op, unsigned bytes,
                             unsigned alignment) {
    if (bytes > 0)
      allocation->addBuffer<T>(op, bytes, alignment);
  }

  /// Initializes temporary shared memory for a given operation.
  void getScratchValueSize(Operation *op) {
    constexpr size_t scratchAlignment = 128;
    if (auto callOp = dyn_cast<CallOpInterface>(op)) {
      auto callable = callOp.resolveCallable();
      auto funcOp = dyn_cast<FunctionOpInterface>(callable);
      auto *funcAlloc = &(*funcAllocMap)[funcOp];
      auto bytes = funcAlloc->getSharedMemorySize();
      maybeAddScratchBuffer<BufferT::BufferKind::Virtual>(op, bytes,
                                                          scratchAlignment);
      return;
    }
    if (auto ws = dyn_cast<gpu::WarpSpecializeOp>(op)) {
      // `ttg.warp_specialize` needs memory to pass its explicit captures. Pack
      // the captures like a struct.
      auto [captureSize, captureAlign] = ws.getCaptureSizeAlign();
      maybeAddScratchBuffer<BufferT::BufferKind::Scratch>(op, captureSize,
                                                          captureAlign);
      return;
    }
    if (auto func = dyn_cast<FunctionOpInterface>(op)) {
      unsigned numWarpIndices = 0;
      // Warp specialization communicates states over shared memory to each
      // warp. Add space for an i8 for each warpgroup warp.
      func.walk([&](gpu::WarpSpecializeOp op) {
        numWarpIndices = std::max(numWarpIndices, op.getTotalPartitionWarps());
      });
      maybeAddScratchBuffer<BufferT::BufferKind::Scratch>(op, numWarpIndices);
      return;
    }
    unsigned bytes = scratchSizeGetter(op);
    maybeAddScratchBuffer<BufferT::BufferKind::Scratch>(op, bytes,
                                                        scratchAlignment);
  }

  void getValueAlias(Value value, SharedMemoryAliasAnalysis &analysis) {
    dataflow::Lattice<AliasInfo> *latticeElement =
        analysis.getLatticeElement(value);
    if (latticeElement) {
      AliasInfo &info = latticeElement->getValue();
      if (!info.getAllocs().empty()) {
        for (auto alloc : info.getAllocs()) {
          allocation->addAlias(value, alloc);
        }
      }
    }
  }

  /// Extract all shared memory values and their sizes
  void getValuesAndSizes() {
    // Get the alloc values
    operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
      getExplicitValueSize(op);
      getScratchValueSize(op);
    });
    // Get the alias values
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    SharedMemoryAliasAnalysis *aliasAnalysis =
        solver->load<SharedMemoryAliasAnalysis>();
    // Run the analysis rooted at every isolated from above operation, including
    // the top-level function but also any nested regions.
    operation->walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
          failed(solver->initializeAndRun(op))) {
        // TODO: return error instead of bailing out..
        llvm_unreachable("failed to run SharedMemoryAliasAnalysis");
      }
    });
    operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
      for (auto operand : op->getOperands()) {
        getValueAlias(operand, *aliasAnalysis);
      }
      for (auto value : op->getResults()) {
        getValueAlias(value, *aliasAnalysis);
      }
    });
  }

  // Extend buffer LiveRange to include interval
  void updateBufferRange(BufferT *buffer, IntervalT interval) {
    if (bufferRange.contains(buffer))
      interval = interval.span(bufferRange[buffer]);
    bufferRange[buffer] = interval;
  }

  /// Computes the liveness range of the allocated value.
  /// Each buffer is allocated only once.
  void resolveExplicitBufferLiveness(LivenessF getLiveness) {
    for (auto valueBufferIter : allocation->valueBuffer) {
      auto value = valueBufferIter.first;
      auto *buffer = valueBufferIter.second;
      auto ranges = getLiveness(value);
      if (!ranges.empty()) {
        updateBufferRange(buffer, ranges.span());
        LLVM_DEBUG({
          llvm::dbgs() << "-- buffer " << buffer->id << "; value: ";
          value.dump();
        });
      }
    }
  }

  /// Following the alias lattice, an alias that:
  ///   - References a single buffer, will simply update the live range of
  ///     that buffer. Note that this will cover gaps also.
  ///   - References multiple buffers as in the case of loop-carried variables,
  ///     creates a new Buffer<Alias> for each disjoint range. The new Alias
  ///     Buffer will represent the live-range for all referenced buffers and
  ///     apply the real buffers allocation during interference graphing.
  ///
  /// Example: (numbers represent op liveness index)
  ///  3    %b0 = convert_layout %g0                  -- Buffer #0
  ///  4    %fr = for (.., %arg0 = %b0) {             -+ Alias #1
  ///  5        %gn = load %pc                         |   Buffers #0,#1
  ///  6        %bc = convert_layout %arg0            -+
  ///  7        %v = add %bc, ...
  ///  8        %bn = convert_layout %gn              -+ Buffer #1
  ///  9        %pn = addptr %pc, %cst                 |   #1 is loop-carried
  ///  10       yield ... %bn                         -+   does not overlap #0
  ///       }
  ///  11   %be = convert_layout %fr#1                -- Alias #2: Buffer #0,#1
  ///  12   %ve = add %be
  void resolveAliasBufferLiveness(LivenessF getLiveness) {
    for (const auto &[value, buffers] : allocation->aliasBuffer) {
      auto ranges = getLiveness(value);
      auto *buffer = buffers.front();
      if (buffers.size() == 1) {
        for (auto interval : ranges)
          updateBufferRange(buffer, interval);
      } else {
        for (auto interval : ranges) {
          // Create Alias Buffer for each disjoint interval
          BufferT *aliasBuf = allocation->addAlias(
              value, buffer->size, buffer->alignment);
          updateBufferRange(aliasBuf, interval);
          for (auto buffer : buffers)
            aliasBuf->aliases.push_back(buffer);
        }
      }
    }
  }

  /// Computes the liveness range of scratched buffers.
  /// Some operations may have a temporary buffer that is not explicitly
  /// allocated, but is used to store intermediate results.
  void resolveScratchBufferLiveness(
      const DenseMap<Operation *, size_t> &operationId) {
    // Analyze liveness of scratch buffers and virtual buffers.
    auto processScratchMemory = [&](const auto &container) {
      for (auto [op, buffer] : container) {
        // Buffers owned by the function are assumed live for the whole
        // function. This memory is used for warp specialization codegen.
        // FIXME: Spooky-action-at-a-distance. Find a better way to model this.
        if (op == operation) {
          bufferRange.insert(
              {buffer, Interval(size_t(), std::numeric_limits<size_t>::max())});
          continue;
        }

        // Any scratch memory's live range is the current operation's live
        // range.
        updateBufferRange(buffer, operationId.at(op));
        LLVM_DEBUG({
          llvm::dbgs() << "-- buffer " << buffer->id << "; value: ";
          op->dump();
        });
      }
    };
    processScratchMemory(allocation->opScratch);
    processScratchMemory(allocation->opVirtual);
  }

  /// Resolves liveness of all values involved under the root operation.
  void resolveLiveness() {
    // Assign an ID to each operation using post-order traversal.
    // To achieve the correct liveness range, the parent operation's ID
    // should be greater than each of its child operation's ID .
    // Example:
    //     ...
    //     %5 = triton.convert_layout %4
    //     %6 = scf.for ... iter_args(%arg0 = %0) -> (i32) {
    //       %2 = triton.convert_layout %5
    //       ...
    //       scf.yield %arg0
    //     }
    // For example, %5 is defined in the parent region and used in
    // the child region, and is not passed as a block argument.
    // %6 should should have an ID greater than its child operations,
    // otherwise %5 liveness range ends before the child operation's liveness
    // range ends.
    DenseMap<Operation *, size_t> operationId;
    operation->walk<WalkOrder::PostOrder>(
        [&](Operation *op) { operationId[op] = operationId.size(); });

    // Analyze liveness of explicit buffers
    Liveness liveness(operation);
    auto getValueLiveRanges = [&](Value value) {
      LiveIntervals intervals;
      auto liveOperations = liveness.resolveLiveness(value);
      llvm::for_each(liveOperations, [&](Operation *liveOp) {
        intervals.push_back(operationId[liveOp]);
      });
      intervals.sortAndJoin();
      return intervals;
    };

    resolveExplicitBufferLiveness(getValueLiveRanges);
    resolveAliasBufferLiveness(getValueLiveRanges);
    resolveScratchBufferLiveness(operationId);
  }

  void dumpBuffers() const {
    LDBG("Dump bufferRange: id size offset ---------");
    for (auto bufferIter : bufferRange) {
      llvm::dbgs() << "-- " << bufferIter.first->id << " "
                   << bufferIter.first->size << " " << bufferIter.first->offset;
      llvm::dbgs() << " interval " << bufferIter.second.start() << " "
                   << bufferIter.second.end() << "\n";
    }
  }

  void dumpAllocationSize() const {
    LDBG("Dump shared memory allocation size -----------");
    auto liveBuffers = allocation->getLiveBuffers();
    auto analyzedSize = 0;
    for (auto [op, bufferIds] : liveBuffers) {
      auto size = 0;
      for (auto bufferId : bufferIds) {
        auto bufferSize = allocation->getAllocatedSize(bufferId);
        size += bufferSize;
      }
      analyzedSize = std::max(analyzedSize, size);
    }
    llvm::dbgs() << "Allocated: " << allocation->sharedMemorySize
                 << ", analyzed: " << analyzedSize << "\n";
  }

  void dumpInterferenceGraph(const GraphT &interference) const {
    LDBG("\n");
    LDBG("Dump interference graph: \n");
    for (auto edges : interference) {
      llvm::dbgs() << "-- from " << edges.first->id << " to ";
      for (auto node : edges.second) {
        llvm::dbgs() << node->id << "; ";
      }
      llvm::dbgs() << "\n";
    }
  }

  /// Computes the shared memory offsets for all related values.
  /// Paper: Algorithms for Compile-Time Memory Optimization
  /// (https://dl.acm.org/doi/pdf/10.5555/314500.315082)
  void computeOffsets() {
    SmallVector<BufferT *> buffers;
    for (auto bufferIter : bufferRange) {
      buffers.emplace_back(bufferIter.first);
    }

    // Sort buffers by size in descending order to reduce the fragmentation
    // on big buffers caused by smaller buffers. Big buffers have a higher
    // chance to overlap with multiple other buffers, and allocating them first
    // (by calculateStarts) ensures a higher chance that they will occupy a
    // standalone smem slot.
    llvm::stable_sort(
        buffers, [&](BufferT *A, BufferT *B) { return A->size > B->size; });

    calculateStarts(buffers);

    // NOTE: The original paper doesn't consider interference between
    // the bumped ranges. Buffers that previously do not interfere with
    // could interfere after offset bumping if their liveness ranges overlap.
    // Therefore, we rerun the interference graph algorithm after bumping so
    // that we regroup the buffers and color them again. Since we always
    // increase the buffer offset and keep reducing conflicts, we will
    // eventually reach a fixed point.
    GraphT interference;
    buildInterferenceGraph(buffers, interference);
    do {
      allocate(buffers, interference);
      buildInterferenceGraph(buffers, interference);
    } while (!interference.empty());

    LLVM_DEBUG(dumpAllocationSize());
  }

  /// Computes the initial shared memory offsets.
  void calculateStarts(const SmallVector<BufferT *> &buffers) {
    //  v = values in shared memory
    //  t = triplet of (size, start, end)
    //  shared memory space
    //  -
    //  |         *******t4
    //  | /|\ v2 inserts t4, t5, and t6
    //  |  |
    //  | ******t5         ************t6
    //  | ^^^^^v2^^^^^^
    //  |  |      *********************t2
    //  | \|/ v2 erases t1
    //  | ******t1 ^^^^^^^^^v1^^^^^^^^^ ************t3
    //  |---------------------------------------------| liveness range
    //    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 ...
    // If the available triple's range is less than a given buffer range,
    // we won't know if there has been an overlap without using graph coloring.
    // Start -> Liveness Range
    using TripleMapT = std::multimap<size_t, IntervalT>;
    TripleMapT tripleMap;
    tripleMap.insert(std::make_pair(0, IntervalT()));
    SmallVector<BufferT *> xBuffers;
    for (auto *buf : buffers) {
      if (buf->kind != BufferT::BufferKind::Alias)
        xBuffers.push_back(buf);
    }

    while (!xBuffers.empty()) {
      auto tripleIt = tripleMap.begin();
      size_t offset = tripleIt->first;
      IntervalT range = tripleIt->second;
      tripleMap.erase(tripleIt);
      auto bufferIt = llvm::find_if(xBuffers, [&](auto *buffer) {
        auto xRange = bufferRange[buffer];
        bool res = xRange.intersects(range);
        for (const auto &val : tripleMap)
          res = res &&
                !val.second.intersects(xRange); // only one buffer intersect
        return res;
      });
      if (bufferIt != xBuffers.end()) {
        auto buffer = *bufferIt;
        auto xSize = buffer->size;
        auto xRange = bufferRange.lookup(buffer);
        // TODO(Keren): A buffer's size shouldn't be determined here, have to
        // clean it up
        size_t alignOffset = buffer->setOffsetAligned(offset);
        tripleMap.insert({alignOffset + xSize,
                          Interval{std::max(range.start(), xRange.start()),
                                   std::min(range.end(), xRange.end())}});
        // We could either insert (range.start, xRange.start) or (range.start,
        // xRange.end), both are correct and determine the potential buffer
        // offset, and the graph coloring algorithm will solve the interference,
        // if any
        if (range.start() < xRange.start())
          tripleMap.insert({offset, Interval{range.start(), xRange.end()}});
        if (xRange.end() < range.end())
          tripleMap.insert({offset, Interval{xRange.start(), range.end()}});
        xBuffers.erase(bufferIt);
      }
    }
    LLVM_DEBUG(dumpBuffers());
  }

  /// Builds a graph of all shared memory values. Edges are created between
  /// shared memory values that are overlapping.
  void buildInterferenceGraph(const SmallVector<BufferT *> &buffers,
                              GraphT &interference) {
    // Reset interference graph
    interference.clear();
    for (auto x : buffers) {
      if (x->kind != BufferT::BufferKind::Alias) {
        for (auto y : buffers) {
          if (x == y)
            continue;
          y->applyToBufferAndAliases([&](BufferT *yActual) {
            if (yActual != x) {
              auto xStart = x->offset;
              auto yStart = yActual->offset;
              auto xSize = x->size;
              auto ySize = yActual->size;
              Interval xSizeRange = {xStart, xStart + xSize};
              Interval ySizeRange = {yStart, yStart + ySize};
              auto xOpRange = bufferRange.lookup(x);
              auto yOpRange = bufferRange.lookup(y);

              // Buffers interfere if their allocation offsets overlap and they are
              // live at the same time.
              if (xOpRange.intersects(yOpRange) &&
                  xSizeRange.intersects(ySizeRange)) {
                interference[x].insert(y);
              }

              // Buffers also interfere if their allocation offsets overlap and they
              // exist within regions that may execute simultaneously with respect to
              // each other.
              auto xOwner = x->owner;
              auto yOwner = yActual->owner;
              auto wsx = xOwner->getParentWithTrait<OpTrait::AsyncRegions>();
              auto wsy = yOwner->getParentWithTrait<OpTrait::AsyncRegions>();
              if (wsx && wsy && wsx == wsy &&
                  xOwner->getParentRegion() != yOwner->getParentRegion() &&
                  xSizeRange.intersects(ySizeRange)) {
                interference[x].insert(y);
              }
            }
          });
        }
      }
    }

    LLVM_DEBUG(dumpInterferenceGraph(interference));
  }

  /// Finalizes shared memory offsets considering interference.
  void allocate(const SmallVector<BufferT *> &buffers,
                const GraphT &interference) {
    // Reset shared memory size
    allocation->sharedMemorySize = 0;

    SmallVector<BufferT *> xBuffers;
    for (auto *buf : buffers) {
      if (buf->kind != BufferT::BufferKind::Alias)
        xBuffers.push_back(buf);
    }

    // First-fit graph coloring
    // Neighbors are nodes that interfere with each other.
    // We color a node by finding the index of the first available
    // non-neighboring node or the first neighboring node without any color.
    // Nodes with the same color do not interfere with each other.
    DenseMap<BufferT *, int> colors;
    for (auto value : xBuffers) {
      colors[value] = (value == xBuffers.front()) ? 0 : -1;
    }
    SmallVector<bool> available(xBuffers.size());
    for (auto x : xBuffers) {
      std::fill(available.begin(), available.end(), true);
      for (auto y : interference.lookup(x)) {
        y->applyToBufferAndAliases([&](BufferT *actual) {
          if (actual != x) {
            int color = colors[actual];
            if (color >= 0)
              available[color] = false;
          }
        });
      }
      auto it = llvm::find(available, true);
      colors[x] = std::distance(available.begin(), it);
      LLDBG_L2V("COLOR", x->id << " -> " << colors[x]);
    }
    // Finalize allocation
    // color0: [0, 7), [0, 8), [0, 15) -> [0, 7), [0, 8), [0, 15)
    // color1: [7, 9) -> [0 + 1 * 15, 9 + 1 * 15) -> [15, 24)
    // color2: [8, 12) -> [8 + 2 * 15, 12 + 2 * 15) -> [38, 42)
    // TODO(Keren): We are wasting memory here.
    // Nodes with color2 can actually start with 24.
    for (auto x : xBuffers) {
      size_t newOffset = 0;
      for (auto y : interference.lookup(x)) {
        y->applyToBufferAndAliases([&](BufferT *actual) {
          if (actual != x)
            newOffset = std::max(newOffset, actual->offset + actual->size);
        });
      }
      if (colors.lookup(x) != 0) {
        if (size_t diff = newOffset % x->alignment) {
          // fix alignment
          newOffset += x->alignment - diff;
        }
        x->offset = newOffset;
      }
      allocation->sharedMemorySize =
          std::max(allocation->sharedMemorySize, x->offset + x->size);
    }
    LLVM_DEBUG(dumpBuffers());
  }

  void dump() const {
    allocation->dump();

    for (auto pair : bufferRange) {
      LLDBG_L1V("BUFFER RANGE", pair.second);
      pair.first->dump();
    }
  }

private:
  Operation *operation;
  Allocation::FuncAllocMapT *funcAllocMap;
  Allocation *allocation;
  BufferRangeMapT bufferRange;
  AllocationAnalysisScratchSizeFn scratchSizeGetter;
};

} // namespace triton

void Allocation::run(
    FuncAllocMapT &funcAllocMap,
    triton::AllocationAnalysisScratchSizeFn scratchSizeGetter) {
  triton::AllocationAnalysis(getOperation(), &funcAllocMap, this,
                             scratchSizeGetter);
}

std::map<Operation *, SmallVector<Allocation::BufferId>>
Allocation::getLiveBuffers() {
  std::map<Operation *, SmallVector<BufferId>> liveBuffers;

  Operation *rootOperation = getOperation();
  Liveness liveness(rootOperation);
  auto analyzeOperation = [&](Operation *op) -> void {
    auto scratchBuffer = getBufferId(op);
    if (scratchBuffer != InvalidBufferId)
      liveBuffers[op].push_back(scratchBuffer);
    for (auto result : op->getOpResults()) {
      auto bufferId = getBufferId(result);
      if (bufferId == Allocation::InvalidBufferId)
        continue;
      auto liveOperations = liveness.resolveLiveness(result);
      for (auto depOp : liveOperations)
        liveBuffers[depOp].push_back(bufferId);
    }
  };
  rootOperation->walk(analyzeOperation);
  return liveBuffers;
}

// Diagnostics
void Allocation::BufferT::dump() const {
  LLDBG_L2("ID", (int)id);
  LLDBG_L2("KIND", (int)kind);
  LLDBG_L2("SIZE", size);
  LLDBG_L2("OFFSET", offset);
  LLDBG_L2("ALIGN", alignment);
  if (!aliases.empty()) {
    LLDBG_L2C_ID("ALIASES", aliases);
  }
}

void Allocation::dump() const {
  LLDBG_L0("ALLOCATION SIZE", sharedMemorySize);
  for (auto pair : opScratch) {
    LLDBG_L1P("SCRATCH");
    pair.first->print(llvm::dbgs());
    llvm::dbgs() << "\n";
    pair.second->dump();
  }
  for (auto pair : opVirtual) {
    LLDBG_L1P("VIRTUAL");
    pair.first->print(llvm::dbgs());
    llvm::dbgs() << "\n";
    pair.second->dump();
  }
  for (auto pair : valueBuffer) {
    LLDBG_L1P("VALUE");
    printValue(pair.first);
    pair.second->dump();
  }
  for (auto pair : aliasBuffer) {
    LLDBG_L1P("ALIAS");
    printValue(pair.first);
    std::string astr;
    llvm::for_each(pair.second, [&astr, cnt = 0](auto *buf) mutable {
      astr += (cnt++ ? ", " : "");
      astr += std::to_string(buf->id);
    });
    LLDBG_L2("IDS", astr);
  }
}

} // namespace mlir
