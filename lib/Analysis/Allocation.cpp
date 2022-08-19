#include "triton/Analysis/Allocation.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <limits>

namespace mlir {

//===----------------------------------------------------------------------===//
// Shared Memory Allocation Analysis
//===----------------------------------------------------------------------===//
namespace triton {
class AllocationAnalysis {
public:
  AllocationAnalysis(Operation *operation, Allocation *allocation)
      : operation(operation), allocation(allocation) {
    run();
  }

private:
  using BufferT = Allocation::BufferT;

  /// Value -> Liveness Range
  /// Use MapVector to ensure determinism.
  using BufferRangeMapT = llvm::MapVector<BufferT *, Range<size_t>>;
  /// Nodes -> Nodes
  using GraphT = DenseMap<BufferT *, DenseSet<BufferT *>>;

  void run() {
    getValuesAndSizes();
    resolveLiveness();
    computeOffsets();
  }

  /// Initializes explicitly defined shared memory values for a given operation.
  void getExplicitValueSize(Operation *op) {
    for (Value result : op->getResults()) {
      auto type = result.getType();
      if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
        auto encoding = tensorType.getEncoding();
        if (encoding && encoding.isa<triton::gpu::SharedEncodingAttr>()) {
          // Bytes could be a different value once we support padding or other
          // allocation policies.
          auto bytes = tensorType.getNumElements() *
                       tensorType.getElementTypeBitWidth() / 8;
          allocation->addBuffer<BufferT::BufferKind::Explicit>(result, bytes);
        }
      }
    }
  }

  /// Initializes temporary shared memory for a given operation.
  void getScratchValueSize(Operation *op) {
    // TODO(Keren): Add atomic ops
    // TODO(Keren): Add convert ops
    if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      // TODO(Keren): Reduce with index is not supported yet.
      auto value = op->getOperand(0);
      if (auto tensorType = value.getType().dyn_cast<RankedTensorType>()) {
        auto bytes = tensorType.getNumElements() *
                     tensorType.getElementTypeBitWidth() / 8;
        allocation->addBuffer<BufferT::BufferKind::Scratch>(op, bytes);
      }
    }
  }

  /// Extract all shared memory values and their sizes
  void getValuesAndSizes() {
    operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
      getExplicitValueSize(op);
      getScratchValueSize(op);
    });
  }

  /// Resolves liveness of all values involved under the root operation.
  void resolveLiveness() {
    // In the SCF dialect, we always have a sequentially nested structure of
    // blocks
    DenseMap<Operation *, size_t> operationId;
    operation->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { operationId[op] = operationId.size(); });

    Liveness liveness(operation);
    operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
      for (Value result : op->getResults()) {
        auto liveOperations = liveness.resolveLiveness(result);
        auto minId = std::numeric_limits<size_t>::max();
        auto maxId = std::numeric_limits<size_t>::min();
        std::for_each(liveOperations.begin(), liveOperations.end(),
                      [&](Operation *liveOp) {
                        if (operationId[liveOp] < minId) {
                          minId = operationId[liveOp];
                        }
                        if (operationId[liveOp] > maxId) {
                          maxId = operationId[liveOp];
                        }
                      });
        if (allocation->valueBuffer.count(result)) {
          auto *buffer = allocation->valueBuffer[result];
          bufferRange.insert({buffer, Range(minId, maxId + 1)});
        }
      }
      if (allocation->opScratch.count(op)) {
        // Any scratch memory's live range is the current operation's live
        // range.
        auto *buffer = allocation->opScratch[op];
        bufferRange.insert(
            {buffer, Range(operationId[op], operationId[op] + 1)});
      }
    });
  }

  /// Computes the shared memory offsets for all related values.
  /// Paper: Algorithms for Compile-Time Memory Optimization
  /// (https://www.cs.utexas.edu/users/harrison/papers/compile-time.pdf)
  void computeOffsets() {
    SmallVector<BufferT *> buffers;
    for (auto bufferIter : bufferRange) {
      buffers.emplace_back(bufferIter.first);
    }

    DenseMap<BufferT *, size_t> bufferStart;
    calculateStarts(buffers, bufferStart);

    GraphT interference;
    buildInterferenceGraph(buffers, bufferStart, interference);

    allocate(buffers, bufferStart, interference);
  }

  /// Computes the initial shared memory offsets.
  void calculateStarts(const SmallVector<BufferT *> &buffers,
                       DenseMap<BufferT *, size_t> &bufferStart) {
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
    /// Start -> Liveness Range
    using TripleMapT = std::multimap<size_t, Range<size_t>>;
    TripleMapT tripleMap;
    tripleMap.insert(std::make_pair(0, Range<size_t>()));
    SmallVector<BufferT *> xBuffers = buffers;
    while (!xBuffers.empty()) {
      auto tripleIt = tripleMap.begin();
      auto size = tripleIt->first;
      auto range = tripleIt->second;
      tripleMap.erase(tripleIt);
      auto bufferIt =
          std::find_if(xBuffers.begin(), xBuffers.end(), [&](auto *buffer) {
            auto xRange = bufferRange[buffer];
            bool res = xRange.intersects(range);
            for (auto val : tripleMap)
              res = res && !val.second.intersects(xRange);
            return res;
          });
      if (bufferIt != xBuffers.end()) {
        auto buffer = *bufferIt;
        auto xSize = buffer->size;
        auto xRange = bufferRange.lookup(buffer);
        bufferStart[buffer] = size;
        tripleMap.insert(
            {size + xSize, Range{std::max(range.start(), xRange.start()),
                                 std::min(range.end(), xRange.end())}});
        if (range.start() < xRange.start())
          tripleMap.insert({size, Range{range.start(), xRange.end()}});
        if (xRange.end() < range.end())
          tripleMap.insert({size, Range{xRange.start(), range.end()}});
        xBuffers.erase(bufferIt);
      }
    }
  }

  /// Builds a graph of all shared memory values. Edges are created between
  /// shared memory values that are overlapping.
  void buildInterferenceGraph(const SmallVector<BufferT *> &buffers,
                              const DenseMap<BufferT *, size_t> &bufferStart,
                              GraphT &interference) {
    for (auto x : buffers) {
      for (auto y : buffers) {
        if (x == y)
          continue;
        auto xStart = bufferStart.lookup(x);
        auto yStart = bufferStart.lookup(y);
        auto xSize = x->size;
        auto ySize = y->size;
        Range xSizeRange = {xStart, xStart + xSize};
        Range ySizeRange = {yStart, yStart + ySize};
        auto xOpRange = bufferRange.lookup(x);
        auto yOpRange = bufferRange.lookup(y);
        if (xOpRange.intersects(yOpRange) &&
            xSizeRange.intersects(ySizeRange)) {
          interference[x].insert(y);
        }
      }
    }
  }

  /// Finalizes shared memory offsets considering interference.
  void allocate(const SmallVector<BufferT *> &buffers,
                const DenseMap<BufferT *, size_t> &bufferStart,
                const GraphT &interference) {
    // First-fit graph coloring
    // Neighbors are nodes that interfere with each other.
    // We color a node by finding the index of the first available
    // non-neighboring node or the first neighboring node without any color.
    // Nodes with the same color do not interfere with each other.
    DenseMap<BufferT *, int> colors;
    for (auto value : buffers) {
      colors[value] = (value == buffers[0]) ? 0 : -1;
    }
    SmallVector<bool> available(buffers.size());
    for (auto x : buffers) {
      std::fill(available.begin(), available.end(), true);
      for (auto y : interference.lookup(x)) {
        int color = colors[y];
        if (color >= 0) {
          available[color] = false;
        }
      }
      auto it = std::find(available.begin(), available.end(), true);
      colors[x] = std::distance(available.begin(), it);
    }
    // Finalize allocation
    // color0: [0, 7), [0, 8), [0, 15) -> [0, 7), [0, 8), [0, 15)
    // color1: [7, 9) -> [0 + 1 * 15, 9 + 1 * 15) -> [15, 24)
    // color2: [8, 12) -> [8 + 2 * 15, 12 + 2 * 15) -> [38, 42)
    // TODO(Keren): We are wasting memory here.
    // Nodes with color2 can actually start with 24.
    for (auto x : buffers) {
      size_t adj = 0;
      for (auto y : interference.lookup(x)) {
        adj = std::max(adj, bufferStart.lookup(y) + y->size);
      }
      x->offset = bufferStart.lookup(x) + colors.lookup(x) * adj;
      allocation->sharedMemorySize =
          std::max(allocation->sharedMemorySize, x->offset + x->size);
    }
  }

private:
  Operation *operation;
  Allocation *allocation;
  BufferRangeMapT bufferRange;
};
} // namespace triton

void Allocation::run() { triton::AllocationAnalysis(getOperation(), this); }

} // namespace mlir
