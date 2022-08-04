#include "triton/Analysis/Allocation.h"
#include "mlir/Analysis/Liveness.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <limits>

namespace mlir {

void AllocationAnalysis::run() {
  ValueRangeMapT valueRange;
  resolveLiveness(valueRange);
  computeOffsets(valueRange);
}

void AllocationAnalysis::resolveLiveness(
    AllocationAnalysis::ValueRangeMapT &valueRange) {
  Liveness liveness(operation);
  DenseMap<Operation *, size_t> operationIds;
  operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
    operationIds.insert({op, operationIds.size()});
  });

  operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (Value result : op->getResults()) {
      auto liveOperations = liveness.resolveLiveness(result);
      auto minId = std::numeric_limits<size_t>::max();
      auto maxId = std::numeric_limits<size_t>::min();
      std::for_each(liveOperations.begin(), liveOperations.end(),
                    [&](Operation *liveOp) {
                      if (operationIds[liveOp] < minId) {
                        minId = operationIds[liveOp];
                      }
                      if (operationIds[liveOp] > maxId) {
                        maxId = operationIds[liveOp];
                      }
                    });
      valueRange.insert({result, Range(minId, maxId + 1)});
    }
  });
}

void AllocationAnalysis::getSharedMemoryValuesAndSizes(
    const AllocationAnalysis::ValueRangeMapT &valueRange,
    SmallVector<Value> &sharedMemoryValues) {
  for (auto &valueRange : valueRange) {
    auto value = valueRange.first;
    auto type = value.getType();
    if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
      auto encoding = tensorType.getEncoding();
      if (encoding &&
          encoding.isa<triton::gpu::TritonGPUSharedEncodingAttr>()) {
        // Bytes could be a different value once we support padding or other
        // allocation policies.
        auto bytes = tensorType.getNumElements() *
                     tensorType.getElementTypeBitWidth() / 8;
        sharedMemoryValues.emplace_back(value);
        valueSize.insert({value, bytes});
      }
    }
  }
}

void AllocationAnalysis::calculateSharedMemoryStarts(
    const AllocationAnalysis::ValueRangeMapT &valueRange,
    const SmallVector<Value> &sharedMemoryValues,
    ValueSizeMapT &sharedMemoryStart) {
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
  TripleMapT tripleMap;
  tripleMap.insert(std::make_pair(0, Range<size_t>()));
  SmallVector<Value> values = sharedMemoryValues;
  while (!values.empty()) {
    auto tripleIt = tripleMap.begin();
    auto size = tripleIt->first;
    auto range = tripleIt->second;
    tripleMap.erase(tripleIt);
    auto valueIt = std::find_if(values.begin(), values.end(), [&](Value value) {
      auto xRange = valueRange.lookup(value);
      bool res = xRange.intersects(range);
      for (auto val : tripleMap)
        res = res && !val.second.intersects(xRange);
      return res;
    });
    if (valueIt != values.end()) {
      auto value = *valueIt;
      auto xSize = valueSize.lookup(value);
      auto xRange = valueRange.lookup(value);
      sharedMemoryStart[value] = size;
      tripleMap.insert(
          {size + xSize, Range{std::max(range.start(), xRange.start()),
                               std::min(range.end(), xRange.end())}});
      if (range.start() < xRange.start())
        tripleMap.insert({size, Range{range.start(), xRange.end()}});
      if (xRange.end() < range.end())
        tripleMap.insert({size, Range{xRange.start(), range.end()}});
      values.erase(valueIt);
    }
  }
}

void AllocationAnalysis::buildInterferenceGraph(
    const AllocationAnalysis::ValueRangeMapT &valueRange,
    const SmallVector<Value> &sharedMemoryValues,
    const ValueSizeMapT &sharedMemoryStart, GraphT &interference) {
  for (auto x : sharedMemoryValues) {
    for (auto y : sharedMemoryValues) {
      if (x == y)
        continue;
      auto xStart = sharedMemoryStart.lookup(x);
      auto yStart = sharedMemoryStart.lookup(y);
      auto xSize = valueSize.lookup(x);
      auto ySize = valueSize.lookup(y);
      Range xSizeRange = {xStart, xStart + xSize};
      Range ySizeRange = {yStart, yStart + ySize};
      auto xOpRange = valueRange.lookup(x);
      auto yOpRange = valueRange.lookup(y);
      if (xOpRange.intersects(yOpRange) && xSizeRange.intersects(ySizeRange)) {
        interference[x].insert(y);
      }
    }
  }
}

void AllocationAnalysis::allocateSharedMemory(
    const AllocationAnalysis::ValueRangeMapT &valueRangeMap,
    const SmallVector<Value> &sharedMemoryValues,
    const AllocationAnalysis::ValueSizeMapT &sharedMemoryStart,
    const AllocationAnalysis::GraphT &interference) {
  // First-fit graph coloring
  // Neighbors are nodes that interfere with each other.
  // We color a node by finding the index of the first available non-neighboring
  // node or the first neighboring node without any color.
  // Nodes with the same color do not interfere with each other.
  DenseMap<Value, int> colors;
  for (auto value : sharedMemoryValues) {
    colors[value] = (value == sharedMemoryValues[0]) ? 0 : -1;
  }
  SmallVector<bool> available(sharedMemoryValues.size());
  for (auto x : sharedMemoryValues) {
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
  for (auto x : sharedMemoryValues) {
    size_t adj = 0;
    for (auto y : interference.lookup(x)) {
      adj = std::max(adj, sharedMemoryStart.lookup(y) + valueSize.lookup(y));
    }
    valueOffset[x] = sharedMemoryStart.lookup(x) + colors.lookup(x) * adj;
    sharedMemorySize =
        std::max(sharedMemorySize, valueOffset[x] + valueSize.lookup(x));
  }
}

void AllocationAnalysis::computeOffsets(
    const AllocationAnalysis::ValueRangeMapT &valueRange) {
  SmallVector<Value> sharedMemoryValues;
  getSharedMemoryValuesAndSizes(valueRange, sharedMemoryValues);

  ValueSizeMapT sharedMemoryStart;
  calculateSharedMemoryStarts(valueRange, sharedMemoryValues,
                              sharedMemoryStart);

  GraphT interference;
  buildInterferenceGraph(valueRange, sharedMemoryValues, sharedMemoryStart,
                         interference);

  allocateSharedMemory(valueRange, sharedMemoryValues, sharedMemoryStart,
                       interference);
}

} // namespace mlir
