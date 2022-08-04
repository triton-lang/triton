#ifndef TRITON_ANALYSIS_ALLOCATION_H
#define TRITON_ANALYSIS_ALLOCATION_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

/// Modified from llvm-15.0: llvm/ADT/AddressRanges.h
/// A class that represents an address range. The range is specified using
/// a start and an end address: [Start, End).
template <typename AddrT> class Range {
public:
  Range() {}
  Range(AddrT S, AddrT E) : Start(S), End(E) { assert(Start <= End); }
  AddrT start() const { return Start; }
  AddrT end() const { return End; }
  AddrT size() const { return End - Start; }
  bool contains(AddrT Addr) const { return Start <= Addr && Addr < End; }
  bool intersects(const Range &R) const {
    return Start < R.End && R.Start < End;
  }
  bool operator==(const Range &R) const {
    return Start == R.Start && End == R.End;
  }
  bool operator!=(const Range &R) const { return !(*this == R); }
  bool operator<(const Range &R) const {
    return std::make_pair(Start, End) < std::make_pair(R.Start, R.End);
  }

private:
  AddrT Start = std::numeric_limits<AddrT>::min();
  AddrT End = std::numeric_limits<AddrT>::max();
};

//===----------------------------------------------------------------------===//
// Shared Memory Allocation Analysis
//===----------------------------------------------------------------------===//
class AllocationAnalysis {
public:
  using ValueSizeMapT = llvm::DenseMap<Value, size_t>;

public:
  /// Creates a new Allocation analysis that computes the shared memory
  /// information for all associated shared memory values.
  AllocationAnalysis(Operation *operation) : operation(operation) { run(); }

  /// Returns the operation this analysis was constructed from.
  Operation *getOperation() const { return operation; }

  /// Returns the offset of the given value in the shared memory.
  size_t getOffset(Value value) const { return valueOffset.lookup(value); }

  /// Returns the size of the given value in the shared memory.
  size_t getAllocatedSize(Value value) const { return valueSize.lookup(value); }

  /// Returns the size of total shared memory allocated
  size_t getSharedMemorySize() const { return sharedMemorySize; }

private:
  /// Value -> Range
  /// Use MapVector to ensure determinism.
  using ValueRangeMapT = llvm::MapVector<Value, Range<size_t>>;
  /// Start -> Range
  using TripleMapT = std::multimap<size_t, Range<size_t>>;
  /// Nodes -> Nodes
  using GraphT = DenseMap<Value, DenseSet<Value>>;

  /// Runs allocation analysis on the given top-level operation.
  void run();

  /// Resolves liveness of all values involved under the root operation.
  void resolveLiveness(ValueRangeMapT &valueRangeMap);

  /// Computes the shared memory offsets for all related values.
  /// Paper: Algorithms for Compile-Time Memory Optimization
  /// (https://www.cs.utexas.edu/users/harrison/papers/compile-time.pdf)
  void computeOffsets(const ValueRangeMapT &valueRangeMap);

  /// Gets shared memory value and size from valueRangeMap.
  void getSharedMemoryValuesAndSizes(const ValueRangeMapT &valueRangeMap,
                                     SmallVector<Value> &sharedMemoryValues);

  /// Computes the initial shared memory offsets.
  void calculateSharedMemoryStarts(const ValueRangeMapT &valueRangeMap,
                                   const SmallVector<Value> &sharedMemoryValues,
                                   ValueSizeMapT &sharedMemoryStart);

  /// Builds a graph of all shared memory values. Edges are created between
  /// between shared memory values that are overlapping.
  void buildInterferenceGraph(const ValueRangeMapT &valueRangeMap,
                              const SmallVector<Value> &sharedMemoryValues,
                              const ValueSizeMapT &sharedMemoryStart,
                              GraphT &interference);

  /// Finalizes shared memory offsets considering interference.
  void allocateSharedMemory(const ValueRangeMapT &valueRangeMap,
                            const SmallVector<Value> &sharedMemoryValues,
                            const ValueSizeMapT &sharedMemoryStart,
                            const GraphT &interference);

private:
  Operation *operation;
  ValueSizeMapT valueOffset;
  ValueSizeMapT valueSize;
  size_t sharedMemorySize = 0;
};

} // namespace mlir

#endif
