#ifndef TRITON_ANALYSIS_ALLOCATION_H
#define TRITON_ANALYSIS_ALLOCATION_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <atomic>
#include <limits>

namespace mlir {

namespace triton {
class AllocationAnalysis;

SmallVector<unsigned>
getScratchConfigForCvtLayout(triton::gpu::ConvertLayoutOp op, unsigned &inVec,
                             unsigned &outVec);

SmallVector<unsigned> getScratchConfigForReduce(triton::ReduceOp op);

} // namespace triton

/// Modified from llvm-15.0: llvm/ADT/AddressRanges.h
/// A class that represents an interval, specified using a start and an end
/// values: [Start, End).
template <typename T> class Interval {
public:
  Interval() {}
  Interval(T S, T E) : Start(S), End(E) { assert(Start <= End); }
  T start() const { return Start; }
  T end() const { return End; }
  T size() const { return End - Start; }
  bool contains(T Addr) const { return Start <= Addr && Addr < End; }
  bool intersects(const Interval &R) const {
    return Start < R.End && R.Start < End;
  }
  bool operator==(const Interval &R) const {
    return Start == R.Start && End == R.End;
  }
  bool operator!=(const Interval &R) const { return !(*this == R); }
  bool operator<(const Interval &R) const {
    return std::make_pair(Start, End) < std::make_pair(R.Start, R.End);
  }

private:
  T Start = std::numeric_limits<T>::min();
  T End = std::numeric_limits<T>::max();
};

class Allocation {
public:
  /// A unique identifier for shared memory buffers
  using BufferId = size_t;
  using BufferIdSetT = DenseSet<BufferId>;

  static constexpr BufferId InvalidBufferId =
      std::numeric_limits<BufferId>::max();

  /// Creates a new Allocation analysis that computes the shared memory
  /// information for all associated shared memory values.
  Allocation(Operation *operation) : operation(operation) { run(); }

  /// Returns the operation this analysis was constructed from.
  Operation *getOperation() const { return operation; }

  /// Returns the offset of the given buffer in the shared memory.
  size_t getOffset(BufferId bufferId) const {
    return bufferSet.lookup(bufferId).offset;
  }

  /// Returns the size of the given buffer in the shared memory.
  size_t getAllocatedSize(BufferId bufferId) const {
    return bufferSet.lookup(bufferId).size;
  }

  /// Returns the buffer id of the given value.
  /// This interface only returns the allocated buffer id.
  /// If you want to get all the buffer ids that are associated with the given
  /// value, including alias buffers, use getBufferIds.
  BufferId getBufferId(Value value) const {
    if (valueBuffer.count(value)) {
      return valueBuffer.lookup(value)->id;
    } else {
      return InvalidBufferId;
    }
  }

  /// Returns all the buffer ids of the given value, including alias buffers.
  BufferIdSetT getBufferIds(Value value) const {
    BufferIdSetT bufferIds;
    auto allocBufferId = getBufferId(value);
    if (allocBufferId != InvalidBufferId)
      bufferIds.insert(allocBufferId);
    for (auto *buffer : aliasBuffer.lookup(value)) {
      if (buffer->id != InvalidBufferId)
        bufferIds.insert(buffer->id);
    }
    return bufferIds;
  }

  /// Returns the scratch buffer id of the given value.
  BufferId getBufferId(Operation *operation) const {
    if (opScratch.count(operation)) {
      return opScratch.lookup(operation)->id;
    } else {
      return InvalidBufferId;
    }
  }

  /// Returns the size of total shared memory allocated
  size_t getSharedMemorySize() const { return sharedMemorySize; }

  bool isIntersected(BufferId lhsId, BufferId rhsId) const {
    if (lhsId == InvalidBufferId || rhsId == InvalidBufferId)
      return false;
    auto lhsBuffer = bufferSet.lookup(lhsId);
    auto rhsBuffer = bufferSet.lookup(rhsId);
    return lhsBuffer.intersects(rhsBuffer);
  }

private:
  /// A class that represents a shared memory buffer
  struct BufferT {
    enum class BufferKind { Explicit, Scratch };

    /// MT: thread-safe
    inline static std::atomic<BufferId> nextId = 0;

    BufferKind kind;
    BufferId id;
    size_t size;
    size_t offset;

    bool operator==(const BufferT &other) const { return id == other.id; }
    bool operator<(const BufferT &other) const { return id < other.id; }

    BufferT() : BufferT(BufferKind::Explicit) {}
    BufferT(BufferKind kind) : BufferT(kind, 0, 0) {}
    BufferT(BufferKind kind, size_t size) : BufferT(kind, size, 0) {}
    BufferT(BufferKind kind, size_t size, size_t offset)
        : kind(kind), id(nextId++), size(size), offset(offset) {}

    bool intersects(const BufferT &other) const {
      return Interval<size_t>(offset, offset + size)
          .intersects(
              Interval<size_t>(other.offset, other.offset + other.size));
    }
  };

  /// Op -> Scratch Buffer
  using OpScratchMapT = DenseMap<Operation *, BufferT *>;
  /// Value -> Explicit Buffer
  using ValueBufferMapT = llvm::MapVector<Value, BufferT *>;
  /// Value -> Alias Buffer
  using AliasBufferMapT = llvm::MapVector<Value, llvm::SetVector<BufferT *>>;
  /// BufferId -> Buffer
  using BufferSetT = DenseMap<BufferId, BufferT>;
  /// Runs allocation analysis on the given top-level operation.
  void run();

private:
  template <BufferT::BufferKind Kind, typename KeyType, typename... Args>
  void addBuffer(KeyType &key, Args &&...args) {
    auto buffer = BufferT(Kind, std::forward<Args>(args)...);
    bufferSet[buffer.id] = std::move(buffer);
    if constexpr (Kind == BufferT::BufferKind::Explicit) {
      valueBuffer[key] = &bufferSet[buffer.id];
    } else {
      opScratch[key] = &bufferSet[buffer.id];
    }
  }

  void addAlias(Value value, Value alloc) {
    aliasBuffer[value].insert(valueBuffer[alloc]);
  }

private:
  Operation *operation;
  OpScratchMapT opScratch;
  ValueBufferMapT valueBuffer;
  AliasBufferMapT aliasBuffer;
  BufferSetT bufferSet;
  size_t sharedMemorySize = 0;

  friend class triton::AllocationAnalysis;
};

} // namespace mlir

#endif // TRITON_ANALYSIS_ALLOCATION_H
