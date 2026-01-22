#ifndef TRITON_ANALYSIS_MEMBAR_H
#define TRITON_ANALYSIS_MEMBAR_H

#include "Allocation.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>

namespace mlir {

class OpBuilder;

/// Callback to allow backend to provide more information on whether a barrier
/// is needed between two operations. Even though two operations access the same
/// shared memory they may not require a barrier in between them.
using MembarFilterFn = std::function<bool(Operation *, Operation *)>;

// Represents the access to a slice of an allocation
// It contains information both on physical memory (the interval) and a
// logical view on it (layout, subslice offsets and shape for the access)
class AllocationSlice {
public:
  // Models an offset that is possibly unknown/dynamic
  struct OffsetValue {
    int64_t offset;
    OffsetValue() : offset(-1) {}
    OffsetValue(int64_t offset) : offset(offset) {}
    bool isKnown() const { return offset >= 0; }
    bool knownLeq(const OffsetValue &other) const {
      if (!isKnown() || !other.isKnown())
        return false;
      return offset <= other.offset;
    }

    void print(raw_ostream &os) const;

    OffsetValue &operator+=(OffsetValue rhs) {
      if (!rhs.isKnown() || !isKnown()) {
        offset = -1;
      } else {
        offset += rhs.offset;
      }
      return *this;
    }
    friend OffsetValue operator+(OffsetValue lhs, OffsetValue rhs) {
      lhs += rhs;
      return lhs;
    }
    friend raw_ostream &operator<<(raw_ostream &os, const OffsetValue &arg) {
      arg.print(os);
      return os;
    }
  };

  // Create allocation slice from a new allocation
  AllocationSlice(triton::gpu::MemDescType allocTy,
                  Interval<size_t> allocationInterval);
  // Create allocation slice from an unknown value, collecting subslice offsets
  AllocationSlice(triton::gpu::MemDescType allocTy,
                  Interval<size_t> allocationInterval,
                  ArrayRef<int64_t> curShape);

  // Builder for accesses that represent accesses to the whole
  // allocation (scratch buffers, ArriveBarrierOp, ..)
  AllocationSlice(Interval<size_t> interval)
      : allocationInterval(interval), allocTy(nullptr) {}

  AllocationSlice subslice(ArrayRef<int32_t> offsets,
                           ArrayRef<int64_t> resShape) const;
  AllocationSlice index(Value index, ArrayRef<int64_t> resShape) const;

  bool operator<(const AllocationSlice &other) const {
    return asTuple() < other.asTuple();
  }

  bool operator==(const AllocationSlice &other) const {
    return asTuple() == other.asTuple();
  }

  // Check if a AllocationSlice intersects with another other.
  // This happens if their subslice regions intersect in all dimensions.
  // Returns true if it can't prove the AllocationSlices are disjoint.
  bool intersects(const AllocationSlice &other) const;

  void print(raw_ostream &os) const;

private:
  std::tuple<Interval<size_t>, const void *, llvm::ArrayRef<int64_t>,
             llvm::ArrayRef<int64_t>>
  asTuple() const {
    llvm::ArrayRef<int64_t> offsetData(
        reinterpret_cast<const int64_t *>(subsliceOffsets.data()),
        subsliceOffsets.size());
    return {allocationInterval, allocTy.getAsOpaquePointer(), subsliceShape,
            offsetData};
  }
  // Offsets of subslice. Empty when offsets are unknown.
  SmallVector<OffsetValue> subsliceOffsets;
  // Shape of the current slice
  SmallVector<int64_t> subsliceShape;
  // The allocated interval for this buffer
  Interval<size_t> allocationInterval;
  // Type of the original allocation, before any slicing
  triton::gpu::MemDescType allocTy;
};

struct ValueHasher {
  size_t operator()(mlir::Value v) const {
    return std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(v.getImpl()));
  }
};

class AllocationSliceAnalysis {
  Allocation *allocation;
  // We use std::unordered_map for reference stability of the values
  std::unordered_map<Value, std::vector<AllocationSlice>, ValueHasher> sliceMap;
  DenseMap<Allocation::BufferId, triton::gpu::MemDescType> allocTypeMap;

public:
  AllocationSliceAnalysis() : allocation(nullptr) {}
  AllocationSliceAnalysis(Allocation *allocation) : allocation(allocation) {}
  void update(Operation *op);
  const std::vector<AllocationSlice> &getAllocationSlices(Value value);
};

struct BlockInfo {
  using SliceMapT = std::map<AllocationSlice, std::set<Operation *>>;

  SliceMapT syncReadSlices;
  SliceMapT syncWriteSlices;

  BlockInfo() = default;

  /// Unions two BlockInfo objects.
  BlockInfo &join(const BlockInfo &other) {
    for (auto &slice : other.syncReadSlices)
      syncReadSlices[slice.first].insert(slice.second.begin(),
                                         slice.second.end());

    for (auto &slice : other.syncWriteSlices)
      syncWriteSlices[slice.first].insert(slice.second.begin(),
                                          slice.second.end());
    return *this;
  }

  void dump() {
    auto &err = llvm::errs();
    err << "Block Interval:\n";
    err << "  Read Intervals:\n";
    for (auto &[slice, ops] : syncReadSlices) {
      err << "    ";
      slice.print(err);
      err << " ";
      for (auto &op : ops)
        err << op->getName() << " ";
      err << "\n";
    }
    err << "  Write Intervals:\n";
    for (auto &[slice, ops] : syncWriteSlices) {
      err << "    ";
      slice.print(err);
      err << " ";
      for (auto &op : ops)
        err << op->getName() << " ";
      err << "\n";
    }
  }

  /// Returns true if Slices in two BlockInfo objects are intersected.
  bool isIntersected(const BlockInfo &other, MembarFilterFn filter) const {
    return /*RAW*/ isIntersected(syncWriteSlices, other.syncReadSlices,
                                 filter) ||
           /*WAR*/
           isIntersected(syncReadSlices, other.syncWriteSlices, filter) ||
           /*WAW*/
           isIntersected(syncWriteSlices, other.syncWriteSlices, filter);
  }

  /// Clears the slices because a barrier is inserted.
  void sync() {
    syncReadSlices.clear();
    syncWriteSlices.clear();
  }

  /// Compares two BlockInfo objects.
  bool operator==(const BlockInfo &other) const {
    return syncReadSlices == other.syncReadSlices &&
           syncWriteSlices == other.syncWriteSlices;
  }

  bool operator!=(const BlockInfo &other) const { return !(*this == other); }

private:
  bool isIntersected(const SliceMapT &lhsSlices, const SliceMapT &rhsSlices,
                     MembarFilterFn filter) const {
    for (auto &lhs : lhsSlices)
      for (auto &rhs : rhsSlices)
        if (lhs.first.intersects(rhs.first))
          for (auto lhsOp : lhs.second)
            for (auto rhsOp : rhs.second)
              if (!filter || !filter(lhsOp, rhsOp))
                return true;
    return false;
  }
};

//===----------------------------------------------------------------------===//
// Shared Memory Barrier Analysis
//===----------------------------------------------------------------------===//

// Common class to analyze membar and fence placement.
class MembarOrFenceAnalysis {
  using VirtualBlock = std::pair<Block *, Block::iterator>;

public:
  using FuncBlockInfoMapT = triton::CallGraph<BlockInfo>::FuncDataMapT;
  /// Creates a new Membar analysis that generates the shared memory barrier
  /// in the following circumstances:
  /// - RAW: If a shared memory write is followed by a shared memory read, and
  /// their addresses are intersected, a barrier is inserted.
  /// - WAR: If a shared memory read is followed by a shared memory write, and
  /// their addresses are intersected, a barrier is inserted.
  /// The following circumstances do not require a barrier:
  /// - WAW: not possible because overlapped memory allocation is not allowed.
  /// - RAR: no write is performed.
  /// Temporary storage of operations such as Reduce are considered as both
  /// a shared memory read. If the temporary storage is written but not read,
  /// it is considered as the problem of the operation itself but not the membar
  /// analysis.
  MembarOrFenceAnalysis() = default;
  explicit MembarOrFenceAnalysis(Allocation *allocation, MembarFilterFn filter)
      : allocation(allocation), filter(filter) {}

  virtual ~MembarOrFenceAnalysis() = default;

  /// Runs the membar analysis to the given operation, inserts a barrier if
  /// necessary.
  void run(FuncBlockInfoMapT &funcBlockInfoMap);

protected:
  /// Applies the barrier analysis based on the SCF dialect, in which each
  /// region has a single basic block only.
  /// Example:
  /// region1
  ///   op1
  ///   op2 (scf.if)
  ///      region2
  ///        op3
  ///        op4
  ///      region3
  ///        op5
  ///        op6
  ///   op7
  /// TODO: Explain why we don't use ForwardAnalysis:
  void resolve(FunctionOpInterface funcOp, FuncBlockInfoMapT *funcBlockInfoMap,
               OpBuilder *builder);

  /// Collects the successors of the terminator
  void visitTerminator(Operation *operation,
                       SmallVector<VirtualBlock> &successors);

  /// Updates the BlockInfo operation based on the operation.
  virtual void update(Operation *operation, BlockInfo *blockInfo,
                      FuncBlockInfoMapT *funcBlockInfoMap,
                      OpBuilder *builder) = 0;

  Allocation *allocation = nullptr;
  MembarFilterFn filter = nullptr;
};

class MembarAnalysis : public MembarOrFenceAnalysis {
public:
  MembarAnalysis() = default;
  explicit MembarAnalysis(Allocation *allocation, MembarFilterFn filter)
      : MembarOrFenceAnalysis(allocation, filter), sliceAnalysis(allocation) {}

  ~MembarAnalysis() override = default;

private:
  /// Updates the BlockInfo operation based on the operation.
  virtual void update(Operation *operation, BlockInfo *blockInfo,
                      FuncBlockInfoMapT *funcBlockInfoMap,
                      OpBuilder *builder) override;

  void insertBarrier(Operation *operation, OpBuilder *builder);

  AllocationSliceAnalysis sliceAnalysis;
};

/// Postorder traversal on the callgraph to insert membar instructions
/// of each function.
/// Each function maintains a BlockInfo map that includes all potential buffers
/// after returning. This way users do not have to explicitly insert membars
/// before and after function calls, but might be a bit conservative.
template <typename AnalysisType>
class ModuleMembarOrFenceAnalysis : public triton::CallGraph<BlockInfo> {
public:
  ModuleMembarOrFenceAnalysis(ModuleAllocation *moduleAllocation,
                              MembarFilterFn filter = nullptr)
      : triton::CallGraph<BlockInfo>(moduleAllocation->getModuleOp()),
        moduleAllocation(moduleAllocation), filter(filter) {}

  void run() {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        // Pre-order walk callback
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        // Post-order walk callback
        [&](FunctionOpInterface funcOp) {
          auto *allocation = moduleAllocation->getFuncData(funcOp);
          auto [it, inserted] = funcMap.try_emplace(funcOp, BlockInfo());
          if (inserted) {
            AnalysisType analysis(allocation, filter);
            analysis.run(funcMap);
          }
        });
  }

private:
  ModuleAllocation *moduleAllocation;
  MembarFilterFn filter;
};

typedef ModuleMembarOrFenceAnalysis<MembarAnalysis> ModuleMembarAnalysis;

} // namespace mlir

#endif // TRITON_ANALYSIS_MEMBAR_H
