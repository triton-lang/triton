#ifndef TRITON_ANALYSIS_MEMBAR_H
#define TRITON_ANALYSIS_MEMBAR_H

#include "Allocation.h"
#include "CallGraph.h"
#include "Function.h"

#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <set>
#include <tuple>
#include <utility>

namespace mlir {

class OpBuilder;
struct AllocationSlice;

/// Callback to allow backend to provide more information on whether a barrier
/// is needed between two operations. Even though two operations access the same
/// shared memory they may not require a barrier in between them.
using MembarFilterFn =
    std::function<bool(Operation *, Operation *, bool /*lhsIsRead*/,
                       bool /*rhsIsRead*/, Allocation *)>;

/// Slice-level filter to allow backends to ignore specific aliasing cases.
using MembarSliceFilterFn =
    std::function<bool(const AllocationSlice &, const AllocationSlice &,
                       bool /*lhsIsRead*/, bool /*rhsIsRead*/, Allocation *)>;

// Represents the access to a slice of an allocation
// It contains information both on physical memory (the interval) and a
// logical view on it (layout, subslice offsets and shape for the access)
struct AllocationSlice {
public:
  // Create allocation slice from a value, collecting subslice offsets
  AllocationSlice(Value value, Interval<size_t> allocationInterval,
                  Allocation::BufferId bufferId);

  // Builder for accesses that represent accesses to the whole
  // allocation (scratch buffers, ArriveBarrierOp, ..)
  AllocationSlice(Interval<size_t> interval)
      : allocationInterval(interval), accessTy(nullptr),
        bufferId(Allocation::InvalidBufferId) {}

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

  Allocation::BufferId getBufferId() const { return bufferId; }

  AllocationSlice translated(size_t offset,
                             bool invalidateBufferId = false) const {
    AllocationSlice shifted = *this;
    shifted.allocationInterval = Interval<size_t>(
        allocationInterval.start() + offset, allocationInterval.end() + offset);
    if (invalidateBufferId)
      shifted.bufferId = Allocation::InvalidBufferId;
    return shifted;
  }

  void print(raw_ostream &os) const;

private:
  std::tuple<Interval<size_t>, Allocation::BufferId, const void *,
             llvm::ArrayRef<int64_t>>
  asTuple() const {
    return {allocationInterval, bufferId, accessTy.getAsOpaquePointer(),
            subsliceOffsets};
  }
  // Offsets from subslice. Empty when offsets are unknown
  SmallVector<int64_t> subsliceOffsets;
  // The allocated interval for this buffer
  Interval<size_t> allocationInterval;
  // Type of the memory descriptor for this access
  triton::gpu::MemDescType accessTy;
  // Buffer id for partial sync on wait_barrier deps.
  Allocation::BufferId bufferId;
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
  bool isIntersected(const BlockInfo &other, MembarFilterFn filter,
                     Allocation *allocation,
                     MembarSliceFilterFn sliceFilter = nullptr) const {
    return /*RAW*/ isIntersected(syncWriteSlices, other.syncReadSlices,
                                 /*lhsIsRead=*/false, /*rhsIsRead=*/true,
                                 filter, sliceFilter, allocation) ||
           /*WAR*/
           isIntersected(syncReadSlices, other.syncWriteSlices,
                         /*lhsIsRead=*/true, /*rhsIsRead=*/false, filter,
                         sliceFilter, allocation) ||
           /*WAW*/
           isIntersected(syncWriteSlices, other.syncWriteSlices,
                         /*lhsIsRead=*/false, /*rhsIsRead=*/false, filter,
                         sliceFilter, allocation);
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
                     bool lhsIsRead, bool rhsIsRead, MembarFilterFn filter,
                     MembarSliceFilterFn sliceFilter,
                     Allocation *allocation) const {
    for (auto &lhs : lhsSlices)
      for (auto &rhs : rhsSlices)
        if (lhs.first.intersects(rhs.first))
          if (!sliceFilter || !sliceFilter(lhs.first, rhs.first, lhsIsRead,
                                           rhsIsRead, allocation))
            for (auto lhsOp : lhs.second)
              for (auto rhsOp : rhs.second)
                if (!filter ||
                    !filter(lhsOp, rhsOp, lhsIsRead, rhsIsRead, allocation))
                  return true;
    return false;
  }
};

inline BlockInfo translateBlockInfoToCallsite(const BlockInfo &calleeBlockInfo,
                                              size_t callOffset) {
  BlockInfo translatedBlockInfo;
  auto translateSlices = [&](const BlockInfo::SliceMapT &srcSlices,
                             BlockInfo::SliceMapT &dstSlices) {
    for (const auto &[slice, ops] : srcSlices) {
      auto translatedSlice =
          slice.translated(callOffset, /*invalidateBufferId=*/true);
      auto &dstOps = dstSlices[translatedSlice];
      dstOps.insert(ops.begin(), ops.end());
    }
  };

  translateSlices(calleeBlockInfo.syncReadSlices,
                  translatedBlockInfo.syncReadSlices);
  translateSlices(calleeBlockInfo.syncWriteSlices,
                  translatedBlockInfo.syncWriteSlices);
  return translatedBlockInfo;
}

/// Returns true if `op` synchronizes local memory accesses for membar-style
/// analyses.
bool containsLocalBarrier(Operation *op);

//===----------------------------------------------------------------------===//
// Shared Memory Barrier Analysis
//===----------------------------------------------------------------------===//

/// Common allocation and filtering state for postorder memory-synchronization
/// analyses.
class MembarOrFenceAnalysis
    : public triton::PostOrderFunctionAnalysis<BlockInfo> {
public:
  MembarOrFenceAnalysis(Allocation &allocation, MembarFilterFn filter)
      : allocation(allocation), filter(std::move(filter)) {}

protected:
  Allocation &allocation;
  MembarFilterFn filter;
};

class MembarAnalysis : public MembarOrFenceAnalysis {
public:
  using MembarOrFenceAnalysis::MembarOrFenceAnalysis;

private:
  /// Inserts a local barrier when the current access conflicts with an
  /// unsynchronized access: read-after-write, write-after-read, or
  /// write-after-write. Read-after-read accesses remain on the frontier.
  void update(Operation *operation, BlockInfo *blockInfo, FuncMapT *funcMap,
              OpBuilder *builder) override;

  void insertBarrier(Operation *operation, OpBuilder *builder);
};

/// Runs one synchronization analysis per function in callgraph postorder.
/// Each function summary contains the unsynchronized access frontier at its
/// exits, so callers can incorporate callee effects without placing barriers
/// unconditionally around every call.
template <typename AnalysisT>
class ModuleMembarOrFenceAnalysis : public triton::CallGraph<BlockInfo> {
public:
  ModuleMembarOrFenceAnalysis(ModuleAllocation &moduleAllocation,
                              MembarFilterFn filter = nullptr)
      : triton::CallGraph<BlockInfo>(moduleAllocation.getModuleOp()),
        moduleAllocation(moduleAllocation), filter(std::move(filter)) {}

  void run() {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        // Pre-order walk callback
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        // Post-order walk callback
        [&](FunctionOpInterface funcOp) {
          auto &allocation = *moduleAllocation.getFuncData(funcOp);
          if (!funcMap.try_emplace(funcOp).second)
            return;
          AnalysisT(allocation, filter).run(funcOp, funcMap);
        });
  }

private:
  ModuleAllocation &moduleAllocation;
  MembarFilterFn filter;
};

using ModuleMembarAnalysis = ModuleMembarOrFenceAnalysis<MembarAnalysis>;

} // namespace mlir

#endif // TRITON_ANALYSIS_MEMBAR_H
