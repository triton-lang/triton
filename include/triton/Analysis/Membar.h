#ifndef TRITON_ANALYSIS_MEMBAR_H
#define TRITON_ANALYSIS_MEMBAR_H

#include "Allocation.h"
#include "BufferIndexAnalysis.h"
#include "BufferRegion.h"
#include "CallGraph.h"
#include "Function.h"

#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <optional>
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
  // Create allocation slice from a value, collecting subslice offsets.
  // Dynamic buffer-index information is attached by BufferIndexAnalysis; use
  // BufferIndexAnalysis::makeSlice when constructing slices for membar.
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

  /// Translate shared-memory geometry and discard callee-local index facts.
  AllocationSlice
  translateToCallsite(CallOpInterface call, FunctionOpInterface callee,
                      triton::BufferRegionAnalysis &regions) const;

  void print(raw_ostream &os) const;

  void invalidateIterationInfo() {
    bufferIndexExpr = nullptr;
    subsliceSource = {};
  }

  // Immutable MAY-addresses covering every dynamic instance of this access.
  // They remain valid across backedges, unlike the epoch-relative facts above.
  // The owning BufferRegionAnalysis outlives every slice using this pointer.
  const triton::BufferRegionFootprint *physicalFootprint = nullptr;

  // Buffer-index expression attached by BufferIndexAnalysis. It participates
  // in ordering/equality so accesses to different slots remain separate.
  // Must not be mutated after the slice is inserted into a sorted container
  // (e.g. BlockInfo::SliceMapT); rebuild the container instead, as
  // BlockInfo::invalidateIterationInfo does.
  const BufferIndexExpr *bufferIndexExpr = nullptr;

  // Parameters have no callee buffer ID. Bind this argument's effects to the
  // caller's allocation when importing the function summary.
  std::optional<unsigned> argumentIndex;

private:
  std::tuple<Interval<size_t>, Allocation::BufferId, const void *,
             llvm::ArrayRef<int32_t>, const BufferIndexExpr *, const void *,
             const void *, std::optional<unsigned>>
  asTuple() const {
    return {allocationInterval,
            bufferId,
            accessTy.getAsOpaquePointer(),
            subsliceOffsets,
            bufferIndexExpr,
            subsliceSource.getAsOpaquePointer(),
            physicalFootprint,
            argumentIndex};
  }
  // Offsets from subslice, borrowed from its immutable context-owned attribute.
  // Empty when offsets are unknown.
  llvm::ArrayRef<int32_t> subsliceOffsets;
  // The source descriptor supplying the coordinates for subslice offsets.
  Value subsliceSource;
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

  template <typename Transform> void transformSlices(Transform transform) {
    for (auto *slices : {&syncReadSlices, &syncWriteSlices}) {
      SliceMapT transformed;
      for (const auto &[slice, ops] : *slices)
        for (const AllocationSlice &mapped : transform(slice)) {
          auto &dst = transformed[mapped];
          dst.insert(ops.begin(), ops.end());
        }
      *slices = std::move(transformed);
    }
  }

  /// Clears the buffer index and access origin of every slice, rebuilding both
  /// maps. Used at loop backedges where the same SSA value can denote a value
  /// from a different dynamic iteration, and before storing function summaries
  /// where per-function SSA identity is no longer meaningful.
  void invalidateIterationInfo();

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

/// Tracks the shared-memory state needed at the current program point and at
/// function boundaries.
struct MembarInfo {
  /// Buffers accessed since the most recent synchronization.
  BlockInfo pending;

  /// Buffers reachable from the function entry block before the first
  /// synchronization.
  /// It keeps incrementing during the iterative algorithm until
  ///  we note all paths to a basic block has synchronized.
  BlockInfo entryBlockInfo;

  /// Whether every path from the function entry block to the current program
  /// point has synchronized.
  bool allPathsFromEntrySynced = false;

  MembarInfo &join(const MembarInfo &other) {
    pending.join(other.pending);
    entryBlockInfo.join(other.entryBlockInfo);
    allPathsFromEntrySynced &= other.allPathsFromEntrySynced;
    return *this;
  }

  void addBlockInfo(const BlockInfo &blockInfo) {
    if (!allPathsFromEntrySynced)
      entryBlockInfo.join(blockInfo);
    pending.join(blockInfo);
  }

  void sync() {
    pending.sync();
    allPathsFromEntrySynced = true;
  }

  void applyCallSummary(const MembarInfo &callee) {
    if (!allPathsFromEntrySynced)
      entryBlockInfo.join(callee.entryBlockInfo);
    if (callee.allPathsFromEntrySynced)
      sync();
    pending.join(callee.pending);
  }

  template <typename Transform> void transformSlices(Transform transform) {
    pending.transformSlices(transform);
    entryBlockInfo.transformSlices(transform);
  }

  bool operator==(const MembarInfo &other) const {
    return pending == other.pending && entryBlockInfo == other.entryBlockInfo &&
           allPathsFromEntrySynced == other.allPathsFromEntrySynced;
  }
};

/// Classify the barriers that synchronize local memory accesses in `op`
/// relative to its memory effects.
triton::BarrierStages getLocalBarrierStages(Operation *op,
                                            Allocation *allocation);

//===----------------------------------------------------------------------===//
// Shared Memory Barrier Analysis
//===----------------------------------------------------------------------===//

class MembarAnalysis : public triton::PostOrderFunctionAnalysis<MembarInfo> {
public:
  enum class AccessMode { AllSharedAccesses, AllocatorAliasesOnly };

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
  MembarAnalysis(Allocation &allocation, MembarFilterFn filter,
                 triton::BufferRegionAnalysis &regions,
                 MembarSliceFilterFn sliceFilter = nullptr,
                 AccessMode accessMode = AccessMode::AllSharedAccesses)
      : allocation(allocation), filter(std::move(filter)), regions(regions),
        sliceFilter(std::move(sliceFilter)), accessMode(accessMode),
        bufferIndexAnalysis(
            cast<FunctionOpInterface>(allocation.getOperation())) {}

private:
  /// Updates the BlockInfo operation based on the operation.
  void update(Operation *operation, MembarInfo *membarInfo, FuncMapT *funcMap,
              OpBuilder *builder) override;

  void updateSuccessor(Operation *terminator, Block *successor,
                       MembarInfo *membarInfo) override;

  void updateExitState(MembarInfo *membarInfo) override;

protected:
  void updateMemoryEffects(Operation *operation, MembarInfo *membarInfo,
                           FuncMapT *funcMap, OpBuilder *builder,
                           bool cluster = false);
  void syncIfNeeded(Operation *operation, const BlockInfo &effects,
                    MembarInfo *membarInfo, OpBuilder *builder,
                    bool cluster = false);
  void insertBarrier(Operation *operation, OpBuilder *builder,
                     bool cluster = false);
  virtual triton::BarrierStages getBarrierStages(Operation *operation);

  Allocation &allocation;
  MembarFilterFn filter;
  triton::BufferRegionAnalysis &regions;

private:
  SmallVector<AllocationSlice> getAllocationSlices(Value value);

  MembarSliceFilterFn sliceFilter;
  AccessMode accessMode;
  BufferIndexAnalysis bufferIndexAnalysis;
};

/// Inserts shared-memory barriers across a module. Function summaries retain
/// entry-prefix and pending exit states for calls.
class ModuleMembarAnalysis {
public:
  ModuleMembarAnalysis(ModuleAllocation &moduleAllocation,
                       MembarFilterFn filter = nullptr)
      : moduleAllocation(moduleAllocation), filter(std::move(filter)) {}

  template <typename AnalysisT = MembarAnalysis> void run() {
    // Geometry and interval slices use the same completed allocation, including
    // backend scratch sizes and shared-memory partition offsets.
    auto solver = createDataFlowSolver();
    auto *regions = solver->load<triton::BufferRegionAnalysis>(
        triton::BufferRegionAnalysis::Mode::AllMemory, &moduleAllocation);
    if (failed(solver->initializeAndRun(moduleAllocation.getModuleOp())))
      llvm::report_fatal_error("failed to analyze allocated buffer regions");
    runAnalysis<AnalysisT>(*regions);
  }

  template <typename AnalysisT>
  void runAnalysis(triton::BufferRegionAnalysis &regions) {
    AnalysisT::runModule(
        moduleAllocation.getModuleOp(), [&](FunctionOpInterface function) {
          auto &allocation = *moduleAllocation.getFuncData(function);
          return AnalysisT(allocation, filter, regions);
        });
  }

private:
  ModuleAllocation &moduleAllocation;
  MembarFilterFn filter;
};

} // namespace mlir

#endif // TRITON_ANALYSIS_MEMBAR_H
