#ifndef TRITON_ANALYSIS_MEMBAR_H
#define TRITON_ANALYSIS_MEMBAR_H

#include "Allocation.h"

#include "mlir/IR/Dominance.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <set>
#include <tuple>

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

/// A buffer index decomposed as `baseValue + constantOffset`, optionally
/// under a known modulus (recorded when the index is wrapped by remsi or
/// the pipeliner's select/cmpi idiom). Used to prove that two
/// multi-buffered shared-memory accesses target different slots.
///
/// Correctness relies on SSA identity: two expressions with the same
/// baseValue refer to the same runtime integer only within a single
/// execution of the enclosing region. Across a loop backedge the same SSA
/// value (e.g. an scf.for iter_arg) denotes different runtime values on
/// different iterations; those cases are handled separately via the
/// isLoopCarried flag on AllocationSlice, which suppresses this shortcut.
struct BufferIndexExpr {
  Value baseValue;
  int64_t constantOffset = 0;
  std::optional<int64_t> modulus;

  bool hasBase() const { return baseValue != nullptr; }

  /// Returns true only if both expressions provably refer to different buffer
  /// slots: same base value, and offsets that differ (modulo the recorded
  /// modulus when one is known).
  bool isProvablyDifferentFrom(const BufferIndexExpr &other) const {
    if (baseValue != other.baseValue)
      return false;
    if (modulus || other.modulus) {
      if (modulus != other.modulus)
        return false;
      int64_t m = *modulus;
      // Euclidean normalization: map each offset into [0, m) so that
      // negative constants compare correctly against positive ones.
      int64_t a = ((constantOffset % m) + m) % m;
      int64_t b = ((other.constantOffset % m) + m) % m;
      return a != b;
    }
    return constantOffset != other.constantOffset;
  }

  bool operator==(const BufferIndexExpr &other) const {
    return baseValue == other.baseValue &&
           constantOffset == other.constantOffset && modulus == other.modulus;
  }

  bool operator<(const BufferIndexExpr &other) const {
    if (baseValue.getAsOpaquePointer() != other.baseValue.getAsOpaquePointer())
      return baseValue.getAsOpaquePointer() <
             other.baseValue.getAsOpaquePointer();
    if (constantOffset != other.constantOffset)
      return constantOffset < other.constantOffset;
    return modulus < other.modulus;
  }
};

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

  /// Returns a copy of this slice with isLoopCarried set. Used when
  /// propagating BlockInfo across a CFG backedge; see BufferIndexExpr for
  /// the SSA-identity invariant this flag protects.
  AllocationSlice createLoopCarriedCopy() const {
    AllocationSlice copy = *this;
    copy.isLoopCarried = true;
    return copy;
  }

  bool getIsLoopCarried() const { return isLoopCarried; }

  void print(raw_ostream &os) const;

private:
  std::tuple<Interval<size_t>, Allocation::BufferId, const void *,
             llvm::ArrayRef<int64_t>, std::optional<BufferIndexExpr>, bool>
  asTuple() const {
    return {allocationInterval, bufferId,        accessTy.getAsOpaquePointer(),
            subsliceOffsets,    bufferIndexExpr, isLoopCarried};
  }
  // Offsets from subslice. Empty when offsets are unknown
  SmallVector<int64_t> subsliceOffsets;
  // The allocated interval for this buffer
  Interval<size_t> allocationInterval;
  // Type of the memory descriptor for this access
  triton::gpu::MemDescType accessTy;
  // Buffer id for partial sync on wait_barrier deps.
  Allocation::BufferId bufferId;
  // Buffer index expression for multi-buffered accesses via MemDescIndexOp.
  std::optional<BufferIndexExpr> bufferIndexExpr;
  bool isLoopCarried = false;
};

struct BlockInfo {
  using SliceMapT = std::map<AllocationSlice, std::set<Operation *>>;

  SliceMapT syncReadSlices;
  SliceMapT syncWriteSlices;

  BlockInfo() = default;

  /// Unions two BlockInfo objects.
  BlockInfo &join(const BlockInfo &other) {
    joinSlices(syncReadSlices, other.syncReadSlices);
    joinSlices(syncWriteSlices, other.syncWriteSlices);
    return *this;
  }

  /// Unions two BlockInfo objects, marking all incoming slices as loop-carried.
  /// Used when propagating state across loop backedges.
  BlockInfo &joinLoopCarried(const BlockInfo &other) {
    joinSlicesAsLoopCarried(syncReadSlices, other.syncReadSlices);
    joinSlicesAsLoopCarried(syncWriteSlices, other.syncWriteSlices);
    return *this;
  }

  void dump() {
    auto &err = llvm::errs();
    err << "Block Interval:\n";
    err << "  Read Intervals:\n";
    for (auto &[slice, ops] : syncReadSlices) {
      err << "    ";
      slice.print(err);
      if (slice.getIsLoopCarried())
        err << " [loop-carried]";
      err << " ";
      for (auto &op : ops)
        err << op->getName() << " ";
      err << "\n";
    }
    err << "  Write Intervals:\n";
    for (auto &[slice, ops] : syncWriteSlices) {
      err << "    ";
      slice.print(err);
      if (slice.getIsLoopCarried())
        err << " [loop-carried]";
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
  static void joinSlices(SliceMapT &lhs, const SliceMapT &rhs) {
    for (const auto &[slice, ops] : rhs)
      lhs[slice].insert(ops.begin(), ops.end());
  }

  static void joinSlicesAsLoopCarried(SliceMapT &lhs, const SliceMapT &rhs) {
    for (const auto &[slice, ops] : rhs) {
      AllocationSlice loopCarriedSlice = slice.createLoopCarriedCopy();
      lhs[loopCarriedSlice].insert(ops.begin(), ops.end());
    }
  }

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

//===----------------------------------------------------------------------===//
// Shared Memory Barrier Analysis
//===----------------------------------------------------------------------===//

// Common class to analyze membar and fence placement.
class MembarOrFenceAnalysis {
  using VirtualBlock = std::pair<Block *, Block::iterator>;
  struct SuccessorInfo {
    VirtualBlock block;
    /// True when this edge is a loop backedge, defined as the target
    /// dominating the source block. Backedges propagate slices as
    /// loop-carried (see BlockInfo::joinLoopCarried).
    bool isBackedge = false;
  };

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
  void resolve(FunctionOpInterface funcOp, FuncBlockInfoMapT *funcBlockInfoMap,
               OpBuilder *builder);

  /// Collects the successors of the terminator and populates
  /// SuccessorInfo::isBackedge for each. Dispatches on the op's
  /// control-flow interface (`BranchOpInterface`,
  /// `RegionBranchOpInterface`, or `RegionBranchTerminatorOpInterface`).
  void visitTerminator(Operation *operation, DominanceInfo &domInfo,
                       SmallVector<SuccessorInfo> &successors);

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
      : MembarOrFenceAnalysis(allocation, filter) {}

  ~MembarAnalysis() override = default;

private:
  /// Updates the BlockInfo operation based on the operation.
  virtual void update(Operation *operation, BlockInfo *blockInfo,
                      FuncBlockInfoMapT *funcBlockInfoMap,
                      OpBuilder *builder) override;

  void insertBarrier(Operation *operation, OpBuilder *builder);
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
