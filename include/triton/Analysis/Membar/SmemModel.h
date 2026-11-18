#ifndef TRITON_ANALYSIS_MEMBAR_SMEMMODEL_H
#define TRITON_ANALYSIS_MEMBAR_SMEMMODEL_H

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferIndexAnalysis.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

namespace mlir {

struct MembarInfo;

struct AllocationSlice;

/// Callback to allow backend to provide more information on whether a barrier
/// is needed between two operations. Even though two operations access the same
/// shared memory they may not require a barrier in between them.
using MembarFilterFn = std::function<bool(
    Operation *, Operation *, bool /*lhsIsRead*/, bool /*rhsIsRead*/,
    Allocation *, const AllocationSlice &, const AllocationSlice &)>;

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

  // Effect classification is preserved when geometry is widened or translated.
  // Unknown for implicit scratch and non-shared accesses.
  std::optional<triton::gpu::SharedKind> sharedKind;

  // Buffer-index expression attached by BufferIndexAnalysis. It participates
  // in ordering/equality so accesses to different slots remain separate.
  // Must not be mutated after the slice is inserted into a sorted container
  // (e.g. BlockInfo::SliceMapT); rebuild the container instead, as
  // BlockInfo::invalidateIterationInfo does.
  const BufferIndexExpr *bufferIndexExpr = nullptr;

  // Parameters have no callee buffer ID. Bind this argument's effects to the
  // caller's allocation when importing the function summary.
  // TODO: Remove this field and Allocation's argumentAliases table with the
  // legacy cluster call summaries.
  std::optional<unsigned> argumentIndex;

private:
  std::tuple<Interval<size_t>, Allocation::BufferId, const void *,
             llvm::ArrayRef<int32_t>, const BufferIndexExpr *, const void *,
             const void *, std::optional<unsigned>,
             std::optional<triton::gpu::SharedKind>>
  asTuple() const {
    return {allocationInterval,
            bufferId,
            accessTy.getAsOpaquePointer(),
            subsliceOffsets,
            bufferIndexExpr,
            subsliceSource.getAsOpaquePointer(),
            physicalFootprint,
            argumentIndex,
            sharedKind};
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

// Empty summaries contain no operations. Nonempty summaries preserve an issuer
// only when every operation has the same known kind.
enum class ThreadIssuer : uint8_t {
  // No contributing operations.
  None,
  // The issuer is unknown or differs across operations.
  Unknown,
  // Thread zero of the current execution partition.
  Thread0,
  // A fixed leader of partition-relative warp zero.
  // The leader's lane may be nonzero.
  Warp0Leader
};

enum class CompletionSync : uint8_t {
  None,
  // Independent global reads may proceed before the rendezvous.
  Shared,
  All,
};

/// Completion scope, refined by a backend before joining pending state.
using CompletionScopeFn = CompletionSync (*)(Operation *, CompletionSync);

enum class ThreadDemandKind : uint8_t {
  None,
  GlobalReadsOnly,
  // Base membar demands not proven to have only global-read effects.
  General
};

struct BlockInfo {
  using SliceMapT = std::map<AllocationSlice, std::set<Operation *>>;

  SliceMapT syncReadSlices;
  SliceMapT syncWriteSlices;

  // Thread ordering is independent of which physical addresses are accessed.
  struct ThreadSyncState {
    ThreadIssuer effectIssuer = ThreadIssuer::None;
    CompletionSync completion = CompletionSync::None;
    bool peerWaitNeedsSync = false;

    bool operator==(const ThreadSyncState &) const = default;
  } threadSync;
  struct ThreadDemandState {
    ThreadDemandKind kind = ThreadDemandKind::None;
    // Only publication demands contribute to this issuer summary.
    ThreadIssuer publicationIssuer = ThreadIssuer::None;
    bool mayNotifyPeer = false;

    bool operator==(const ThreadDemandState &) const = default;
  } threadDemands;

  BlockInfo() = default;

  bool hasEffects() const {
    return threadSync.effectIssuer != ThreadIssuer::None ||
           !syncReadSlices.empty() || !syncWriteSlices.empty();
  }

  /// Unions two BlockInfo objects.
  BlockInfo &join(const BlockInfo &other) {
    for (auto &slice : other.syncReadSlices)
      syncReadSlices[slice.first].insert(slice.second.begin(),
                                         slice.second.end());

    for (auto &slice : other.syncWriteSlices)
      syncWriteSlices[slice.first].insert(slice.second.begin(),
                                          slice.second.end());
    joinThreadState(other);
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
  bool isIntersected(const BlockInfo &other, const MembarFilterFn &filter,
                     Allocation *allocation,
                     const MembarSliceFilterFn &sliceFilter = nullptr) const {
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

  /// Whether pending thread effects must rendezvous before upcoming demands.
  static bool requiresThreadSync(const BlockInfo &pending,
                                 const BlockInfo &effects);

  /// Clears the effects because a barrier is inserted.
  void sync() {
    syncReadSlices.clear();
    syncWriteSlices.clear();
    threadSync = {};
    threadDemands = {};
  }

  bool operator==(const BlockInfo &) const = default;

  /// Checks one pair of access maps with the same alias and operation filters.
  static bool isIntersected(const SliceMapT &lhsSlices,
                            const SliceMapT &rhsSlices, bool lhsIsRead,
                            bool rhsIsRead, const MembarFilterFn &filter,
                            const MembarSliceFilterFn &sliceFilter,
                            Allocation *allocation) {
    for (auto &lhs : lhsSlices)
      for (auto &rhs : rhsSlices)
        if (lhs.first.intersects(rhs.first))
          if (!sliceFilter || !sliceFilter(lhs.first, rhs.first, lhsIsRead,
                                           rhsIsRead, allocation))
            for (auto lhsOp : lhs.second)
              for (auto rhsOp : rhs.second)
                if (!filter || !filter(lhsOp, rhsOp, lhsIsRead, rhsIsRead,
                                       allocation, lhs.first, rhs.first))
                  return true;
    return false;
  }

private:
  void joinThreadState(const BlockInfo &other);
};

namespace triton::membar {

struct FixedSync;
struct OpModel;

struct SharedAccess {
  const BufferRegionFootprint *footprint;
  bool isWrite;
  bool crossCTA = false;

  bool operator==(const SharedAccess &) const = default;
};

/// Physical shared accesses in kernel coordinates for whole-function placement.
class SharedAccessModel {
public:
  explicit SharedAccessModel(BufferRegionAnalysis &regions)
      : regions(regions) {}
  const BufferIndexAnalysis &getIndices(FunctionOpInterface function);

protected:
  SharedAccess getSharedAccess(Operation *op, const MemoryAccess &access);
  const BufferRegionFootprint *getFootprint(Value value);
  const BufferRegionFootprint *getScratchFootprint(Operation *op);

private:
  BufferRegionAnalysis &regions;
  std::map<Operation *, BufferIndexAnalysis> indices;
};

bool hasRegionLocalUses(Value root, bool followCalls);

/// Describes shared-memory accesses and synchronization for one operation.
class SMemModel {
public:
  enum class AccessMode {
    AllSharedAccesses,
    AllocatorAliasesOnly,
    SynchronizationOnly
  };

  /// Only legacy MembarAnalysis uses followCalls=false.
  /// TODO: When it retires, remove the flag, follow descriptor uses through
  /// calls unconditionally, and keep callee footprints in kernel coordinates.
  SMemModel(Allocation &allocation, BufferRegionAnalysis &regions,
            AccessMode accessMode = AccessMode::AllSharedAccesses,
            CompletionScopeFn completionScope = nullptr,
            bool followCalls = false)
      : allocation(allocation), regions(regions), accessMode(accessMode),
        completionScope(completionScope), followCalls(followCalls),
        bufferIndexAnalysis(
            cast<FunctionOpInterface>(allocation.getOperation())) {}

  /// SynchronizationOnly records fixed barriers, including scratch rendezvous.
  /// Cluster placement supplies its own stages and tracks only memory slices.
  /// TODO: Remove clusterStages, AllocatorAliasesOnly and MembarSliceFilterFn
  /// when the legacy cluster analysis migrates.
  void describe(Operation *op, OpModel &model, FixedSync &fixed,
                ArrayRef<MemoryAccess> accesses,
                std::optional<BarrierStages> clusterStages = std::nullopt);
  /// TODO: Retire this method and AllocationSlice::translateToCallsite with the
  /// legacy forward-flow summaries.
  MembarInfo translateCallSummary(CallOpInterface call,
                                  FunctionOpInterface callee,
                                  MembarInfo summary);
  /// Shared-memory lifetime markers do not issue memory accesses.
  static bool hasThreadEffects(Operation *operation);
  const BufferIndexAnalysis &getBufferIndexAnalysis() const {
    return bufferIndexAnalysis;
  }

private:
  SmallVector<AllocationSlice>
  getAllocationSlices(Value value, std::optional<gpu::SharedKind> sharedKind);
  bool isRegionLocal(Value value);
  bool mayNotifyPeer(Operation *op, ArrayRef<MemoryAccess> accesses);
  BlockInfo getThreadEffects(Operation *op);
  void addThreadDemand(BlockInfo &effects, Operation *op,
                       ArrayRef<MemoryAccess> accesses);

  Allocation &allocation;
  BufferRegionAnalysis &regions;
  AccessMode accessMode;
  CompletionScopeFn completionScope;
  bool followCalls;
  BufferIndexAnalysis bufferIndexAnalysis;
  DenseMap<Value, bool> regionLocalAllocations;
};

} // namespace triton::membar

} // namespace mlir

#endif // TRITON_ANALYSIS_MEMBAR_SMEMMODEL_H
