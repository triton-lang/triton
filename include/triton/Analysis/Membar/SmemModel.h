#ifndef TRITON_ANALYSIS_MEMBAR_SMEMMODEL_H
#define TRITON_ANALYSIS_MEMBAR_SMEMMODEL_H

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferIndexAnalysis.h"
#include "triton/Analysis/BufferRegion.h"
#include <cstdint>
#include <functional>
#include <map>
#include <optional>

namespace mlir {

namespace triton::membar {
struct SMemAccess;
}

/// Exempt overlapping accesses that commute or are already ordered. The
/// exemption must hold even across a backedge or call entry.
using MembarFilterFn =
    std::function<bool(const triton::membar::SMemAccess &,
                       const triton::membar::SMemAccess &, Allocation &)>;

/// Physical storage and the relative coordinates valid in one SSA binding.
struct AllocationSlice {
  // The value supplies the subslice coordinates. Use
  // BufferIndexAnalysis::makeSlice to attach the dynamic slot as well.
  AllocationSlice(Value value, Interval<size_t> interval,
                  Allocation::BufferId bufferId);
  // Implicit scratch has no descriptor. Use the whole allocation interval.
  explicit AllocationSlice(Interval<size_t> interval)
      : allocationInterval(interval) {}

  bool operator<(const AllocationSlice &other) const;
  bool operator==(const AllocationSlice &) const = default;

  bool mayOverlapPhysically(const AllocationSlice &other) const;
  bool disjointWithinBinding(const AllocationSlice &other) const;

  Allocation::BufferId getBufferId() const { return bufferId; }
  bool hasIterationInfo() const { return bufferIndexExpr || subslice; }
  // Relative coordinates are part of the sort key. Rebuild any ordered
  // container when clearing them across a backedge or a call boundary.
  void invalidateIterationInfo() {
    bufferIndexExpr = nullptr;
    subslice = nullptr;
  }

  /// Translate shared-memory geometry to the caller's frame and discard
  /// callee-local index and subslice facts.
  /// TODO: Retire this method with the AMD forward-flow summaries.
  AllocationSlice
  translateToCallsite(CallOpInterface call, FunctionOpInterface callee,
                      triton::BufferRegionAnalysis &regions) const;

  // The footprint covers every possible origin across loop iterations,
  // retaining the memory space, CTA, partition, subview and padding.
  // BufferRegionAnalysis outlives the slices that borrow it. A null
  // footprint leaves only the allocator interval for physical overlap.
  const triton::BufferRegionFootprint *physicalFootprint = nullptr;
  // BufferIndexAnalysis owns the expression. Include it in slice ordering
  // and equality to keep accesses to different slots separate.
  const BufferIndexExpr *bufferIndexExpr = nullptr;

private:
  // Bounds for this allocator buffer, also available for implicit scratch.
  Interval<size_t> allocationInterval;
  // Identifies a local buffer for slot comparisons and backend alias checks.
  Allocation::BufferId bufferId = Allocation::InvalidBufferId;
  // A direct subslice gives its source, offsets and shape for relative proofs.
  Operation *subslice = nullptr;
};

namespace triton::membar {

// A source can require a rendezvous independently of storage overlap.
// Summaries union the effects on each side of an operation.
struct ThreadEffects {
  enum : uint8_t {
    // A release with a known issuer observes that issuer in program order.
    // Thread and warp numbers are relative to the executing partition.
    // The elected lane of warp zero can differ from thread zero.
    Thread0 = 1 << 0,
    WarpLeaders = 1 << 1, // Lane zero of every warp.
    Warp0Leader = 1 << 2, // Elected lane of the first warp.
    // Use all issuer bits when the issuing lanes are unknown.
    AnyIssuer = Thread0 | WarpLeaders | Warp0Leader,
    SharedCompletion = 1 << 3, // May overlap independent global reads.
    FullCompletion = 1 << 4,   // Must also finish before reads and kernel exit.
    Completions = SharedCompletion | FullCompletion,
    PeerWait = 1 << 5, // A peer notification can advance the counter phase.
  };

  // Effects left pending when the operation exits.
  uint8_t pending = 0;
  // Earlier effects that require a rendezvous before this operation.
  // Only a publication adds issuer bits to this mask.
  // FixedSync records the barriers the operation itself provides.
  uint8_t joins = 0;

  void merge(ThreadEffects other, bool usesShared);
  bool operator==(const ThreadEffects &) const = default;
};

} // namespace triton::membar

// A wait leaves these completion effects pending in its issuing threads.
enum class CompletionSync : uint8_t {
  None = 0,
  Shared = triton::membar::ThreadEffects::SharedCompletion,
  All = triton::membar::ThreadEffects::Completions,
};

/// The backend can refine the completion scope of an individual wait.
using CompletionScopeFn = CompletionSync (*)(Operation *, CompletionSync);

namespace triton::membar {

struct FixedSync;

enum class Conflict { None, AfterRebind, Always };

struct SMemAccess {
  // A write subsumes a read of the same bytes for hazard detection.
  enum Kind : uint8_t { None, Read, Write };
  // Only the backend barrier filter should inspect this operation.
  Operation *issuer;
  AllocationSlice slice;
  // Exposed to predecessors and successors respectively. An internal
  // rendezvous separates the first scratch write from the final read.
  Kind entry, exit;
  // A single operation can signal a counter and also access its payload.
  // Preserve that distinction when translating or widening the slice.
  bool isMBarrier = false;

  Conflict conflict(const SMemAccess &sink, const MembarFilterFn &filter,
                    Allocation &allocation) const;
  bool operator==(const SMemAccess &) const = default;
};

struct SMemData {
  // Shared bytes the operation reads or writes, including implicit scratch.
  // A guard or a call summary can include accesses from several operations.
  SmallVector<SMemAccess, 0> accesses;
  // What the issuing threads leave pending and what earlier work they need
  // to observe.
  ThreadEffects threads;

  void merge(const SMemData &other);
  Conflict conflict(const SMemData &sink, const MembarFilterFn &filter,
                    Allocation &allocation) const;
  bool operator==(const SMemData &) const = default;
};

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
  enum class AccessMode { AllSharedAccesses, SynchronizationOnly };

  /// Only AMD::MembarAnalysis uses followCalls=false.
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
  SMemData describe(Operation *op, FixedSync &fixed,
                    ArrayRef<MemoryAccess> accesses);
  const BufferIndexAnalysis &getBufferIndexAnalysis() const {
    return bufferIndexAnalysis;
  }

private:
  SmallVector<AllocationSlice> getAllocationSlices(Value value);
  bool isRegionLocal(Value value);
  bool mayNotifyPeer(Operation *op, ArrayRef<MemoryAccess> accesses);
  ThreadEffects getThreadEffects(Operation *op, bool hasScratch,
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
