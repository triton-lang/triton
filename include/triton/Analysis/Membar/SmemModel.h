#ifndef TRITON_ANALYSIS_MEMBAR_SMEMMODEL_H
#define TRITON_ANALYSIS_MEMBAR_SMEMMODEL_H

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferIndexAnalysis.h"
#include "triton/Analysis/BufferRegion.h"
#include <compare>
#include <cstdint>
#include <functional>
#include <map>

namespace mlir {

namespace triton::membar {

// A source can require a rendezvous independently of storage overlap.
// Summaries union the effects on each side of an operation.
struct ThreadEffects {
  enum : uint8_t {
    // Each participant in a release publishes its own earlier work.
    // Thread and warp numbers are relative to the executing partition.
    // The elected lane of warp zero can differ from thread zero.
    Thread0 = 1 << 0,
    WarpLeaders = Thread0 | (1 << 1), // Lane zero of every warp.
    Warp0Leader = 1 << 2,             // Elected lane of the first warp.
    // Use all issuer bits when the issuing lanes are unknown.
    AnyIssuer = WarpLeaders | Warp0Leader,
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

  void merge(ThreadEffects other) {
    pending |= other.pending;
    joins |= other.joins;
  }
  bool operator==(const ThreadEffects &) const = default;
};

struct SMemAccess;

} // namespace triton::membar

/// Exempt overlapping accesses that commute or are already ordered. The
/// exemption must hold even across a backedge or call entry.
using MembarFilterFn =
    std::function<bool(const triton::membar::SMemAccess &,
                       const triton::membar::SMemAccess &, Allocation &)>;

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
  // Exposed to predecessors and successors respectively. An internal
  // rendezvous separates the first scratch write from the final read.
  Kind entry, exit;
  // A transfer can signal a counter and access a separate payload.
  bool isMBarrier = false;
  // Only the backend barrier filter should inspect this operation.
  Operation *issuer = nullptr;
  // The footprint covers every possible origin across loop iterations,
  // retaining the memory space, CTA, partition, subview and padding.
  // BufferRegionAnalysis outlives the accesses that borrow it. A null
  // footprint imposes no restriction on physical overlap.
  const BufferRegionFootprint *footprint = nullptr;
  // Relative slot and subslice coordinates are valid in one SSA binding.
  RelativeAddress relative;

  // The caller supplies physical storage in its own coordinate system.
  static SMemAccess fromDescriptor(Operation *op, const MemoryAccess &effect,
                                   BufferIndexAnalysis &indices,
                                   bool useIndex = true);
  Conflict conflict(const SMemAccess &sink, const MembarFilterFn &filter,
                    Allocation &allocation) const;
  auto operator<=>(const SMemAccess &) const = default;
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

// Proxy and cluster placement need the physical footprint and CTA routing.
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

bool hasRegionLocalUses(Value root);

/// Describes shared-memory accesses and synchronization for one operation.
class SMemModel {
public:
  enum class Mode { Role, FunctionSummary, SynchronizationOnly };

  SMemModel(Allocation &allocation, BufferRegionAnalysis &regions, Mode mode,
            CompletionScopeFn completionScope = nullptr)
      : allocation(allocation), regions(regions), mode(mode),
        completionScope(completionScope),
        bufferIndexAnalysis(
            cast<FunctionOpInterface>(allocation.getOperation())) {}

  /// Describe the ordering requirements and fixed barriers of one operation.
  /// SynchronizationOnly records fixed barriers, including scratch rendezvous.
  SMemData describe(Operation *op, FixedSync &fixed,
                    ArrayRef<MemoryAccess> accesses);
  const BufferIndexAnalysis &getBufferIndexAnalysis() const {
    return bufferIndexAnalysis;
  }

private:
  bool isRegionLocal(Value value);
  bool mayNotifyPeer(Operation *op, ArrayRef<MemoryAccess> accesses);
  ThreadEffects getThreadEffects(Operation *op, bool hasScratch,
                                 ArrayRef<MemoryAccess> accesses);

  // Scratch belongs to the function. Descriptor footprints are module-wide.
  Allocation &allocation;
  BufferRegionAnalysis &regions;
  // A role shares thread identities and kernel coordinates across calls.
  // AMD function summaries translate storage at each call site.
  Mode mode;
  // AMD refines a wait's scope when assigning hardware wait counts.
  CompletionScopeFn completionScope;
  // Owns the slot expressions and subslice records borrowed by this
  // function's nodes.
  BufferIndexAnalysis bufferIndexAnalysis;
  // A counter confined to this region cannot advance a peer's phase.
  DenseMap<Value, bool> regionLocalAllocations;
};

} // namespace triton::membar

} // namespace mlir

#endif // TRITON_ANALYSIS_MEMBAR_SMEMMODEL_H
