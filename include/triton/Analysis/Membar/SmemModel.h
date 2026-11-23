#ifndef TRITON_ANALYSIS_MEMBAR_SMEMMODEL_H
#define TRITON_ANALYSIS_MEMBAR_SMEMMODEL_H

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferIndexAnalysis.h"
#include "triton/Analysis/BufferRegion.h"
#include <compare>
#include <cstdint>
#include <map>

namespace mlir {

namespace triton::membar {

// A source can require a rendezvous independently of storage overlap.
// Summaries union the effects on each side of an operation.
struct ThreadEffects {
  enum : uint8_t {
    // Each participant in a release publishes its own earlier work.
    Thread0 = 1 << 0,
    WarpLeaders = Thread0 | (1 << 1), // Lane zero of every warp.
    Warp0Leader = 1 << 2,             // Elected lane of the first warp.
    AnyIssuer = WarpLeaders | Warp0Leader,
    SharedCompletion = 1 << 3, // May overlap independent global reads.
    FullCompletion = 1 << 4,   // Must also finish before reads and kernel exit.
    Completions = SharedCompletion | FullCompletion,
    PeerWait = 1 << 5, // A peer notification can advance the counter phase.
  };

  // Effects left pending when the operation exits.
  uint8_t pending = 0;
  // Pending effects that require a rendezvous before this operation.
  uint8_t joins = 0;

  void merge(ThreadEffects other) {
    pending |= other.pending;
    joins |= other.joins;
  }
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

// A backend may order a specific effect even when its storage overlaps.
enum class OrderingRule : uint8_t {
  Ordinary,
  // NVIDIA transaction completion, mbarrier protocol and atomic updates.
  AsyncPayload,
  Counter,
  CounterInit,
  CounterWait,
  CounterSignal,
  CommutativeAtomic,
  // AMD async tokens and mbarriers. An ordered copy uses both protocols.
  AsyncCopy,
  OrderedCopy,
  OrderedLoad,
  OrderedBarrier
};

struct SMemAccess;

/// Classify effects before merging guarded operations or call summaries.
/// Keys belong to this ordering instance and outlive all its access records.
class SMemOrdering {
public:
  virtual ~SMemOrdering() = default;
  // Only descriptor effects have backend-specific ordering. Scratch is
  // ordinary.
  virtual void classify(SMemAccess &access, Operation *op,
                        const MemoryAccess &effect, ThreadEffects threads,
                        Allocation &allocation) = 0;
  // Exempt overlapping accesses only if the ordering survives rebinding.
  virtual bool canSkip(const SMemAccess &source, const SMemAccess &sink,
                       Allocation &allocation, bool inKernel) const = 0;
};

struct SMemAccess {
  // A write subsumes a read of the same bytes for hazard detection.
  enum Kind : uint8_t { None, Read, Write };
  // Exposed to predecessors and successors respectively. An internal
  // rendezvous separates the first scratch write from the final read.
  Kind entry, exit;
  // A single operation can have different ordering rules for its payload
  // and counter. The key identifies extra comparison facts, or zero if unused.
  OrderingRule rule = OrderingRule::Ordinary;
  uint32_t key = 0;
  // All possible physical origins, including allocation padding and CTA IDs.
  // A null footprint imposes no restriction on overlap.
  const BufferRegionFootprint *footprint = nullptr;
  // Relative slot and subslice coordinates are valid in one SSA binding.
  const BufferIndexExpr *index = nullptr;
  const Subslice *subslice = nullptr;

  // The caller supplies physical storage in its own coordinate system.
  static SMemAccess fromDescriptor(const MemoryAccess &effect,
                                   BufferIndexAnalysis &indices,
                                   bool useIndex = true);
  Conflict conflict(const SMemAccess &sink, const SMemOrdering *ordering,
                    Allocation &allocation, bool inKernel = false) const;
  auto operator<=>(const SMemAccess &) const = default;
};

struct SMemData {
  // Shared bytes the operation reads or writes.
  SmallVector<SMemAccess, 0> accesses;
  // What the issuing threads leave pending and what earlier work they need
  // to observe.
  ThreadEffects threads;

  void merge(const SMemData &other);
  Conflict conflict(const SMemData &sink, const SMemOrdering *ordering,
                    Allocation &allocation, bool inKernel = false) const;
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

  SMemModel(Allocation &allocation, BufferRegionAnalysis &regions,
            SMemOrdering *ordering, Mode mode,
            CompletionScopeFn completionScope = nullptr)
      : allocation(allocation), regions(regions), ordering(ordering),
        mode(mode), completionScope(completionScope),
        bufferIndexAnalysis(
            cast<FunctionOpInterface>(allocation.getOperation())) {}

  /// Describe the ordering requirements and fixed barriers of one operation.
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
  // Backend classification and key tables shared across this module's calls.
  SMemOrdering *ordering;
  // A role shares thread identities and kernel coordinates across calls.
  // Function summaries translate per call. TMEM-only placement needs fixed
  // sync.
  Mode mode;
  // AMD refines the completion scope after assigning wait counts.
  CompletionScopeFn completionScope;
  // Owns the slot expressions borrowed by the nodes of this function.
  BufferIndexAnalysis bufferIndexAnalysis;
  // A counter confined to this region cannot advance a peer's phase.
  DenseMap<Value, bool> regionLocalAllocations;
};

} // namespace triton::membar

} // namespace mlir

#endif // TRITON_ANALYSIS_MEMBAR_SMEMMODEL_H
