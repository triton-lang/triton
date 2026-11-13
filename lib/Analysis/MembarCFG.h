// Internal program model for synchronization placement.
#ifndef TRITON_LIB_ANALYSIS_MEMBARCFG_H
#define TRITON_LIB_ANALYSIS_MEMBARCFG_H

#include "triton/Analysis/MembarPlacement.h"
#include "llvm/ADT/BitVector.h"
#include <array>
#include <cstdint>

namespace mlir::triton::membar {

enum class Action { WaitLoad, WaitStore, RegionBarrier, WarpBarrier };

// Site IDs follow emission order so waits precede barriers at the same gap.
inline constexpr std::array ACTIONS{Action::WaitLoad, Action::WaitStore,
                                    Action::RegionBarrier, Action::WarpBarrier};

constexpr unsigned bit(Action action) { return 1u << unsigned(action); }
enum class Need {
  WaitLoad = unsigned(Action::WaitLoad),
  WaitStore = unsigned(Action::WaitStore),
  RegionOrder = unsigned(Action::RegionBarrier),
  WarpOrder = unsigned(Action::WarpBarrier),
  WarpConvergence
};

constexpr unsigned bit(Need need) { return 1u << unsigned(need); }
constexpr unsigned warpBarrier =
    bit(Need::WarpOrder) | bit(Need::WarpConvergence);
constexpr unsigned fullBarrier = bit(Need::RegionOrder) | warpBarrier;

constexpr unsigned coverage(Action action) {
  if (action == Action::RegionBarrier)
    return fullBarrier;
  if (action == Action::WarpBarrier)
    return warpBarrier;
  return bit(action) | bit(Need::WarpConvergence);
}

enum class Boundary {
  None,
  WaitAll,     // Complete issued TMEM accesses.
  Publication, // Order peer-visible TMEM effects.
  Clean,       // Order every outstanding effect at a collective call or return.
  KernelReturn, // Complete issued TMEM accesses and order CompletionSync::All.
};

struct TMemAccess {
  const BufferRegionFootprint *footprint;
  bool isWrite;

  bool operator==(const TMemAccess &) const = default;
};

// Observe the input before the operation and start new producers at its output.
// A scratch rendezvous lies between those two sets of effects.
struct OpModel {
  unsigned fixedBefore = 0, fixedInside = 0, fixedAfter = 0;
  BlockInfo sharedIn, sharedOut;
  SmallVector<TMemAccess, 1> tmemAccesses;
  // Native transfers need a completion wait; descriptor asm needs region order.
  std::optional<Need> tmemProducerNeed;
  Boundary boundary = Boundary::None;
  bool privateTMem = false;
  bool needsWarp = false;

  bool operator==(const OpModel &) const = default;
};

struct Edge {
  unsigned to;
  bool invalidatesIteration = false;
};

// Arrival observes incoming effects; departure starts outgoing dependencies.
// A fixed internal rendezvous orders paths passing through the operation.
struct Node {
  Operation *op = nullptr;
  // Zero at entry and wherever the execution group cannot rendezvous.
  unsigned insertionCost = 0;
  OpModel effects;
  SmallVector<Edge, 2> next;
};

struct MembarCFG {
  static constexpr unsigned entry = 0;
  SmallVector<Node, 0> nodes;
};

// Loop-carried shared hazards and ordered TMEM handoffs each need one extra
// state. The latter always ends with a region barrier.
enum class Path { Any, LoopCarried, WaitThenOrder };

struct Demand {
  unsigned source, sink;
  Need first;
  Path path = Path::Any;
};

class TMemModel;

SmallVector<Demand> discoverDemands(const MembarCFG &cfg,
                                    Allocation &allocation,
                                    const MembarFilterFn &sharedFilter,
                                    TMemModel *tmem);

struct CutSolution {
  llvm::BitVector sites;
  uint64_t cost = 0;
  // A matching placement cost certifies optimality.
  uint64_t lowerBound = 0;
};

std::optional<CutSolution> solveMembarCut(const MembarCFG &cfg,
                                          ArrayRef<Demand> demands);

bool hasRegionLocalUses(Value root);

// These target rules describe TMEM instruction effects and word ownership.
class TMemModel {
public:
  TMemModel(ModuleOp module, BufferRegionAnalysis &regions);
  ~TMemModel();
  void describe(Operation *op, OpModel &model, ArrayRef<MemoryAccess> accesses);
  // Transfers in one placement graph execute in the same warp role.
  bool isSameWarp(Operation *lhs, Operation *rhs);

private:
  struct Info;
  const Info *getInfo(Operation *op);
  bool isPrivate(const BufferRegionFootprint *footprint, Region *issuer) const;

  BufferRegionAnalysis &regions;
  SmallVector<std::pair<const BufferRegionFootprint *, Region *>> allocations;
  DenseMap<Operation *, std::unique_ptr<Info>> info;
};

struct PlacementState {
  SMemModel shared;
  TMemModel *tmem;
  Boundary returnBoundary;

  MembarCFG buildMembarCFG(Region &region);

private:
  OpModel describe(Operation *op);
};

} // namespace mlir::triton::membar
#endif
