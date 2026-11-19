#ifndef TRITON_ANALYSIS_MEMBAR_CFGBUILDER_H
#define TRITON_ANALYSIS_MEMBAR_CFGBUILDER_H

#include "triton/Analysis/Membar/SmemModel.h"
#include "triton/Analysis/Membar/TmemModel.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include <map>
#include <optional>

namespace mlir::triton::membar {

enum class Action { WaitLoad, WaitStore, RegionBarrier, WarpBarrier };

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
  WaitAll,      // Complete issued TMEM accesses.
  Publication,  // Order peer-visible TMEM effects.
  KernelReturn, // Complete issued TMEM accesses and order CompletionSync::All.
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
  // Zero wherever the execution group cannot rendezvous.
  unsigned insertionCost = 0;
  unsigned body = 0;
  OpModel effects;
  SmallVector<Edge, 2> next;
};

struct MembarCFG {
  SmallVector<Node, 0> nodes;

  // A callee entry or loop backedge can rebind SSA values in its own body.
  bool invalidatesIteration(Edge edge, unsigned source) const {
    return edge.invalidatesIteration &&
           nodes[edge.to].body == nodes[source].body;
  }
};

unsigned getBlockWeight(Block *block, const BufferIndexAnalysis &indices,
                        unsigned invocationWeight);

class MembarCFGBuilder {
public:
  MembarCFGBuilder(std::map<Operation *, SMemModel> &models, TMemModel *tmem,
                   const llvm::SetVector<Operation *> &callbacks,
                   const DenseMap<Operation *, unsigned> &calleeWeights)
      : models(models), tmem(tmem), callbacks(callbacks),
        calleeWeights(calleeWeights) {}

  /// Build one execution region and link its reachable noinline callees.
  MembarCFG build(Region &root);

private:
  void addRegion(MembarCFG &cfg, Region &region);
  OpModel describe(Operation *op, SMemModel &shared);

  std::map<Operation *, SMemModel> &models;
  TMemModel *tmem;
  const llvm::SetVector<Operation *> &callbacks;
  const DenseMap<Operation *, unsigned> &calleeWeights;
};

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_CFGBUILDER_H
