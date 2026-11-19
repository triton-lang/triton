#ifndef TRITON_ANALYSIS_MEMBAR_CFGBUILDER_H
#define TRITON_ANALYSIS_MEMBAR_CFGBUILDER_H

#include "mlir/Analysis/Presburger/Fraction.h"
#include "triton/Analysis/Membar/SmemModel.h"
#include "triton/Analysis/Membar/TmemModel.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include <map>
#include <optional>
#include <utility>

namespace mlir::triton::membar {

// A demand names the minimum synchronization it requires. An action can
// satisfy several kinds of demand at its insertion site.
enum class SyncKind {
  WaitLoad,
  WaitStore,
  RegionBarrier,
  WarpBarrier,
  WarpConvergence, // Demand only.
  CTAProxyFence,
  ClusterProxyFence,
  ClusterBarrier,
  ClusterFenceMbarInit
};

constexpr unsigned bit(SyncKind kind) { return 1u << unsigned(kind); }

constexpr unsigned coverage(SyncKind action) {
  switch (action) {
  case SyncKind::WaitLoad:
  case SyncKind::WaitStore:
  case SyncKind::WarpBarrier:
    return bit(action) | bit(SyncKind::WarpConvergence);
  case SyncKind::RegionBarrier:
    return bit(action) | coverage(SyncKind::WarpBarrier);
  case SyncKind::ClusterProxyFence:
    return bit(action) | bit(SyncKind::CTAProxyFence);
  case SyncKind::ClusterBarrier:
    return bit(action) | bit(SyncKind::ClusterFenceMbarInit);
  default:
    return bit(action);
  }
}

constexpr unsigned warpBarrier = coverage(SyncKind::WarpBarrier);
constexpr unsigned fullBarrier = coverage(SyncKind::RegionBarrier);

enum class Boundary {
  None,
  WaitAll,      // Complete issued TMEM accesses.
  Publication,  // Order peer-visible TMEM effects.
  KernelReturn, // Complete issued TMEM accesses and order CompletionSync::All.
};

// Observe the input before the operation and start new producers at its output.
// A scratch rendezvous lies between those two sets of effects.
struct OpModel {
  BlockInfo sharedIn, sharedOut;
  SmallVector<TMemAccess, 1> tmemAccesses;
  // Native transfers need a completion wait; descriptor asm needs region order.
  std::optional<SyncKind> tmemProducerNeed;
  Boundary boundary = Boundary::None;
  bool privateTMem = false;
  bool needsWarp = false;

  bool operator==(const OpModel &) const = default;
};

enum class WarpEdge { None, EnterWorker, ExitWorker };

struct Edge {
  unsigned to;
  bool invalidatesIteration = false;
  WarpEdge warp = WarpEdge::None;

  // A worker exports its own effects. An effect inherited at launch follows
  // the default warps to the continuation, including on later loop iterations.
  bool follow(bool &mayExitWorker) const {
    if (warp != WarpEdge::ExitWorker)
      return true;
    return std::exchange(mayExitWorker, false);
  }
};

struct FixedSync {
  unsigned before = 0, inside = 0, after = 0;

  static FixedSync fromStages(BarrierStages stages, unsigned mask) {
    return {stages.beforeMemoryEffects ? mask : 0,
            stages.betweenMemoryEffects ? mask : 0,
            stages.afterMemoryEffects ? mask : 0};
  }
  bool operator==(const FixedSync &) const = default;
};

// Arrival observes incoming effects; departure starts outgoing dependencies.
// A fixed internal rendezvous orders paths passing through the operation.
template <typename Facts> struct Node {
  Operation *op = nullptr;
  // Zero wherever the execution group cannot rendezvous.
  unsigned insertionCost = 0;
  unsigned loopDepth = 0;
  // First point in this function or warp role. Zero identifies the root body.
  unsigned body = 0;
  FixedSync fixed;
  Facts effects;
  SmallVector<Edge, 2> next;
  // First CFG point for this operation. Copies in different roles share a gap.
  unsigned gap = 0;
  bool inWorker = false;
};

template <typename Facts> struct MembarCFG {
  SmallVector<Node<Facts>, 0> nodes;

  // A callee entry or loop backedge can rebind SSA values in its own body.
  bool invalidatesIteration(Edge edge, unsigned source) const {
    return edge.invalidatesIteration &&
           nodes[edge.to].body == nodes[source].body;
  }
};

class MembarModel {
public:
  using Facts = OpModel;
  MembarModel(std::map<Operation *, SMemModel> &models, TMemModel *tmem)
      : models(models), tmem(tmem) {}

  const BufferIndexAnalysis &getIndices(FunctionOpInterface function);
  std::optional<Facts> describe(Operation *op, FixedSync &fixed,
                                Block *active = nullptr);

private:
  std::map<Operation *, SMemModel> &models;
  TMemModel *tmem;
};

template <typename Model> class MembarCFGBuilder {
  using Facts = typename Model::Facts;

public:
  MembarCFGBuilder(Model &model, ModuleOp module);

  /// Link reachable noinline callees and optionally expand warp-specialized
  /// roles.
  MembarCFG<Facts> build(Region &root, bool expandWarpRoles);

private:
  struct BlockWeight {
    presburger::Fraction cost;
    unsigned loopDepth;
  };
  BlockWeight getBlockWeight(Block *block);
  void normalizeCosts();
  void appendWithCallees(MembarCFG<Facts> &cfg, Region &root,
                         std::optional<Edge> roleExit = std::nullopt);
  void addRegion(MembarCFG<Facts> &cfg, Region &region);

  Model &model;
  llvm::SetVector<Operation *> callbacks;
  DenseMap<Operation *, SmallVector<Block *, 1>> callers;
  DenseMap<Block *, BlockWeight> blockWeights;
  // Legal blocks in the graph currently being built.
  DenseMap<Block *, unsigned> insertionCosts;
};

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_CFGBUILDER_H
