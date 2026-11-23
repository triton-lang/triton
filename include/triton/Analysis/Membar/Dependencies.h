#ifndef TRITON_ANALYSIS_MEMBAR_DEPENDENCIES_H
#define TRITON_ANALYSIS_MEMBAR_DEPENDENCIES_H

#include "triton/Analysis/Membar/CFGBuilder.h"

namespace mlir::triton::membar {

enum class Path {
  Any,
  LoopCarried,
  WaitThenOrder,    // Complete a TMEM access before a region barrier.
  ClusterThenProxy, // A cluster acquire lets the owner use a CTA proxy fence.
};

struct Demand {
  unsigned source, sink;
  SyncKind first;
  Path path = Path::Any;
};

struct MembarDemands {
  SmallVector<Demand> required;
  // An uncovered scalar publication requires a per-warp notification.
  SmallVector<Demand> optional;
};

// A model may leave its state unchanged at a fixed sync or on an edge.
template <typename State> struct KeepState {
  std::optional<State> operator()(State state, unsigned) const { return state; }
  State operator()(Edge, State state) const { return state; }
};

// Start after the producer and visit each (point, worker context, state) once
// before internal sync. The visitor and fixed sync transfer return std::nullopt
// to retire the producer on this path. State must support DenseMapInfo.
template <typename Facts, typename State, typename Visit,
          typename Apply = KeepState<State>, typename OnEdge = KeepState<State>>
void walkEdges(const MembarCFG<Facts> &cfg, const Node<Facts> &producer,
               State initial, Visit visit, Apply apply = {},
               OnEdge onEdge = {}) {
  llvm::SmallSetVector<std::pair<unsigned, State>, 16> arrivals;
  auto depart = [&](const Node<Facts> &node, State state, bool worker) {
    auto live = apply(state, node.fixed.after);
    if (!live)
      return;
    for (Edge edge : node.next) {
      bool nextWorker = worker;
      if (!edge.follow(nextWorker))
        continue;
      if (auto next = apply(*live, cfg.nodes[edge.to].fixed.before))
        arrivals.insert({2 * edge.to + nextWorker, onEdge(edge, *next)});
    }
  };
  depart(producer, initial, producer.inWorker);
  for (unsigned i = 0; i != arrivals.size(); ++i) {
    auto [arrival, incoming] = arrivals[i];
    unsigned point = arrival / 2;
    const auto &node = cfg.nodes[point];
    bool canExitWorker = arrival & 1;
    if (auto next = visit(point, incoming, canExitWorker))
      if (auto live = apply(*next, node.fixed.inside))
        depart(node, *live, canExitWorker);
  }
}

void coalesceDemands(SmallVectorImpl<Demand> &demands);

// Expose completed shared accesses at the waits that make them available.
void recordSharedCompletions(MembarCFG<OpModel> &cfg,
                             const SMemOrdering &ordering);

MembarDemands discoverDemands(const MembarCFG<OpModel> &cfg,
                              Allocation &allocation,
                              const SMemOrdering *ordering, TMemModel *tmem);

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_DEPENDENCIES_H
