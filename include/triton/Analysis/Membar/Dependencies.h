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
  Need first;
  Path path = Path::Any;
};

using EdgeTransfer = llvm::function_ref<unsigned(Edge, unsigned)>;
using SyncTransfer = llvm::function_ref<unsigned(unsigned, unsigned)>;

// Visit each (point, state) once, observing the consumer before internal sync.
// A sync transfer returns zero when it covers the producer. Completion walks
// omit the transfer so zero can represent an uncommitted copy.
template <typename Facts, typename Visit>
void walkEdges(const MembarCFG<Facts> &cfg, const Node<Facts> &producer,
               unsigned state, Visit visit, SyncTransfer apply = {},
               EdgeTransfer onEdge = {}) {
  llvm::SmallSetVector<std::pair<unsigned, unsigned>, 16> arrivals;
  auto enqueue = [&](ArrayRef<Edge> edges, unsigned state, bool worker) {
    for (Edge edge : edges) {
      bool nextWorker = worker;
      if (!edge.follow(nextWorker))
        continue;
      unsigned live =
          apply ? apply(state, cfg.nodes[edge.to].fixed.before) : state;
      if (!apply || live)
        arrivals.insert(
            {2 * edge.to + nextWorker, onEdge ? onEdge(edge, live) : live});
    }
  };
  enqueue(producer.next, apply ? apply(state, producer.fixed.after) : state,
          producer.inWorker);
  for (unsigned i = 0; i != arrivals.size(); ++i) {
    auto [arrival, incoming] = arrivals[i];
    unsigned point = arrival / 2;
    const auto &node = cfg.nodes[point];
    bool canExitWorker = arrival & 1;
    if (auto next = visit(point, incoming, canExitWorker))
      enqueue(node.next,
              apply ? apply(apply(*next, node.fixed.inside), node.fixed.after)
                    : *next,
              canExitWorker);
  }
}

// A model returns the ordering required at a reachable consumer. The first
// SMEM conflict also orders subsequent consumers on that path.
struct Conflict {
  Need need{};
  std::optional<Path> path;
  bool stopOnMatch = false;
};

// Start after the producer. Observe each consumer before its internal sync.
template <typename Facts, typename Rule>
void discoverDemands(const MembarCFG<Facts> &cfg,
                     SmallVectorImpl<Demand> &demands, unsigned source,
                     unsigned state, SyncTransfer apply, Rule rule,
                     EdgeTransfer onEdge = {}) {
  walkEdges(
      cfg, cfg.nodes[source], state,
      [&](unsigned sink, unsigned live, bool) {
        auto conflict = rule(cfg.nodes[sink], live);
        if (conflict.path) {
          demands.push_back({source, sink, conflict.need, *conflict.path});
          if (conflict.stopOnMatch)
            return std::optional<unsigned>{};
        }
        return std::optional(live);
      },
      apply, onEdge);
}

void coalesceDemands(SmallVectorImpl<Demand> &demands);

SmallVector<Demand> discoverDemands(const MembarCFG<OpModel> &cfg,
                                    Allocation &allocation,
                                    const MembarFilterFn &sharedFilter,
                                    TMemModel *tmem);

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_DEPENDENCIES_H
