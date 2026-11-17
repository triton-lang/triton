#ifndef TRITON_ANALYSIS_MEMBAR_DEPENDENCIES_H
#define TRITON_ANALYSIS_MEMBAR_DEPENDENCIES_H

#include "triton/Analysis/Membar/CFGBuilder.h"

namespace mlir::triton::membar {

// Loop-carried shared hazards and ordered TMEM handoffs each need one extra
// state. The latter always ends with a region barrier.
enum class Path { Any, LoopCarried, WaitThenOrder };

struct Demand {
  unsigned source, sink;
  Need first;
  Path path = Path::Any;
};

using EdgeTransfer = llvm::function_ref<unsigned(Edge, unsigned)>;
using SyncTransfer = llvm::function_ref<unsigned(unsigned, unsigned)>;

// Visit each (point, state) once, observing the consumer before internal sync.
// A sync transfer returns zero when it covers the producer. Completion walks
// omit the transfer so they can propagate an uncommitted cp.async group (age
// 0).
template <typename Facts, typename Visit>
void walkEdges(const MembarCFG<Facts> &cfg, ArrayRef<Edge> edges,
               unsigned state, Visit visit, SyncTransfer apply = {},
               EdgeTransfer onEdge = {}) {
  llvm::SmallSetVector<std::pair<unsigned, unsigned>, 16> arrivals;
  auto enqueue = [&](ArrayRef<Edge> edges, unsigned state) {
    for (Edge edge : edges) {
      unsigned live =
          apply ? apply(state, cfg.nodes[edge.to].fixed.before) : state;
      if (!apply || live)
        arrivals.insert({edge.to, onEdge ? onEdge(edge, live) : live});
    }
  };
  enqueue(edges, state);
  for (unsigned i = 0; i != arrivals.size(); ++i) {
    auto [point, incoming] = arrivals[i];
    const auto &node = cfg.nodes[point];
    if (auto next = visit(point, incoming))
      enqueue(node.next,
              apply ? apply(apply(*next, node.fixed.inside), node.fixed.after)
                    : *next);
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
  const auto &producer = cfg.nodes[source];
  walkEdges(
      cfg, producer.next, apply(state, producer.fixed.after),
      [&](unsigned sink, unsigned live) {
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
