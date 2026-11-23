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

using EdgeTransfer = llvm::function_ref<unsigned(Edge, unsigned)>;
using SyncTransfer = llvm::function_ref<unsigned(unsigned, unsigned)>;

// Start after the producer and visit each (point, state) once before internal
// sync. The visitor returns the next state or std::nullopt to stop the path.
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

void coalesceDemands(SmallVectorImpl<Demand> &demands);

MembarDemands discoverDemands(const MembarCFG<OpModel> &cfg,
                              Allocation &allocation,
                              const SMemOrdering *ordering, TMemModel *tmem);

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_DEPENDENCIES_H
