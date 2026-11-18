#include "triton/Analysis/Membar/Solver.h"
#include "triton/Analysis/Membar/AsyncFenceModel.h"
#include "triton/Analysis/Membar/ClusterBarrierModel.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SparseSet.h"
#include <limits>
#include <tuple>

namespace mlir::triton::membar {
namespace {

using SiteList = SmallVector<unsigned>;
constexpr unsigned unseen = std::numeric_limits<unsigned>::max();

auto origin(const Demand &d) { return std::tie(d.first, d.path, d.source); }

template <typename Facts> class CutSolver {
public:
  CutSolver(const MembarCFG<Facts> &cfg, ArrayRef<Demand> unsorted,
            ArrayRef<Action> actions)
      : cfg(cfg), demands(unsorted), actions(actions) {
    llvm::sort(demands, [](const Demand &a, const Demand &b) {
      return std::tuple(origin(a), a.sink) < std::tuple(origin(b), b.sink);
    });
  }

  std::optional<CutSolution> choose(bool reverseTies) const;

private:
  std::optional<SiteList> shortestCut(const llvm::BitVector &selected,
                                      size_t &first) const;

  const MembarCFG<Facts> &cfg;
  SmallVector<Demand> demands;
  ArrayRef<Action> actions;
};

// Keep the first uncovered path with the fewest distinct legal sites. A
// covered prefix stays covered while the placement grows.
template <typename Facts>
std::optional<SiteList>
CutSolver<Facts>::shortestCut(const llvm::BitVector &selected,
                              size_t &first) const {
  SmallVector<unsigned> credits(cfg.nodes.size(), 0), parent, queue;
  for (unsigned id : selected.set_bits())
    credits[id / actions.size()] |= coverage(actions[id % actions.size()]);
  llvm::SparseSet<unsigned> labels;
  labels.setUniverse(selected.size());
  std::optional<SiteList> best;
  size_t next = 0;

  // The root is the departure after the producer. Returning to the producer
  // itself requires traversing a CFG edge.
  for (size_t i = first; i != demands.size(); ++i) {
    const Demand &d = demands[i];
    // A worker's own effect can leave once. Incoming parent effects follow
    // the default continuation, while remaining visible to worker consumers.
    const bool workerSource = cfg.nodes[d.source].inWorker;
    const unsigned width = workerSource ? 4 : 2;
    const unsigned root = width * cfg.nodes.size();
    if (queue.empty() || origin(demands[i - 1]) != origin(d)) {
      parent.assign(root, unseen);
      queue.assign(1, root);
      next = 0;
    }
    // Only a cluster-scoped fence covers a peer access before its owner
    // acquires it. The insertion gap at the acquire still precedes it.
    auto needAtGap = [&](bool advanced) {
      return d.path == Path::ClusterThenProxy && !advanced ? Need::ProxyCluster
                                                           : d.first;
    };
    auto covered = [&](unsigned mask, bool &advanced) {
      if (d.path != Path::WaitThenOrder)
        return bool(mask & bit(needAtGap(advanced)));
      advanced |= bool(mask & bit(d.first));
      return advanced && (mask & bit(Need::RegionOrder));
    };

    // Resume this group's BFS until the sink is reached or the queue is empty.
    unsigned sink = unseen;
    for (;;) {
      for (unsigned flags = 0; flags < width; ++flags) {
        unsigned point = width * d.sink + flags;
        if (parent[point] != unseen &&
            (d.path != Path::LoopCarried || (flags & 1))) {
          sink = point;
          break;
        }
      }
      if (sink != unseen || next == queue.size())
        break;
      unsigned state = queue[next++];
      const Node<Facts> &from =
          cfg.nodes[state == root ? d.source : state / width];
      bool advanced = state & 1;
      if ((state != root && covered(from.fixed.inside, advanced)) ||
          covered(from.fixed.after, advanced))
        continue;
      if (d.path == Path::ClusterThenProxy &&
          (from.fixed.after & bit(Need::ClusterAcquire)))
        advanced = true;
      for (Edge edge : from.next) {
        bool mayExitWorker = workerSource && (state == root || (state & 2));
        if (!edge.follow(mayExitWorker))
          continue;
        bool after = advanced || (d.path == Path::LoopCarried &&
                                  cfg.invalidatesIteration(edge, d.source));
        const auto &to = cfg.nodes[edge.to];
        if (covered(credits[to.gap], after) || covered(to.fixed.before, after))
          continue;
        unsigned child = width * edge.to + 2 * mayExitWorker + after;
        if (parent[child] == unseen) {
          parent[child] = state;
          queue.push_back(child);
        }
      }
    }
    if (sink == unseen) {
      if (!best)
        ++first;
      continue;
    }

    // A violated handoff needs a wait up to its last rendezvous or another
    // rendezvous after it. At a shared gap, the wait executes first.
    labels.clear();
    bool needsBarrier = d.path == Path::WaitThenOrder;
    for (unsigned state = sink; state != root; state = parent[state]) {
      const Node<Facts> &node = cfg.nodes[state / width];
      unsigned mask = credits[node.gap] | node.fixed.before;
      if (state != sink)
        mask |= node.fixed.inside | node.fixed.after;
      needsBarrier &= !(mask & bit(Need::RegionOrder));
      if (!node.insertionCost)
        continue;
      Need need = needsBarrier ? Need::RegionOrder : needAtGap(state & 1);
      for (unsigned i = actions.size(); i-- > 0;)
        if (coverage(actions[i]) & bit(need))
          labels.insert(node.gap * actions.size() + i);
      if (best && labels.size() >= best->size())
        break;
    }
    if (labels.size() <= 1)
      return llvm::to_vector(labels);
    if (!best || labels.size() < best->size())
      best = llvm::to_vector(labels);
  }
  return best;
}

template <typename Facts>
std::optional<CutSolution> CutSolver<Facts>::choose(bool reverseTies) const {
  SmallVector<unsigned> slack, order;
  for (const Node<Facts> &node : cfg.nodes)
    slack.append(actions.size(), node.insertionCost);
  SmallVector<unsigned> loopSlack = slack;
  CutSolution result{llvm::BitVector(slack.size())};
  auto cost = [&](unsigned site) {
    return cfg.nodes[site / actions.size()].insertionCost;
  };
  size_t first = 0;
  while (auto cut = shortestCut(result.sites, first)) {
    if (cut->empty())
      return std::nullopt;
    // Charge this path before searching again with the selected site.
    auto less = [&](unsigned a, unsigned b) { return slack[a] < slack[b]; };
    unsigned site = reverseTies ? *llvm::min_element(llvm::reverse(*cut), less)
                                : *llvm::min_element(*cut, less);
    unsigned charge = slack[site];
    // Reverse the choice of gap, preserving action preference at that gap.
    if (reverseTies)
      site = *llvm::find_if(*cut, [&](unsigned id) {
        return id / actions.size() == site / actions.size() &&
               slack[id] == charge;
      });
    // A cold exit can share a loop barrier, but saving its one-off cost
    // need not pull a rendezvous earlier on every iteration. At equal cost,
    // compare savings within the loop, then distance to the consumer
    // (the witness lists sites from the consumer back to the producer).
    unsigned siteCost = cost(site);
    if (loopSlack[site] > charge && llvm::all_of(*cut, [&](unsigned id) {
          return actions[id % actions.size()] == Action::RegionBarrier &&
                 cost(id) == siteCost;
        }))
      site = *llvm::min_element(*cut, [&](unsigned a, unsigned b) {
        return loopSlack[a] < loopSlack[b];
      });
    unsigned depth = cfg.nodes[site / actions.size()].loopDepth;
    for (unsigned id : *cut) {
      slack[id] -= charge;
      if (cost(id) <= siteCost ||
          cfg.nodes[id / actions.size()].loopDepth <= depth)
        loopSlack[id] -= charge;
    }
    result.sites.set(site);
    result.cost += cost(site);
    result.lowerBound += charge;
    if (cut->size() > 1)
      order.push_back(site);
  }
  if (result.cost == result.lowerBound)
    return result;
  for (unsigned site : llvm::reverse(order)) {
    result.sites.reset(site);
    first = 0;
    if (!shortestCut(result.sites, first))
      result.cost -= cost(site);
    else
      result.sites.set(site);
  }
  return result;
}

} // namespace

template <typename Facts>
std::optional<CutSolution> solveMembarCut(const MembarCFG<Facts> &cfg,
                                          ArrayRef<Demand> demands,
                                          ArrayRef<Action> actions) {
  CutSolver solver(cfg, demands, actions);
  auto best = solver.choose(false);
  if (!best || best->cost == best->lowerBound)
    return best;
  auto other = solver.choose(true);
  assert(other && "changing ties cannot change feasibility");
  if (other->cost < best->cost)
    best.swap(other);
  best->lowerBound = std::max(best->lowerBound, other->lowerBound);
  return best;
}

template std::optional<CutSolution>
solveMembarCut(const MembarCFG<OpModel> &, ArrayRef<Demand>, ArrayRef<Action>);
template std::optional<CutSolution>
solveMembarCut(const MembarCFG<AsyncFenceFacts> &, ArrayRef<Demand>,
               ArrayRef<Action>);
template std::optional<CutSolution>
solveMembarCut(const MembarCFG<ClusterBarrierModel::Facts> &, ArrayRef<Demand>,
               ArrayRef<Action>);

} // namespace mlir::triton::membar
