#include "MembarCFG.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SparseSet.h"
#include <limits>
#include <tuple>

namespace mlir::triton::membar {
namespace {

using SiteList = SmallVector<unsigned>;
constexpr unsigned unseen = std::numeric_limits<unsigned>::max();

auto origin(const Demand &d) { return std::tie(d.first, d.path, d.source); }

// Keep the first uncovered path with the fewest distinct legal sites. A
// covered prefix stays covered while the placement grows.
std::optional<SiteList> shortestCut(const MembarCFG &cfg,
                                    ArrayRef<Demand> demands,
                                    const llvm::BitVector &selected,
                                    size_t &first) {
  SmallVector<unsigned> credits(cfg.nodes.size(), 0), parent, queue;
  for (unsigned id : selected.set_bits())
    credits[id / ACTIONS.size()] |= coverage(ACTIONS[id % ACTIONS.size()]);
  llvm::SparseSet<unsigned> labels;
  labels.setUniverse(selected.size());
  std::optional<SiteList> best;
  size_t next = 0;

  // The root is the departure after the producer. Returning to the producer
  // itself requires traversing a CFG edge.
  const unsigned root = 2 * cfg.nodes.size();
  for (size_t i = first; i != demands.size(); ++i) {
    const Demand &d = demands[i];
    if (queue.empty() || origin(demands[i - 1]) != origin(d)) {
      parent.assign(root, unseen);
      queue.assign(1, root);
      next = 0;
    }
    auto covered = [&](unsigned mask, bool &advanced) {
      if (d.path != Path::WaitThenOrder)
        return bool(mask & bit(d.first));
      advanced |= bool(mask & bit(d.first));
      return advanced && (mask & bit(Need::RegionOrder));
    };

    // Resume this group's BFS until the sink is reached or the queue is empty.
    unsigned sink = 2 * d.sink + (d.path == Path::LoopCarried);
    while (parent[sink] == unseen) {
      if (d.path == Path::WaitThenOrder && parent[sink + 1] != unseen) {
        ++sink;
        break;
      }
      if (next == queue.size())
        break;
      unsigned state = queue[next++];
      const Node &from = cfg.nodes[state == root ? d.source : state / 2];
      bool advanced = state & 1;
      if ((state != root && covered(from.effects.fixedInside, advanced)) ||
          covered(from.effects.fixedAfter, advanced))
        continue;
      for (Edge edge : from.next) {
        bool after = advanced ||
                     (d.path == Path::LoopCarried && edge.invalidatesIteration);
        if (covered(credits[edge.to], after) ||
            covered(cfg.nodes[edge.to].effects.fixedBefore, after))
          continue;
        unsigned child = 2 * edge.to + after;
        if (parent[child] == unseen) {
          parent[child] = state;
          queue.push_back(child);
        }
      }
    }
    if (parent[sink] == unseen) {
      if (!best)
        ++first;
      continue;
    }

    // A violated handoff needs a wait up to its last rendezvous or another
    // rendezvous after it. At a shared gap, the wait executes first.
    labels.clear();
    bool needsBarrier = d.path == Path::WaitThenOrder;
    for (unsigned state = sink; state != root; state = parent[state]) {
      unsigned point = state / 2;
      const Node &node = cfg.nodes[point];
      unsigned mask = credits[point] | node.effects.fixedBefore;
      if (state != sink)
        mask |= node.effects.fixedInside | node.effects.fixedAfter;
      needsBarrier &= !(mask & bit(Need::RegionOrder));
      if (!node.insertionCost)
        continue;
      for (Action action : llvm::reverse(ACTIONS))
        if (coverage(action) & bit(needsBarrier ? Need::RegionOrder : d.first))
          labels.insert(point * ACTIONS.size() + unsigned(action));
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

std::optional<CutSolution> choose(const MembarCFG &cfg,
                                  ArrayRef<Demand> demands, bool reverseTies) {
  SmallVector<unsigned> slack, order;
  for (const Node &node : cfg.nodes)
    slack.append(ACTIONS.size(), node.insertionCost);
  CutSolution result{llvm::BitVector(slack.size())};
  auto cost = [&](unsigned site) {
    return cfg.nodes[site / ACTIONS.size()].insertionCost;
  };
  size_t first = 0;
  while (auto cut = shortestCut(cfg, demands, result.sites, first)) {
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
        return id / ACTIONS.size() == site / ACTIONS.size() &&
               slack[id] == charge;
      });
    for (unsigned id : *cut)
      slack[id] -= charge;
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
    if (!shortestCut(cfg, demands, result.sites, first))
      result.cost -= cost(site);
    else
      result.sites.set(site);
  }
  return result;
}

} // namespace

std::optional<CutSolution> solveMembarCut(const MembarCFG &cfg,
                                          ArrayRef<Demand> demands) {
  SmallVector<Demand> sorted(demands);
  llvm::sort(sorted, [](const Demand &a, const Demand &b) {
    return std::tuple(origin(a), a.sink) < std::tuple(origin(b), b.sink);
  });
  auto best = choose(cfg, sorted, false);
  if (!best || best->cost == best->lowerBound)
    return best;
  auto other = choose(cfg, sorted, true);
  assert(other && "changing ties cannot change feasibility");
  if (other->cost < best->cost)
    best.swap(other);
  best->lowerBound = std::max(best->lowerBound, other->lowerBound);
  return best;
}

} // namespace mlir::triton::membar
