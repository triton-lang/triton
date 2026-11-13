#include "lib/Analysis/MembarCFG.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <limits>

using namespace mlir::triton::membar;

namespace {

constexpr unsigned site(unsigned point, Action action) {
  return point * ACTIONS.size() + unsigned(action);
}

// Node zero is entry. A zero insertion cost marks a node with no legal site.
MembarCFG linearCFG(llvm::ArrayRef<unsigned> costs) {
  MembarCFG cfg;
  cfg.nodes.resize(costs.size());
  for (unsigned i = 1; i < costs.size(); ++i) {
    cfg.nodes[i - 1].next.push_back({i});
    cfg.nodes[i].insertionCost = costs[i];
  }
  return cfg;
}

TEST(MembarCut, EmptyAndUncuttableDemands) {
  auto cfg = linearCFG({0, 0});
  auto solution = solveMembarCut(cfg, {});
  ASSERT_TRUE(solution);
  EXPECT_TRUE(solution->sites.none());
  EXPECT_EQ(solution->cost, 0u);
  EXPECT_EQ(solution->lowerBound, 0u);
  EXPECT_FALSE(solveMembarCut(cfg, {{0, 1, Need::RegionOrder}}));
}

TEST(MembarCut, CompletionAlsoConvergesTheWarp) {
  auto cfg = linearCFG({0, 1, 1});
  auto solution = solveMembarCut(
      cfg, {{1, 2, Need::WarpConvergence}, {1, 2, Need::WaitStore}});
  ASSERT_TRUE(solution);
  EXPECT_EQ(solution->sites.count(), 1u);
  EXPECT_TRUE(solution->sites.test(site(2, Action::WaitStore)));
  EXPECT_EQ(solution->cost, 1u);
  EXPECT_EQ(solution->lowerBound, 1u);
}

TEST(MembarCut, TwoActionsAtOneGapPayTwice) {
  constexpr unsigned cost = std::numeric_limits<unsigned>::max();
  auto cfg = linearCFG({0, cost});
  auto solution =
      solveMembarCut(cfg, {{0, 1, Need::WaitStore, Path::WaitThenOrder}});
  ASSERT_TRUE(solution);
  EXPECT_TRUE(solution->sites.test(site(1, Action::WaitStore)));
  EXPECT_TRUE(solution->sites.test(site(1, Action::RegionBarrier)));
  EXPECT_EQ(solution->cost, uint64_t{2} * cost);
  EXPECT_EQ(solution->lowerBound, solution->cost);
}

TEST(MembarCut, RefreshAfterEachSelectedSite) {
  auto cfg = linearCFG({0, 1, 1, 1});
  const Demand demands[] = {{0, 2, Need::WaitLoad, Path::WaitThenOrder},
                            {1, 3, Need::WaitStore, Path::WaitThenOrder},
                            {2, 3, Need::WarpOrder}};
  auto solution = solveMembarCut(cfg, demands);
  ASSERT_TRUE(solution);
  // After placing the load's barrier at 2, the store can share it by waiting
  // at 2. Processing the initial paths as a batch puts the store wait at 3.
  EXPECT_EQ(solution->sites.count(), 4u);
  EXPECT_TRUE(solution->sites.test(site(2, Action::WaitLoad)));
  EXPECT_TRUE(solution->sites.test(site(2, Action::WaitStore)));
  EXPECT_TRUE(solution->sites.test(site(2, Action::RegionBarrier)));
  EXPECT_TRUE(solution->sites.test(site(3, Action::RegionBarrier)));
}

TEST(MembarCut, OrderedHandoffUsesFirstBFSArrival) {
  // The short arm 0 -> 2 -> 4 reaches the sink having waited. The longer
  // arm 0 -> 1 -> 3 -> 4 would reach it without waiting.
  auto cfg = linearCFG({0, 1, 1, 1, 4});
  cfg.nodes[0].next.push_back({2});
  cfg.nodes[1].next = {{3}};
  cfg.nodes[2].next = {{4}};
  cfg.nodes[2].effects.fixedAfter = bit(Need::WaitStore);
  auto solution =
      solveMembarCut(cfg, {{0, 4, Need::WaitStore, Path::WaitThenOrder}});
  ASSERT_TRUE(solution);
  EXPECT_EQ(solution->sites.count(), 4u);
  for (unsigned gap : {2u, 3u}) {
    EXPECT_TRUE(solution->sites.test(site(gap, Action::WaitStore)));
    EXPECT_TRUE(solution->sites.test(site(gap, Action::RegionBarrier)));
  }
}

TEST(MembarCut, ReverseGapTiesKeepWarpBarrierForConvergence) {
  // Six shared store/read pairs need three barriers with forward ties and two
  // with reverse ties. The independent convergence demand has no TMEM issuer.
  auto cfg = linearCFG({0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  const Demand demands[] = {
      {1, 5, Need::RegionOrder},    {2, 8, Need::RegionOrder},
      {3, 9, Need::RegionOrder},    {4, 7, Need::RegionOrder},
      {6, 11, Need::RegionOrder},   {10, 12, Need::RegionOrder},
      {0, 1, Need::WarpConvergence}};
  auto solution = solveMembarCut(cfg, demands);
  ASSERT_TRUE(solution);
  EXPECT_EQ(solution->sites.count(), 3u);
  EXPECT_TRUE(solution->sites.test(site(1, Action::WarpBarrier)));
}

TEST(MembarCut, WeightedSharedSite) {
  // The two consumer gaps cost two each. One barrier at the fork costs three.
  const unsigned costs[] = {0, 1, 3, 2, 2};
  auto cfg = linearCFG(costs);
  cfg.nodes[2].next.push_back({4});
  cfg.nodes[3].next.clear();
  const Demand demands[] = {{1, 3, Need::RegionOrder},
                            {1, 4, Need::RegionOrder}};
  auto solution = solveMembarCut(cfg, demands);
  ASSERT_TRUE(solution);
  EXPECT_EQ(solution->sites.count(), 1u);
  EXPECT_EQ(solution->lowerBound, 3u);
  EXPECT_TRUE(solution->sites.test(site(2, Action::RegionBarrier)));
}

// Give each original node a separate predecessor for its insertion gap. This
// preserves the order of gap actions relative to the node's fixedBefore.
void checkPlacements(
    const MembarCFG &cfg, Demand demand,
    llvm::function_ref<bool(const llvm::BitVector &)> covered) {
  unsigned points = cfg.nodes.size();
  llvm::BitVector legal(points * ACTIONS.size());
  MembarCFG fixed = cfg;
  fixed.nodes.resize(2 * points);
  for (unsigned i = 0; i < points; ++i) {
    if (cfg.nodes[i].insertionCost)
      legal.set(i * ACTIONS.size(), (i + 1) * ACTIONS.size());
    fixed.nodes[i].insertionCost = 0;
    for (auto &edge : fixed.nodes[i].next)
      edge.to += points;
    fixed.nodes[points + i].next.push_back({i});
  }

  auto solution = solveMembarCut(cfg, {demand});
  ASSERT_EQ(solution.has_value(), covered(legal));
  if (solution) {
    EXPECT_TRUE(covered(solution->sites));
    for (unsigned id : solution->sites.set_bits())
      EXPECT_TRUE(legal.test(id));
  }

  // With no legal insertion sites, the solver succeeds precisely when the
  // supplied placement covers the demand. Check all placements independently.
  uint64_t optimum = std::numeric_limits<uint64_t>::max();
  for (unsigned mask = 0; mask < (1u << legal.count()); ++mask) {
    SCOPED_TRACE(mask);
    for (unsigned i = 0; i < points; ++i)
      fixed.nodes[points + i].effects.fixedAfter = 0;
    uint64_t cost = 0;
    llvm::BitVector sites(legal.size());
    for (auto [i, id] : llvm::enumerate(legal.set_bits())) {
      if (!(mask & (1u << i)))
        continue;
      sites.set(id);
      unsigned point = id / ACTIONS.size();
      cost += cfg.nodes[point].insertionCost;
      fixed.nodes[points + point].effects.fixedAfter |=
          coverage(ACTIONS[id % ACTIONS.size()]);
    }
    bool feasible = covered(sites);
    EXPECT_EQ(solveMembarCut(fixed, {demand}).has_value(), feasible);
    if (feasible)
      optimum = std::min(optimum, cost);
  }
  if (solution) {
    EXPECT_LE(solution->lowerBound, optimum);
    EXPECT_LE(optimum, solution->cost);
  }
}

TEST(MembarCut, OrderedWaitAndBarrier) {
  // S -> M -> U.
  auto cfg = linearCFG({0, 1, 1, 1});
  Demand demand{1, 3, Need::WaitStore, Path::WaitThenOrder};
  constexpr unsigned waitM = site(2, Action::WaitStore);
  constexpr unsigned waitU = site(3, Action::WaitStore);
  constexpr unsigned barrierM = site(2, Action::RegionBarrier);
  constexpr unsigned barrierU = site(3, Action::RegionBarrier);
  auto ordered = [](const llvm::BitVector &s) {
    return (s[waitM] && s[barrierM]) || ((s[waitM] || s[waitU]) && s[barrierU]);
  };
  checkPlacements(cfg, demand, ordered);

  // A fixed barrier after S is too early. A fixed wait before U follows
  // U's insertion gap, and U's internal rendezvous follows the deadline.
  cfg.nodes[1].effects.fixedAfter = fullBarrier;
  cfg.nodes[3].effects.fixedBefore = bit(Need::WaitStore);
  cfg.nodes[3].effects.fixedInside = fullBarrier;
  checkPlacements(cfg, demand, ordered);

  // A fixed rendezvous before U also covers a wait inserted at U's gap.
  cfg.nodes[3].effects.fixedBefore = fullBarrier;
  checkPlacements(cfg, demand, [](const llvm::BitVector &s) {
    return s[waitM] || s[waitU];
  });
  cfg.nodes[3].effects.fixedBefore = 0;
  cfg.nodes[2].effects.fixedInside = fullBarrier;
  cfg.nodes[2].effects.fixedAfter = bit(Need::WaitStore);
  checkPlacements(cfg, demand, [](const llvm::BitVector &s) {
    return s[waitM] || s[barrierU];
  });
}

TEST(MembarCut, OrderedPathsThroughAJoin) {
  auto cfg = linearCFG({0, 1, 1, 1, 1});
  cfg.nodes[1].next.push_back({3});
  cfg.nodes[2].next = {{4}};
  cfg.nodes[2].effects.fixedAfter = bit(Need::WaitStore);
  cfg.nodes[4].effects.fixedBefore = fullBarrier;
  // The left arm has already waited. The right needs a wait there or at
  // the join, so the two arrivals at the join cannot be conflated.
  checkPlacements(cfg, {1, 4, Need::WaitStore, Path::WaitThenOrder},
                  [](const llvm::BitVector &s) {
                    return s[site(3, Action::WaitStore)] ||
                           s[site(4, Action::WaitStore)];
                  });

  Demand self{1, 1, Need::WaitStore, Path::WaitThenOrder};
  checkPlacements(cfg, self, [](const llvm::BitVector &) { return true; });
  cfg.nodes[4].next.push_back({1, true});
  checkPlacements(cfg, self, [](const llvm::BitVector &s) {
    return s[site(3, Action::WaitStore)] || s[site(4, Action::WaitStore)] ||
           (s[site(1, Action::WaitStore)] && s[site(1, Action::RegionBarrier)]);
  });
}

TEST(MembarCut, SelfDependencyAndLoopCarriedPaths) {
  // Node two is inside the restricted arm.
  auto cfg = linearCFG({0, 1, 0, 1});
  Demand self{1, 1, Need::WarpOrder};
  checkPlacements(cfg, self, [](const llvm::BitVector &) { return true; });

  cfg.nodes[3].next.push_back({1, true});
  // Neither endpoint uses the operation's internal synchronization. The
  // cycle must traverse a real edge and either barrier can order the warp.
  cfg.nodes[1].effects.fixedInside = fullBarrier;
  checkPlacements(cfg, self, [](const llvm::BitVector &s) {
    return s[site(1, Action::WarpBarrier)] ||
           s[site(1, Action::RegionBarrier)] ||
           s[site(3, Action::WarpBarrier)] || s[site(3, Action::RegionBarrier)];
  });
  cfg.nodes[1].effects.fixedInside = 0;
  checkPlacements(cfg, {1, 2, Need::RegionOrder, Path::Any},
                  [](const llvm::BitVector &) { return false; });
  checkPlacements(cfg, {1, 2, Need::RegionOrder, Path::LoopCarried},
                  [](const llvm::BitVector &s) {
                    return s[site(1, Action::RegionBarrier)] ||
                           s[site(3, Action::RegionBarrier)];
                  });
}

TEST(MembarCut, BranchBypassesOnlyLegalGap) {
  // The consumer executes inside a physical guard.
  auto cfg = linearCFG({0, 1, 1, 0});
  Demand demand{1, 3, Need::RegionOrder};
  checkPlacements(cfg, demand, [](const llvm::BitVector &s) {
    return s[site(2, Action::RegionBarrier)];
  });
  cfg.nodes[1].next.push_back({3});
  checkPlacements(cfg, demand, [](const llvm::BitVector &) { return false; });
}

} // namespace
