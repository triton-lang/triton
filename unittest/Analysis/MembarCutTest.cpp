#include "triton/Analysis/Membar/Solver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <limits>

using namespace mlir::triton::membar;

namespace {

constexpr unsigned site(unsigned point, SyncKind action) {
  return point * ACTIONS.size() + unsigned(action);
}

// Give each point its own gap unless a test explicitly shares one.
template <typename Facts>
void addPoints(MembarCFG<Facts> &cfg, unsigned count) {
  unsigned first = cfg.nodes.size();
  cfg.nodes.resize(first + count);
  for (unsigned i = first; i < cfg.nodes.size(); ++i)
    cfg.nodes[i].gap = i;
}

// Node zero is entry. A zero insertion cost marks a node with no legal site.
MembarCFG<OpModel> linearCFG(llvm::ArrayRef<unsigned> costs) {
  MembarCFG<OpModel> cfg;
  addPoints(cfg, costs.size());
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
  EXPECT_FALSE(solveMembarCut(cfg, {{0, 1, SyncKind::RegionBarrier}}));
}

TEST(MembarCut, SiteIndexesUseTheAvailableActions) {
  auto cfg = linearCFG({0, 1});
  constexpr SyncKind actions[] = {SyncKind::RegionBarrier,
                                  SyncKind::WarpBarrier};
  auto solution = solveMembarCut(cfg, {{0, 1, SyncKind::WarpBarrier}}, actions);
  ASSERT_TRUE(solution);
  EXPECT_EQ(solution->sites.size(), 4u);
  EXPECT_TRUE(solution->sites.test(3));
  EXPECT_FALSE(solveMembarCut(cfg, {{0, 1, SyncKind::WaitStore}}, actions));
}

TEST(MembarCut, TwoActionsAtOneGapPayTwice) {
  constexpr unsigned cost = std::numeric_limits<unsigned>::max();
  auto cfg = linearCFG({0, cost});
  auto solution =
      solveMembarCut(cfg, {{0, 1, SyncKind::WaitStore, Path::WaitThenOrder}});
  ASSERT_TRUE(solution);
  EXPECT_TRUE(solution->sites.test(site(1, SyncKind::WaitStore)));
  EXPECT_TRUE(solution->sites.test(site(1, SyncKind::RegionBarrier)));
  EXPECT_EQ(solution->cost, uint64_t{2} * cost);
  EXPECT_EQ(solution->lowerBound, solution->cost);
}

TEST(MembarCut, RefreshAfterEachSelectedSite) {
  auto cfg = linearCFG({0, 1, 1, 1});
  const Demand demands[] = {{0, 2, SyncKind::WaitLoad, Path::WaitThenOrder},
                            {1, 3, SyncKind::WaitStore, Path::WaitThenOrder},
                            {2, 3, SyncKind::WarpBarrier}};
  auto solution = solveMembarCut(cfg, demands);
  ASSERT_TRUE(solution);
  // After placing the load's barrier at 2, the store can share it by waiting
  // at 2. Processing the initial paths as a batch puts the store wait at 3.
  EXPECT_EQ(solution->sites.count(), 4u);
  EXPECT_TRUE(solution->sites.test(site(2, SyncKind::WaitLoad)));
  EXPECT_TRUE(solution->sites.test(site(2, SyncKind::WaitStore)));
  EXPECT_TRUE(solution->sites.test(site(2, SyncKind::RegionBarrier)));
  EXPECT_TRUE(solution->sites.test(site(3, SyncKind::RegionBarrier)));
}

TEST(MembarCut, DeferLoopBarrierToConsumer) {
  // Header 1 goes through prefix 2 to consumer 3, or exits to consumer 5.
  // Both overwrite scratch read by 3 on the preceding iteration.
  auto cfg = linearCFG({0, 4, 4, 4, 4, 1});
  for (unsigned i = 1; i <= 4; ++i)
    cfg.nodes[i].loopDepth = 1;
  cfg.nodes[1].next.push_back({5});
  cfg.nodes[4].next = {{1, true}};
  auto solution = solveMembarCut(
      cfg, {{3, 3, SyncKind::RegionBarrier}, {3, 5, SyncKind::RegionBarrier}});
  ASSERT_TRUE(solution);
  EXPECT_EQ(solution->sites.count(), 2u);
  EXPECT_TRUE(solution->sites.test(site(3, SyncKind::RegionBarrier)));
  EXPECT_TRUE(solution->sites.test(site(5, SyncKind::RegionBarrier)));
  EXPECT_EQ(solution->cost, 5u);
  EXPECT_EQ(solution->lowerBound, 4u);
}

TEST(MembarCut, ShareLoopBarrierAcrossConsumers) {
  // Each trip branches at 2 to either consumer 3 or 4, then reaches producer
  // 5. A single barrier at the fork orders both consumers on the next trip.
  auto cfg = linearCFG({0, 4, 4, 4, 4, 4, 4, 1});
  cfg.nodes[1].next.push_back({7});
  cfg.nodes[2].next.push_back({4});
  cfg.nodes[3].next = {{5}};
  cfg.nodes[6].next = {{1, true}};
  auto solution = solveMembarCut(
      cfg, {{5, 3, SyncKind::RegionBarrier}, {5, 4, SyncKind::RegionBarrier}});
  ASSERT_TRUE(solution);
  EXPECT_EQ(solution->sites.count(), 1u);
  EXPECT_TRUE(solution->sites.test(site(2, SyncKind::RegionBarrier)));
}

TEST(MembarCut, OrderedHandoffUsesFirstBFSArrival) {
  // The short arm 0 -> 2 -> 4 reaches the sink having waited. The longer
  // arm 0 -> 1 -> 3 -> 4 would reach it without waiting.
  auto cfg = linearCFG({0, 1, 1, 1, 4});
  cfg.nodes[0].next.push_back({2});
  cfg.nodes[1].next = {{3}};
  cfg.nodes[2].next = {{4}};
  cfg.nodes[2].fixed.after = bit(SyncKind::WaitStore);
  auto solution =
      solveMembarCut(cfg, {{0, 4, SyncKind::WaitStore, Path::WaitThenOrder}});
  ASSERT_TRUE(solution);
  EXPECT_EQ(solution->sites.count(), 4u);
  for (unsigned gap : {2u, 3u}) {
    EXPECT_TRUE(solution->sites.test(site(gap, SyncKind::WaitStore)));
    EXPECT_TRUE(solution->sites.test(site(gap, SyncKind::RegionBarrier)));
  }
}

TEST(MembarCut, ReverseGapTiesKeepWarpBarrier) {
  // Six shared store/read pairs need three barriers with forward ties and two
  // with reverse ties. The independent demand only needs warp ordering.
  auto cfg = linearCFG({0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  const Demand demands[] = {
      {1, 5, SyncKind::RegionBarrier},  {2, 8, SyncKind::RegionBarrier},
      {3, 9, SyncKind::RegionBarrier},  {4, 7, SyncKind::RegionBarrier},
      {6, 11, SyncKind::RegionBarrier}, {10, 12, SyncKind::RegionBarrier},
      {0, 1, SyncKind::WarpBarrier}};
  auto solution = solveMembarCut(cfg, demands);
  ASSERT_TRUE(solution);
  EXPECT_EQ(solution->sites.count(), 3u);
  EXPECT_TRUE(solution->sites.test(site(1, SyncKind::WarpBarrier)));
}

TEST(MembarCut, WeightedSharedSite) {
  // The two consumer gaps cost two each. One barrier at the fork costs three.
  const unsigned costs[] = {0, 1, 3, 2, 2};
  auto cfg = linearCFG(costs);
  cfg.nodes[2].next.push_back({4});
  cfg.nodes[3].next.clear();
  const Demand demands[] = {{1, 3, SyncKind::RegionBarrier},
                            {1, 4, SyncKind::RegionBarrier}};
  auto solution = solveMembarCut(cfg, demands);
  ASSERT_TRUE(solution);
  EXPECT_EQ(solution->sites.count(), 1u);
  EXPECT_EQ(solution->lowerBound, 3u);
  EXPECT_TRUE(solution->sites.test(site(2, SyncKind::RegionBarrier)));
}

TEST(MembarCut, EqualCostPrefersLowerRegionBarrierWeight) {
  // Producers 1 and 3 feed consumer 5. Each arm costs half as much as the join.
  auto cfg = linearCFG({0, 1, 1, 1, 1, 2});
  cfg.nodes[0].next.push_back({3});
  cfg.nodes[2].next = {{5}};
  const Demand demands[] = {{1, 5, SyncKind::RegionBarrier},
                            {3, 5, SyncKind::WarpBarrier}};

  // Forward reaches the lower bound with a region barrier at the join.
  // Reverse synchronizes the whole region only on the left path.
  auto cut = solveMembarCut(cfg, demands);
  ASSERT_TRUE(cut);
  EXPECT_EQ(cut->cost, 2u);
  EXPECT_EQ(cut->lowerBound, 2u);
  EXPECT_TRUE(cut->sites.test(site(2, SyncKind::RegionBarrier)));
  EXPECT_TRUE(cut->sites.test(site(4, SyncKind::WarpBarrier)));

  // For one warp, both scores tie. Keep the later barrier at the join.
  cfg.multiWarp = false;
  cut = solveMembarCut(cfg, demands);
  ASSERT_TRUE(cut);
  EXPECT_EQ(cut->sites.count(), 1u);
  EXPECT_TRUE(cut->sites.test(site(5, SyncKind::RegionBarrier)));
}

TEST(MembarCut, ReportEachUncoveredOptionalSinkOnce) {
  auto cfg = linearCFG({0, 0, 0});
  cfg.nodes[2].next.push_back({1, true});
  // Two producers reach 2. Only the backedge carries 1's effect to itself.
  const Demand demands[] = {{0, 2, SyncKind::RegionBarrier},
                            {1, 2, SyncKind::RegionBarrier},
                            {1, 1, SyncKind::RegionBarrier, Path::LoopCarried}};
  auto sinks = findUncoveredSinks(cfg, demands, {}, {});
  llvm::sort(sinks);
  EXPECT_EQ(sinks, (llvm::SmallVector<unsigned>{1, 2}));
}

TEST(MembarCut, OptionalHandoffStillRequiresCompletion) {
  auto cfg = linearCFG({0, 0, 1});
  cfg.nodes[1].fixed.after = fullBarrier;
  auto cut = solveMembarCut(cfg, {{0, 2, SyncKind::WaitLoad}});
  ASSERT_TRUE(cut);
  EXPECT_TRUE(cut->sites.test(site(2, SyncKind::WaitLoad)));
  EXPECT_EQ(cut->cost, 1u);
  auto sinks =
      findUncoveredSinks(cfg, {{0, 2, SyncKind::WaitLoad, Path::WaitThenOrder}},
                         ACTIONS, cut->sites);
  EXPECT_EQ(sinks, llvm::SmallVector<unsigned>{2});
}

TEST(MembarCut, RequiredCutCoversOptionalDemandAtLowerBound) {
  // The region barrier for the longer path also covers the shorter one.
  auto cfg = linearCFG({0, 1, 1, 1});
  auto cut = solveMembarCut(cfg, {{0, 3, SyncKind::RegionBarrier}});
  ASSERT_TRUE(cut);
  EXPECT_EQ(cut->cost, cut->lowerBound);
  EXPECT_EQ(cut->sites.count(), 1u);
  EXPECT_TRUE(cut->sites.test(site(3, SyncKind::RegionBarrier)));
  auto sinks = findUncoveredSinks(cfg, {{2, 3, SyncKind::RegionBarrier}},
                                  ACTIONS, cut->sites);
  EXPECT_TRUE(sinks.empty());
}

// Give each original node a separate predecessor for its insertion gap. This
// preserves the order of gap actions relative to the node's fixed.before.
void checkPlacements(const MembarCFG<OpModel> &cfg, Demand demand,
                     llvm::function_ref<bool(const llvm::BitVector &)> covered,
                     llvm::ArrayRef<SyncKind> offered = ACTIONS) {
  unsigned points = cfg.nodes.size();
  llvm::BitVector legal(points * offered.size());
  auto fixed = cfg;
  addPoints(fixed, points);
  for (unsigned i = 0; i < points; ++i) {
    if (cfg.nodes[i].insertionCost)
      legal.set(i * offered.size(), (i + 1) * offered.size());
    fixed.nodes[i].insertionCost = 0;
    for (auto &edge : fixed.nodes[i].next)
      edge.to += points;
    fixed.nodes[points + i].next.push_back({i});
  }

  auto solution = solveMembarCut(cfg, {demand}, offered);
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
      fixed.nodes[points + i].fixed.after = 0;
    uint64_t cost = 0;
    llvm::BitVector sites(legal.size());
    for (auto [i, id] : llvm::enumerate(legal.set_bits())) {
      if (!(mask & (1u << i)))
        continue;
      sites.set(id);
      unsigned point = id / offered.size();
      cost += cfg.nodes[point].insertionCost;
      fixed.nodes[points + point].fixed.after |=
          coverage(offered[id % offered.size()]);
    }
    bool feasible = covered(sites);
    EXPECT_EQ(solveMembarCut(fixed, {demand}, offered).has_value(), feasible);
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
  Demand demand{1, 3, SyncKind::WaitStore, Path::WaitThenOrder};
  constexpr unsigned waitM = site(2, SyncKind::WaitStore);
  constexpr unsigned waitU = site(3, SyncKind::WaitStore);
  constexpr unsigned barrierM = site(2, SyncKind::RegionBarrier);
  constexpr unsigned barrierU = site(3, SyncKind::RegionBarrier);
  auto ordered = [](const llvm::BitVector &s) {
    return (s[waitM] && s[barrierM]) || ((s[waitM] || s[waitU]) && s[barrierU]);
  };
  checkPlacements(cfg, demand, ordered);

  // A fixed barrier after S is too early. A fixed wait before U follows
  // U's insertion gap, and U's internal rendezvous follows the deadline.
  cfg.nodes[1].fixed.after = fullBarrier;
  cfg.nodes[3].fixed.before = bit(SyncKind::WaitStore);
  cfg.nodes[3].fixed.inside = fullBarrier;
  checkPlacements(cfg, demand, ordered);

  // A fixed rendezvous before U also covers a wait inserted at U's gap.
  cfg.nodes[3].fixed.before = fullBarrier;
  checkPlacements(cfg, demand, [](const llvm::BitVector &s) {
    return s[waitM] || s[waitU];
  });
  cfg.nodes[3].fixed.before = 0;
  cfg.nodes[2].fixed.inside = fullBarrier;
  cfg.nodes[2].fixed.after = bit(SyncKind::WaitStore);
  checkPlacements(cfg, demand, [](const llvm::BitVector &s) {
    return s[waitM] || s[barrierU];
  });
}

TEST(MembarCut, CTAFenceAfterClusterAcquire) {
  // Peer store -> CTA fence -> cluster rendezvous -> local TMA reader.
  // A gap at the rendezvous is still on the writer's side.
  auto cfg = linearCFG({0, 1, 1, 1});
  const SyncKind fences[] = {SyncKind::ClusterProxyFence,
                             SyncKind::CTAProxyFence};
  Demand demand{0, 3, SyncKind::CTAProxyFence, Path::ClusterThenProxy};
  cfg.nodes[1].fixed.after = coverage(SyncKind::CTAProxyFence);

  // A relaxed cluster barrier leaves the remote store in the cluster window.
  checkPlacements(
      cfg, demand,
      [](const llvm::BitVector &s) { return s[2] || s[4] || s[6]; }, fences);

  cfg.nodes[2].fixed.after = bit(SyncKind::ClusterBarrier);
  checkPlacements(
      cfg, demand,
      [](const llvm::BitVector &s) {
        return s[2] || s[4] || s[6] || s[7]; // Cluster at 1/2/3 or CTA at 3.
      },
      fences);

  auto alreadyCovered = [](const llvm::BitVector &) { return true; };
  cfg.nodes[1].fixed.after = coverage(SyncKind::ClusterProxyFence);
  checkPlacements(cfg, demand, alreadyCovered, fences);
  cfg.nodes[1].fixed.after = coverage(SyncKind::CTAProxyFence);
  cfg.nodes[3].fixed.before = coverage(SyncKind::CTAProxyFence);
  checkPlacements(cfg, demand, alreadyCovered, fences);

  // On the path bypassing the acquire, the fixed CTA fence at the join still
  // excludes the producer's remote write.
  cfg.nodes[1].next.push_back({3});
  checkPlacements(
      cfg, demand,
      [](const llvm::BitVector &s) {
        return s[2] || s[6]; // Cluster at the fork or the join.
      },
      fences);
}

TEST(MembarCut, WorkerFencesFollowTheirIssuerThroughTheJoin) {
  // Two launches in sequence. Each default fences before its join.
  MembarCFG<OpModel> cfg;
  addPoints(cfg, 7);
  cfg.nodes[0].next = {{1}, {2, false, WarpEdge::EnterWorker}};
  cfg.nodes[1].fixed.after = coverage(SyncKind::RegionBarrier);
  cfg.nodes[1].next = {{3}};
  cfg.nodes[2].inWorker = true;
  cfg.nodes[2].next = {{3, false, WarpEdge::ExitWorker}};
  cfg.nodes[3].next = {{4}, {5, false, WarpEdge::EnterWorker}};
  cfg.nodes[4].fixed.after = coverage(SyncKind::RegionBarrier);
  cfg.nodes[4].next = {{6}};
  cfg.nodes[5].inWorker = true;
  cfg.nodes[5].next = {{6, false, WarpEdge::ExitWorker}};

  const SyncKind fences[] = {SyncKind::RegionBarrier};
  // With no legal insertion gaps, a solution exists only for covered demands.
  auto covered = [&](unsigned from, unsigned to) {
    return solveMembarCut(cfg, {{from, to, SyncKind::RegionBarrier}}, fences)
        .has_value();
  };
  EXPECT_TRUE(covered(0, 3));  // The default orders its own incoming effects.
  EXPECT_FALSE(covered(0, 2)); // The worker can consume an inherited effect.
  EXPECT_FALSE(covered(2, 3)); // The worker's own effects escape its launch.
  EXPECT_FALSE(covered(2, 5)); // They can enter the next launch's worker.
  EXPECT_TRUE(covered(2, 6));  // They cannot escape that worker again.
}

TEST(MembarCut, SharedCalleeFenceIsChargedOnce) {
  MembarCFG<OpModel> cfg;
  addPoints(cfg, 3);
  cfg.nodes[0].next = {{1}, {2, false, WarpEdge::EnterWorker}};
  cfg.nodes[2].inWorker = true;
  // Both role contexts execute the same physical instruction.
  cfg.nodes[2].gap = cfg.nodes[1].gap;
  cfg.nodes[1].insertionCost = cfg.nodes[2].insertionCost = 4;
  const SyncKind fences[] = {SyncKind::RegionBarrier};
  auto cut = solveMembarCut(
      cfg, {{0, 1, SyncKind::RegionBarrier}, {0, 2, SyncKind::RegionBarrier}},
      fences);
  ASSERT_TRUE(cut);
  EXPECT_EQ(cut->sites.count(), 1u);
  EXPECT_EQ(cut->cost, 4u);
  EXPECT_EQ(cut->lowerBound, cut->cost);
}

TEST(MembarCut, OrderedPathsThroughAJoin) {
  auto cfg = linearCFG({0, 1, 1, 1, 1});
  cfg.nodes[1].next.push_back({3});
  cfg.nodes[2].next = {{4}};
  cfg.nodes[2].fixed.after = bit(SyncKind::WaitStore);
  cfg.nodes[4].fixed.before = fullBarrier;
  // The left arm has already waited. The right needs a wait there or at
  // the join, so the two arrivals at the join cannot be conflated.
  checkPlacements(cfg, {1, 4, SyncKind::WaitStore, Path::WaitThenOrder},
                  [](const llvm::BitVector &s) {
                    return s[site(3, SyncKind::WaitStore)] ||
                           s[site(4, SyncKind::WaitStore)];
                  });

  Demand self{1, 1, SyncKind::WaitStore, Path::WaitThenOrder};
  checkPlacements(cfg, self, [](const llvm::BitVector &) { return true; });
  cfg.nodes[4].next.push_back({1, true});
  checkPlacements(cfg, self, [](const llvm::BitVector &s) {
    return s[site(3, SyncKind::WaitStore)] || s[site(4, SyncKind::WaitStore)] ||
           (s[site(1, SyncKind::WaitStore)] &&
            s[site(1, SyncKind::RegionBarrier)]);
  });
}

TEST(MembarCut, SelfDependencyAndLoopCarriedPaths) {
  // Node two is inside the restricted arm.
  auto cfg = linearCFG({0, 1, 0, 1});
  Demand self{1, 1, SyncKind::WarpBarrier};
  checkPlacements(cfg, self, [](const llvm::BitVector &) { return true; });

  cfg.nodes[3].next.push_back({1, true});
  // Neither endpoint uses the operation's internal synchronization. The
  // cycle must traverse a real edge and either barrier can order the warp.
  cfg.nodes[1].fixed.inside = fullBarrier;
  checkPlacements(cfg, self, [](const llvm::BitVector &s) {
    return s[site(1, SyncKind::WarpBarrier)] ||
           s[site(1, SyncKind::RegionBarrier)] ||
           s[site(3, SyncKind::WarpBarrier)] ||
           s[site(3, SyncKind::RegionBarrier)];
  });
  cfg.nodes[1].fixed.inside = 0;
  checkPlacements(cfg, {1, 2, SyncKind::RegionBarrier, Path::Any},
                  [](const llvm::BitVector &) { return false; });
  checkPlacements(cfg, {1, 2, SyncKind::RegionBarrier, Path::LoopCarried},
                  [](const llvm::BitVector &s) {
                    return s[site(1, SyncKind::RegionBarrier)] ||
                           s[site(3, SyncKind::RegionBarrier)];
                  });
}

TEST(MembarCut, BranchBypassesOnlyLegalGap) {
  // The consumer executes inside a physical guard.
  auto cfg = linearCFG({0, 1, 1, 0});
  Demand demand{1, 3, SyncKind::RegionBarrier};
  checkPlacements(cfg, demand, [](const llvm::BitVector &s) {
    return s[site(2, SyncKind::RegionBarrier)];
  });
  cfg.nodes[1].next.push_back({3});
  checkPlacements(cfg, demand, [](const llvm::BitVector &) { return false; });
}

} // namespace
