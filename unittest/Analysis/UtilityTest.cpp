#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Parser/Parser.h"
#include "llvm/Support/Signals.h"
#include <gtest/gtest.h>

#include <deque>
#include <set>

namespace mlir {

TEST(Analysis, reorder) {
  SmallVector<int> shape({10, 20, 30});
  {
    SmallVector<unsigned> order({2, 1, 0});
    auto reordered = triton::applyPermutation(shape, order);
    EXPECT_EQ(reordered[0], 30);
    EXPECT_EQ(reordered[1], 20);
    EXPECT_EQ(reordered[2], 10);
  }
  {
    SmallVector<unsigned> order({1, 0, 2});
    auto reordered = triton::applyPermutation(shape, order);
    EXPECT_EQ(reordered[0], 20);
    EXPECT_EQ(reordered[1], 10);
    EXPECT_EQ(reordered[2], 30);
  }
}

TEST(Analysis, SharedMemoryEffectsPreserveResourceAndKind) {
  MLIRContext context;
  context.getOrLoadDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #shared = #ttg.swizzled_shared<{
      vec = 1, perPhase = 1, maxPhase = 1, order = [0]
    }>
    #smem = #ttg.shared_memory
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
      tt.func @access(%payload: !ttg.memdesc<2xi64, #shared, #smem>,
                      %barrier: !ttg.memdesc<1xi64, #shared, #smem>) {
        ttng.inval_barrier %barrier : !ttg.memdesc<1xi64, #shared, #smem>
        ttng.clc_try_cancel %payload, %barrier :
            !ttg.memdesc<2xi64, #shared, #smem>,
            !ttg.memdesc<1xi64, #shared, #smem>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);

  SmallVector<triton::gpu::SharedMemoryAccessKind> kinds;
  module->walk([&](MemoryEffectOpInterface op) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    op.getEffects(effects);
    for (const auto &effect : effects) {
      EXPECT_EQ(effect.getResource(), triton::gpu::SharedMemory::get());
      kinds.push_back(
          cast<triton::gpu::SharedMemoryAccessKindAttr>(effect.getParameters())
              .getValue());
    }
  });
  using Kind = triton::gpu::SharedMemoryAccessKind;
  EXPECT_EQ(kinds,
            (SmallVector<Kind>{Kind::Generic, Kind::Async, Kind::Barrier}));
}

TEST(Analysis, AddressSetExhaustiveEightUnitUniverse) {
  constexpr unsigned universe = 8;
  auto fromMask = [](unsigned mask) {
    triton::AddressSet result;
    for (unsigned bit = 0; bit < universe; ++bit)
      if (mask & (1u << bit))
        result.set(bit);
    return result;
  };

  for (unsigned lhsMask = 0; lhsMask < (1u << universe); ++lhsMask) {
    triton::AddressSet lhs = fromMask(lhsMask);

    for (unsigned rhsMask = 0; rhsMask < (1u << universe); ++rhsMask) {
      triton::AddressSet rhs = fromMask(rhsMask);
      EXPECT_EQ(lhs.intersects(rhs), (lhsMask & rhsMask) != 0);
      EXPECT_EQ(lhs.contains(rhs), (rhsMask & ~lhsMask) == 0);
      EXPECT_EQ(lhs.intersection(rhs), fromMask(lhsMask & rhsMask));

      triton::AddressSet difference = lhs;
      difference.subtract(rhs);
      EXPECT_EQ(difference, fromMask(lhsMask & ~rhsMask));
    }
  }
}

TEST(Analysis, BufferRegionViewPreservesSubviewProvenance) {
  triton::BufferRegion region{
      /*baseOffset=*/16,
      /*length=*/8,
      {{0, triton::AddressSet::fromRange(/*begin=*/16, /*length=*/8)}}};
  triton::BufferRegionView fromSubview{region, /*storageBase=*/0,
                                       /*affineOffset=*/16};
  triton::BufferRegionView fromAllocation{region, /*storageBase=*/16,
                                          /*affineOffset=*/0};

  EXPECT_EQ(fromSubview.region, fromAllocation.region);
  EXPECT_FALSE(fromSubview == fromAllocation);
  triton::RegionInfo joined = triton::RegionInfo::join(
      triton::RegionInfo({fromSubview}), triton::RegionInfo({fromAllocation}));
  EXPECT_EQ(joined.views.size(), 2);

  std::set<triton::BufferRegion> physicalRegions;
  for (const triton::BufferRegionView &view : joined.views)
    physicalRegions.insert(view.region);
  EXPECT_EQ(physicalRegions.size(), 1);
}

namespace {

uint64_t toBits(const llvm::SmallBitVector &mask) {
  assert(mask.size() <= 64);
  uint64_t result = 0;
  for (unsigned bit = 0; bit < mask.size(); ++bit)
    if (mask.test(bit))
      result |= uint64_t{1} << bit;
  return result;
}

void expectFullCoveragePartition(ArrayRef<triton::BufferRegion> regions) {
  ASSERT_EQ(regions.size(), 64u);
  triton::BufferStatePlan plan = triton::createBufferStatePlan(regions);
  ASSERT_EQ(plan.numLanes, 64u);
  ASSERT_EQ(plan.regionMasks.size(), regions.size());
  EXPECT_EQ(toBits(plan.regionMasks.front()), ~uint64_t{0});
  for (unsigned tile = 0; tile < 63; ++tile)
    EXPECT_EQ(toBits(plan.regionMasks[tile + 1]), uint64_t{1} << tile)
        << "tile " << tile;
}

triton::BufferRegion makeRegion(unsigned id, unsigned addressMask,
                                unsigned universe) {
  triton::AddressSet addresses;
  for (unsigned bit = 0; bit < universe; ++bit)
    if (addressMask & (1u << bit))
      addresses.set(bit);
  triton::BufferRegion region{/*baseOffset=*/id * 16, /*length=*/universe};
  if (!addresses.empty())
    region.ctaAddresses.emplace_back(0, std::move(addresses));
  return region;
}

bool planNeverMissesHazard(ArrayRef<unsigned> addressMasks, unsigned universe) {
  SmallVector<triton::BufferRegion> regions;
  for (auto [id, mask] : llvm::enumerate(addressMasks))
    regions.push_back(makeRegion(id, mask, universe));
  triton::BufferStatePlan plan = triton::createBufferStatePlan(regions);
  if (plan.numLanes > 64)
    return false;

  using State = std::pair<unsigned, uint64_t>;
  std::deque<State> worklist;
  std::set<State> seen;
  worklist.push_back({0, 0});
  seen.insert({0, 0});
  while (!worklist.empty()) {
    auto [exactState, planState] = worklist.front();
    worklist.pop_front();
    for (unsigned region = 0; region < regions.size(); ++region) {
      uint64_t check = toBits(plan.regionMasks[region]);
      bool exactHazard = (exactState & addressMasks[region]) != 0;
      bool planHazard = (planState & check) != 0;
      if (exactHazard && !planHazard)
        return false;

      State updated = {exactState | addressMasks[region],
                       planState | toBits(plan.regionMasks[region])};
      if (seen.insert(updated).second)
        worklist.push_back(updated);

      State completed = {exactState & ~addressMasks[region],
                         planState & ~toBits(plan.regionMasks[region])};
      if (seen.insert(completed).second)
        worklist.push_back(completed);
    }
  }
  return true;
}

bool planPublishesEveryAliasingAtom(ArrayRef<unsigned> addressMasks,
                                    unsigned universe) {
  SmallVector<triton::BufferRegion> regions;
  for (auto [id, mask] : llvm::enumerate(addressMasks))
    regions.push_back(makeRegion(id, mask, universe));
  triton::BufferStatePlan plan = triton::createBufferStatePlan(regions);
  if (plan.numLanes > 64)
    return false;

  for (unsigned generic = 0; generic < regions.size(); ++generic) {
    for (unsigned async = 0; async < regions.size(); ++async) {
      bool exactOverlap = (addressMasks[generic] & addressMasks[async]) != 0;
      bool published = (toBits(plan.regionMasks[generic]) &
                        toBits(plan.regionMasks[async])) != 0;
      if (exactOverlap != published)
        return false;
    }
  }
  return true;
}

} // namespace

TEST(Analysis, BufferStatePlanExhaustiveThreeViewsFourAddresses) {
  constexpr unsigned universe = 4;
  constexpr unsigned setCount = 1u << universe;
  for (unsigned a = 0; a < setCount; ++a)
    for (unsigned b = 0; b < setCount; ++b)
      for (unsigned c = 0; c < setCount; ++c) {
        ASSERT_TRUE(planNeverMissesHazard({a, b, c}, universe))
            << "address masks: " << a << ", " << b << ", " << c;
        ASSERT_TRUE(planPublishesEveryAliasingAtom({a, b, c}, universe))
            << "proxy publication masks: " << a << ", " << b << ", " << c;
      }
}

TEST(Analysis, BufferStatePlanUsesAtomsForSparsePartition) {
  constexpr unsigned universe = 8;
  SmallVector<unsigned> addressMasks = {
      0xff, // full
      0x55, // even
      0xaa, // odd
  };
  SmallVector<triton::BufferRegion> regions;
  for (auto [id, mask] : llvm::enumerate(addressMasks))
    regions.push_back(makeRegion(id, mask, universe));

  triton::BufferStatePlan plan = triton::createBufferStatePlan(regions);
  EXPECT_EQ(plan.numLanes, 2);

  for (unsigned exactState = 0; exactState < (1u << universe); ++exactState) {
    uint64_t planState = 0;
    if (exactState & 0x55)
      planState |= toBits(plan.regionMasks[1]);
    if (exactState & 0xaa)
      planState |= toBits(plan.regionMasks[2]);
    for (unsigned region = 0; region < regions.size(); ++region) {
      EXPECT_EQ((exactState & addressMasks[region]) != 0,
                (planState & toBits(plan.regionMasks[region])) != 0);
    }
  }
}

TEST(Analysis, BufferStatePlanKeepsPartialOverlapExact) {
  SmallVector<triton::BufferRegion> regions = {
      makeRegion(0, 0b0011, 4),
      makeRegion(1, 0b0110, 4),
  };
  triton::BufferStatePlan plan = triton::createBufferStatePlan(regions);
  EXPECT_EQ(plan.numLanes, 3);
  EXPECT_EQ(toBits(plan.regionMasks[0]), 0b011);
  EXPECT_EQ(toBits(plan.regionMasks[1]), 0b110);
  EXPECT_TRUE(planNeverMissesHazard({0b0011, 0b0110}, 4));
  EXPECT_TRUE(planPublishesEveryAliasingAtom({0b0011, 0b0110}, 4));
}

TEST(Analysis, BufferStatePlanKeepsLargePaddedIntervalsExact) {
  constexpr uint32_t tileCount = 64;
  constexpr uint32_t tileLength = 3648;
  SmallVector<triton::BufferRegion> regions;
  regions.reserve(tileCount);
  regions.push_back(
      {0,
       tileCount * tileLength,
       {{0, triton::AddressSet::fromRange(0, tileCount * tileLength)}}});
  for (uint32_t tile = 0; tile < tileCount - 1; ++tile) {
    uint32_t base = tile * tileLength;
    regions.push_back({base,
                       tileLength,
                       {{0, triton::AddressSet::fromRange(base, tileLength)}}});
  }

  expectFullCoveragePartition(regions);
}

TEST(Analysis, BufferStatePlanKeepsLargeSparsePartitionsExact) {
  constexpr uint32_t tileCount = 64;
  constexpr uint32_t stripeCount = 4;
  constexpr uint32_t stripeLength = 912;
  constexpr uint32_t tileLength = stripeCount * stripeLength;
  SmallVector<triton::BufferRegion> regions;
  regions.reserve(tileCount);
  regions.push_back(
      {0,
       tileCount * tileLength,
       {{0, triton::AddressSet::fromRange(0, tileCount * tileLength)}}});

  for (uint32_t tile = 0; tile < tileCount - 1; ++tile) {
    triton::AddressSet addresses;
    for (uint32_t stripe = 0; stripe < stripeCount; ++stripe)
      addresses.insert(triton::AddressSet::fromRange(
          (stripe * tileCount + tile) * stripeLength, stripeLength));
    regions.push_back(
        {tile * stripeLength, tileLength, {{0, std::move(addresses)}}});
  }

  expectFullCoveragePartition(regions);
}

} // namespace mlir

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
