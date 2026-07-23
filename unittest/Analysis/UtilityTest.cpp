#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Analysis/BufferRegion.h"

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

TEST(Analysis, AddressSetMembership) {
  triton::AddressSet set = triton::AddressSet::fromRange(1, 4);
  set.insert(triton::AddressSet::fromRange(8, 6));
  set.set(10);

  EXPECT_TRUE(set.contains(1));
  EXPECT_FALSE(set.contains(5));
  EXPECT_TRUE(set.contains(13));

  triton::AddressSet same =
      triton::AddressSet::fromAddresses({13, 12, 11, 10, 9, 8, 4, 3, 2, 1, 10});
  EXPECT_EQ(set, same);
}

TEST(Analysis, AddressSetExhaustiveEightUnitUniverse) {
  constexpr unsigned universe = 8;
  for (unsigned lhsMask = 0; lhsMask < (1u << universe); ++lhsMask) {
    SmallVector<uint32_t> lhsAddresses;
    for (unsigned bit = 0; bit < universe; ++bit)
      if (lhsMask & (1u << bit))
        lhsAddresses.push_back(bit);
    triton::AddressSet lhs = triton::AddressSet::fromAddresses(lhsAddresses);

    for (unsigned rhsMask = 0; rhsMask < (1u << universe); ++rhsMask) {
      SmallVector<uint32_t> rhsAddresses;
      for (unsigned bit = 0; bit < universe; ++bit)
        if (rhsMask & (1u << bit))
          rhsAddresses.push_back(bit);
      triton::AddressSet rhs = triton::AddressSet::fromAddresses(rhsAddresses);
      EXPECT_EQ(lhs.intersects(rhs), (lhsMask & rhsMask) != 0);
      EXPECT_EQ(lhs.contains(rhs), (rhsMask & ~lhsMask) == 0);
    }
  }
}

TEST(Analysis, BufferRegionIdentityPreservesSubviewProvenance) {
  triton::AddressSet addresses = triton::AddressSet::fromRange(16, 8);
  triton::BufferRegion fromSubview(
      /*baseOffset=*/16, /*length=*/8, addresses,
      /*storageBase=*/0, /*affineOffset=*/16);
  triton::BufferRegion fromAllocation(
      /*baseOffset=*/16, /*length=*/8, addresses,
      /*storageBase=*/16, /*affineOffset=*/0);

  EXPECT_FALSE(fromSubview == fromAllocation);
  triton::RegionInfo joined = triton::RegionInfo::join(
      triton::RegionInfo({fromSubview}), triton::RegionInfo({fromAllocation}));
  EXPECT_EQ(joined.regions.size(), 2);
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

triton::BufferRegion makeRegion(unsigned id, unsigned addressMask,
                                unsigned universe) {
  SmallVector<uint32_t> addresses;
  for (unsigned bit = 0; bit < universe; ++bit)
    if (addressMask & (1u << bit))
      addresses.push_back(bit);
  return triton::BufferRegion(
      /*baseOffset=*/id * 16, /*length=*/universe,
      triton::AddressSet::fromAddresses(addresses),
      /*storageBase=*/0, /*affineOffset=*/0);
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

} // namespace mlir

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
