#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"

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

TEST(Analysis, SharedMemoryResourcesRetainParent) {
  auto *shared = triton::gpu::SharedMemory::get();
  MLIRContext context;
  Block block;
  BlockArgument value = block.addArgument(IntegerType::get(&context, 32),
                                          UnknownLoc::get(&context));
  for (auto *resource : {static_cast<SideEffects::Resource *>(
                             triton::gpu::GenericSharedMemory::get()),
                         static_cast<SideEffects::Resource *>(
                             triton::gpu::AsyncSharedMemory::get()),
                         static_cast<SideEffects::Resource *>(
                             triton::gpu::BarrierSharedMemory::get())}) {
    EXPECT_TRUE(resource->isSubresourceOf(shared));

    SmallVector<MemoryEffects::EffectInstance> effects;
    triton::gpu::addSharedMemoryEffects<MemoryEffects::Write>(effects, value,
                                                              resource);
    ASSERT_EQ(effects.size(), 2u);
    EXPECT_EQ(effects[0].getResource(), shared);
    EXPECT_EQ(effects[1].getResource(), resource);
  }
}

TEST(Analysis, SharedMemoryEffectsAreClassified) {
  DialectRegistry registry;
  registry
      .insert<triton::gpu::TritonGPUDialect,
              triton::instrument::TritonInstrumentDialect,
              triton::nvidia_gpu::TritonNvidiaGPUDialect,
              triton::amdgpu::TritonAMDGPUDialect, triton::nvws::NVWSDialect,
              triton::proton::gpu::ProtonGPUDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr unsigned numValues = 128;
  auto dim0 = StringAttr::get(&context, "dim0");
  auto layout = triton::LinearLayout::identity1D(
                    1, StringAttr::get(&context, "offset"), dim0) *
                triton::LinearLayout::identity1D(
                    1, StringAttr::get(&context, "block"), dim0);
  auto encoding = triton::gpu::SharedLinearEncodingAttr::get(
      &context, std::move(layout), /*layoutAlignment=*/1);
  auto memDescType = triton::gpu::MemDescType::get(
      {1}, IntegerType::get(&context, 32), encoding,
      triton::gpu::SharedMemorySpaceAttr::get(&context),
      /*mutableMemory=*/true, /*allocShape=*/{1});
  Block block;
  SmallVector<Value> memDescOperands, tensorOperands, i1Operands, tokenOperands;
  for (unsigned i = 0; i < numValues; ++i) {
    memDescOperands.push_back(
        block.addArgument(memDescType, UnknownLoc::get(&context)));
    tensorOperands.push_back(block.addArgument(
        RankedTensorType::get({1}, IntegerType::get(&context, 32)),
        UnknownLoc::get(&context)));
    i1Operands.push_back(block.addArgument(IntegerType::get(&context, 1),
                                           UnknownLoc::get(&context)));
    tokenOperands.push_back(block.addArgument(
        triton::gpu::AsyncTokenType::get(&context), UnknownLoc::get(&context)));
  }
  SmallVector<Type> resultTypes(numValues, memDescType);
  auto isClassifiedResource = [](SideEffects::Resource *resource) {
    return resource == triton::gpu::GenericSharedMemory::get() ||
           resource == triton::gpu::AsyncSharedMemory::get() ||
           resource == triton::gpu::BarrierSharedMemory::get();
  };
  auto sameEffect = [](const MemoryEffects::EffectInstance &lhs,
                       const MemoryEffects::EffectInstance &rhs) {
    return lhs.getEffect() == rhs.getEffect() &&
           lhs.getValue() == rhs.getValue() &&
           lhs.getSymbolRef() == rhs.getSymbolRef() &&
           lhs.getParameters() == rhs.getParameters() &&
           lhs.getStage() == rhs.getStage() &&
           lhs.getEffectOnFullRegion() == rhs.getEffectOnFullRegion();
  };

  for (StringRef dialect :
       {triton::gpu::TritonGPUDialect::getDialectNamespace(),
        triton::instrument::TritonInstrumentDialect::getDialectNamespace(),
        triton::nvidia_gpu::TritonNvidiaGPUDialect::getDialectNamespace(),
        triton::amdgpu::TritonAMDGPUDialect::getDialectNamespace(),
        triton::nvws::NVWSDialect::getDialectNamespace(),
        triton::proton::gpu::ProtonGPUDialect::getDialectNamespace()}) {
    for (RegisteredOperationName opName :
         context.getRegisteredOperationsByDialect(dialect)) {
      if (!opName.hasInterface<MemoryEffectOpInterface>())
        continue;
      OperationState state(UnknownLoc::get(&context), opName.getStringRef());
      SmallVector<Value> opOperands = memDescOperands;
      if (opName.getStringRef() ==
              triton::gpu::LocalAllocOp::getOperationName() ||
          opName.getStringRef() ==
              triton::nvidia_gpu::TMEMAllocOp::getOperationName()) {
        opOperands = tensorOperands;
      } else if (opName.getStringRef() ==
                 triton::nvidia_gpu::TCGen5MMAOp::getOperationName()) {
        opOperands[3] = tokenOperands[3];
        opOperands[4] = i1Operands[4];
        opOperands[5] = i1Operands[5];
        opOperands[7] = i1Operands[7];
      } else if (opName.getStringRef() ==
                 triton::nvidia_gpu::TCGen5MMAScaledOp::getOperationName()) {
        opOperands[3] = tokenOperands[3];
        opOperands[6] = i1Operands[6];
        opOperands[7] = i1Operands[7];
        opOperands[9] = i1Operands[9];
      }
      state.addOperands(opOperands);
      state.addTypes(resultTypes);
      OwningOpRef<Operation *> op(Operation::create(state));
      NamedAttrList inherentAttrs;
      opName.populateInherentAttrs(op.get(), inherentAttrs);
      for (StringRef name : {"operandSegmentSizes", "resultSegmentSizes"}) {
        auto segments =
            dyn_cast_or_null<DenseI32ArrayAttr>(inherentAttrs.get(name));
        if (!segments)
          continue;
        SmallVector<int32_t> sizes(segments.size(), 1);
        opName.setInherentAttr(op.get(), StringAttr::get(&context, name),
                               DenseI32ArrayAttr::get(&context, sizes));
      }

      SmallVector<MemoryEffects::EffectInstance> effects;
      cast<MemoryEffectOpInterface>(op.get()).getEffects(effects);
      for (const MemoryEffects::EffectInstance &effect : effects) {
        if (effect.getResource() != triton::gpu::SharedMemory::get() ||
            !isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect()))
          continue;
        EXPECT_EQ(llvm::count_if(effects,
                                 [&](const auto &candidate) {
                                   return isClassifiedResource(
                                              candidate.getResource()) &&
                                          sameEffect(effect, candidate);
                                 }),
                  1u)
            << opName.getStringRef().str();
      }
    }
  }
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
