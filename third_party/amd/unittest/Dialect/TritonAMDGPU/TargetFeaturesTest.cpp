#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"

#include <algorithm>
#include <gtest/gtest.h>

namespace mlir::triton::amdgpu {
namespace {

struct ExpectedTarget {
  const char *arch;
  ISAFamily family;
  int warpSize;
  int sharedMemorySize;
  int sharedMemoryBanks;
  bool strict;
};

// Heuristics were tuned against these LLVM-provided values; pin them so an LLVM
// change prompts revisiting those heuristics.
constexpr ExpectedTarget kExpectedTargets[] = {
    {"gfx90a", ISAFamily::CDNA2, 64, 64 * 1024, 32, false},
    {"gfx942", ISAFamily::CDNA3, 64, 64 * 1024, 32, false},
    {"gfx950", ISAFamily::CDNA4, 64, 160 * 1024, 64, false},
    {"gfx1100", ISAFamily::RDNA3, 32, 64 * 1024, 32, false},
    {"gfx1151", ISAFamily::RDNA3, 32, 64 * 1024, 32, false},
    {"gfx1170", ISAFamily::RDNA4m, 32, 64 * 1024, 32, false},
    {"gfx1200", ISAFamily::RDNA4, 32, 64 * 1024, 32, false},
    {"gfx1201", ISAFamily::RDNA4, 32, 64 * 1024, 32, false},
    {"gfx1250", ISAFamily::CDNA5, 32, 320 * 1024, 64, false},
    {"gfx1250-strict", ISAFamily::CDNA5, 32, 320 * 1024, 64, true},
};

class TargetFeaturesTest : public ::testing::TestWithParam<ExpectedTarget> {};

TEST_P(TargetFeaturesTest, MatchesTunedValues) {
  const ExpectedTarget &expected = GetParam();
  TargetFeatures features{StringRef(expected.arch)};
  EXPECT_EQ(features.getArch(), expected.arch);
  EXPECT_EQ(features.getISAFamily(), expected.family);
  EXPECT_EQ(features.getWarpSize(), expected.warpSize);
  EXPECT_EQ(features.getSharedMemorySize(), expected.sharedMemorySize);
  EXPECT_EQ(features.getSharedMemoryBanks(), expected.sharedMemoryBanks);
  EXPECT_EQ(features.isGFX1250Strict(), expected.strict);
  EXPECT_FALSE(features.getTargetInfo().isUnknown());
}

INSTANTIATE_TEST_SUITE_P(
    AMDTargets, TargetFeaturesTest, ::testing::ValuesIn(kExpectedTargets),
    [](const ::testing::TestParamInfo<ExpectedTarget> &info) {
      std::string name = info.param.arch;
      std::replace(name.begin(), name.end(), '-', '_');
      return name;
    });

TEST(TargetFeaturesSpellingTest, ClassifiesGenericTargets) {
  struct {
    const char *arch;
    ISAFamily family;
  } cases[] = {
      {"gfx9-4-generic", ISAFamily::CDNA3},
      {"gfx11-generic", ISAFamily::RDNA3},
      {"gfx11-7-generic", ISAFamily::RDNA4m},
      {"gfx12-generic", ISAFamily::RDNA4},
      {"gfx12-5-generic", ISAFamily::CDNA5},
      // Generations Triton does not target stay unknown.
      {"gfx9-generic", ISAFamily::Unknown},
      {"gfx908", ISAFamily::Unknown},
      {"gfx1030", ISAFamily::Unknown},
      {"gfx1310", ISAFamily::Unknown},
  };
  for (const auto &c : cases) {
    TargetFeatures features{StringRef(c.arch)};
    EXPECT_EQ(features.getISAFamily(), c.family) << c.arch;
    EXPECT_FALSE(features.isGFX1250Strict()) << c.arch;
  }
}

TEST(TargetFeaturesSpellingTest, ExposesFeatureBits) {
  TargetFeatures gfx942(StringRef("gfx942"));
  EXPECT_TRUE(gfx942.has(llvm::AMDGPU::FEAT_GFX940_INSTS));
  EXPECT_TRUE(gfx942.getTargetInfo().hasFnuzFp8());
  EXPECT_FALSE(gfx942.getTargetInfo().hasOcpFp8());

  TargetFeatures gfx950(StringRef("gfx950"));
  EXPECT_TRUE(gfx950.has(llvm::AMDGPU::FEAT_GFX950_INSTS));
  EXPECT_TRUE(gfx950.getTargetInfo().hasOcpFp8());
}

TEST(TargetFeaturesSpellingTest, UnknownTargets) {
  for (std::optional<StringRef> arch :
       {std::optional<StringRef>(), std::optional<StringRef>(""),
        std::optional<StringRef>("gfx1130"),
        std::optional<StringRef>("not-a-gpu")}) {
    TargetFeatures features(arch);
    EXPECT_EQ(features.getISAFamily(), ISAFamily::Unknown);
    EXPECT_TRUE(features.getTargetInfo().isUnknown());
    EXPECT_FALSE(features.has(llvm::AMDGPU::FEAT_GFX9_INSTS));
    EXPECT_EQ(features.getWarpSize(), 32);
    EXPECT_EQ(features.getSharedMemorySize(), 64 * 1024);
    EXPECT_EQ(features.getSharedMemoryBanks(), 32);
  }
}

} // namespace
} // namespace mlir::triton::amdgpu
