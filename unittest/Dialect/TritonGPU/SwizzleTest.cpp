#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

#include "mlir/Support/LLVM.h"
#include "llvm/Support/Signals.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::triton;

using mlir::triton::gpu::intersectionBasis;
using mlir::triton::gpu::optimalSwizzling;

namespace {

SmallVector<int32_t> flatten(const LinearLayout &ll, StringAttr dim) {
  assert(ll.hasInDim(dim) && "in dim must exist");
  SmallVector<int32_t> vec;
  for (const auto &basis : ll.getBases().lookup(dim)) {
    assert(basis.size() == 1 && "basis must be a single int32_t");
    vec.push_back(basis[0]);
  }
  return vec;
}
class SwizzleTest : public ::testing::Test {
public:
  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

  std::pair<int, int> logBankConflicts(const LinearLayout &src,
                                       const LinearLayout &dst,
                                       const LinearLayout &smem) {
    // build vector + segment basis
    auto srcFlat = src.flattenOuts();
    auto dstFlat = dst.flattenOuts();
    auto smemFlat = smem.flattenOuts();
    auto vecBasis = flatten(smemFlat, S("vector"));
    auto segBasis = flatten(smemFlat, S("segment"));
    auto bank0 = llvm::to_vector(llvm::concat<int32_t>(vecBasis, segBasis));
    int32_t rank = smem.getTotalOutDimSizeLog2();
    // compute conflicts
    int read =
        intersectionBasis(bank0, flatten(dstFlat, S("lane")), rank).size();
    int write =
        intersectionBasis(bank0, flatten(srcFlat, S("lane")), rank).size();
    return {read, write};
  }

  int32_t logWavefronts(const LinearLayout &ll, int32_t bitwidth) {
    // Returns the log of the wavefronts used by the read and write operations
    // In other words:
    // LDS.32, STS.32 -> Return 0
    // LDS.64, STS.64 -> Return 1
    // LDS.128, STS.128 -> Return 2
    auto bitsPerThread = ll.getInDimSize(S("vector")) * bitwidth;
    assert(bitsPerThread >= 32 && "each thread must write at least 32 bits");
    return llvm::Log2_32(bitsPerThread / 32);
  }

protected:
  MLIRContext ctx;
};

// computes (read, write) bank‐conflicts exactly as in ll.py

// ——— Tests ———

TEST_F(SwizzleTest, Test128x128Float8Transpose) {
  // 128x128 float8 matrix transpose
  LinearLayout matrix(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}}},
       {S("lane"), {{0, 16}, {0, 32}, {0, 64}, {4, 0}, {8, 0}}},
       {S("warp"), {{16, 0}, {32, 0}, {64, 0}}}},
      {{S("dim0"), 128}, {S("dim1"), 128}}, /*requireSurjective=*/true);
  auto matrix_t = transposeLinearLayout(matrix, {1, 0});

  auto smem = optimalSwizzling(matrix, matrix_t, /*bitwidth=*/8);
  auto [r, w] = logBankConflicts(matrix, matrix_t, smem);
  auto logWf = logWavefronts(smem, 8);
  EXPECT_EQ(r, logWf);
  EXPECT_EQ(w, logWf);
}

TEST_F(SwizzleTest, Test16x16Bf16BlockedMma) {
  // 16×16 bf16 MMA
  LinearLayout blocked({{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                        {S("lane"), {{0, 8}, {1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                        {S("warp"), {}}},
                       {{S("dim0"), 16}, {S("dim1"), 16}},
                       /*requireSurjective=*/true);
  LinearLayout mma({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}}},
                   {{S("dim0"), 16}, {S("dim1"), 16}},
                   /*requireSurjective=*/true);

  auto smem = optimalSwizzling(blocked, mma, /*bitwidth=*/16);
  auto [r, w] = logBankConflicts(blocked, mma, smem);
  auto logWf = logWavefronts(smem, 16);
  EXPECT_EQ(r, logWf);
  EXPECT_EQ(w, logWf);
}

TEST_F(SwizzleTest, Test16x256U4Mma) {
  // 16×256 u4 MMA
  LinearLayout blocked(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}, {8, 0}}},
       {S("lane"), {{0, 32}, {0, 64}, {0, 128}, {1, 0}, {2, 0}}},
       {S("warp"), {}}},
      {{S("dim0"), 16}, {S("dim1"), 256}}, /*requireSurjective=*/true);
  LinearLayout mma(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}, {0, 64}, {0, 128}}},
       {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {}}},
      {{S("dim0"), 16}, {S("dim1"), 256}}, /*requireSurjective=*/true);

  auto smem = optimalSwizzling(blocked, mma, /*bitwidth=*/4);
  auto [r, w] = logBankConflicts(blocked, mma, smem);
  auto logWf = logWavefronts(smem, 4);
  EXPECT_EQ(r, logWf);
  EXPECT_EQ(w, logWf);
}

TEST_F(SwizzleTest, Test32x16F32Transpose) {
  // 32×16 f32 transpose
  LinearLayout matrix({{S("register"), {{4, 0}, {8, 0}, {16, 0}}},
                       {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                       {S("warp"), {{2, 0}}}},
                      {{S("dim0"), 32}, {S("dim1"), 16}},
                      /*requireSurjective=*/true);
  LinearLayout matrix_t({{S("register"), {{0, 2}, {0, 4}, {0, 8}}},
                         {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                         {S("warp"), {{0, 1}}}},
                        {{S("dim0"), 32}, {S("dim1"), 16}},
                        /*requireSurjective=*/true);
  auto smem = optimalSwizzling(matrix, matrix_t, /*bitwidth=*/32);
  auto [r, w] = logBankConflicts(matrix, matrix_t, smem);
  auto logWf = logWavefronts(smem, 32);
  EXPECT_EQ(r, logWf);
  EXPECT_EQ(w, logWf);
}
} // namespace

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
