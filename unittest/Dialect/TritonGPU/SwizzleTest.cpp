#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

#include "mlir/Support/LLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <llvm/ADT/SmallSet.h>
#include <random>

using namespace mlir;
using namespace mlir::triton;

using mlir::triton::gpu::bankConflictsLdSt;
using mlir::triton::gpu::optimalSwizzling;
using mlir::triton::gpu::optimalSwizzlingLdSt;

namespace {

static std::string attrStr(Attribute a) {
  std::string s;
  llvm::raw_string_ostream os(s);
  a.print(os);
  return s;
}

template <typename IntTy>
static IntTy sampleInt(std::mt19937 &rng, IntTy lo, IntTy hi) {
  std::uniform_int_distribution<IntTy> dist(lo, hi);
  return dist(rng);
}

static SmallVector<uint32_t> randomColumns(std::mt19937 &rng, unsigned k,
                                           unsigned n) {
  assert(k > 0 && "Expected non-empty output space");
  assert(n >= k && "Surjective map needs enough input bits");

  SmallVector<uint32_t> cols;
  cols.reserve(n);
  for (int i = 0; i < k; i++) {
    cols.push_back(1u << i);
  }

  for (int iter = 0; iter < 8 * k; iter++) {
    unsigned i = sampleInt(rng, 0u, k - 1);
    unsigned j = sampleInt(rng, 0u, k - 1);
    if (i == j) {
      continue;
    }
    int op = sampleInt(rng, 0, 1);
    if (op == 0) {
      std::swap(cols[i], cols[j]);
    } else {
      cols[j] ^= cols[i];
    }
  }

  for (int i = k; i < n; i++) {
    unsigned col = sampleInt(rng, 0u, (1u << k) - 1);
    cols.push_back(col);
  }
  return cols;
}

static std::vector<int32_t> columnToBasis(uint32_t value,
                                          ArrayRef<unsigned> outDimSizesLog2) {
  std::vector<int32_t> basis;
  basis.reserve(outDimSizesLog2.size());
  int shift = 0;
  for (auto bits : outDimSizesLog2) {
    int32_t out = 0;
    if (bits > 0) {
      out = static_cast<int32_t>((value >> shift) & ((1u << bits) - 1u));
    }
    basis.push_back(out);
    shift += bits;
  }
  return basis;
}

static LinearLayout
buildRandomLayout(std::mt19937 &rng,
                  ArrayRef<std::pair<StringAttr, unsigned>> inDimsBits,
                  ArrayRef<unsigned> outDimSizesLog2, int totalOutSizeBits,
                  const SmallVector<std::pair<StringAttr, int32_t>> &outDims) {
  int n = 0;
  for (const auto &[_, bits] : inDimsBits) {
    n += bits;
  }

  auto columns = randomColumns(rng, totalOutSizeBits, n);
  int colIdx = 0;
  std::vector<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> inBases;
  inBases.reserve(inDimsBits.size());
  for (const auto &[dim, bits] : inDimsBits) {
    std::vector<std::vector<int32_t>> bases;
    bases.reserve(bits);
    for (int i = 0; i < bits; i++) {
      bases.push_back(columnToBasis(columns[colIdx++], outDimSizesLog2));
    }
    inBases.push_back({dim, std::move(bases)});
  }
  return LinearLayout(inBases, outDims, /*requireSurjective=*/true);
}

class SwizzleTest : public ::testing::Test {
public:
  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

class BankConflictTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<mlir::triton::TritonDialect,
                    mlir::triton::gpu::TritonGPUDialect>();
  }

  MLIRContext ctx;

  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

  mlir::triton::gpu::BlockedEncodingAttr
  blocked(ArrayRef<unsigned> spt, ArrayRef<unsigned> tpw,
          ArrayRef<unsigned> wpcta, ArrayRef<unsigned> order,
          ArrayRef<unsigned> cpg = {}, ArrayRef<unsigned> split = {},
          ArrayRef<unsigned> cOrder = {}) {
    SmallVector<unsigned> cpgStorage;
    SmallVector<unsigned> splitStorage;
    SmallVector<unsigned> cOrderStorage;
    if (cpg.empty())
      cpgStorage.assign(spt.size(), 1);
    if (split.empty())
      splitStorage.assign(spt.size(), 1);
    if (cOrder.empty())
      cOrderStorage.assign(order.begin(), order.end());

    auto cta = mlir::triton::gpu::CGAEncodingAttr::fromSplitParams(
        &ctx, cpgStorage.empty() ? cpg : ArrayRef<unsigned>(cpgStorage),
        splitStorage.empty() ? split : ArrayRef<unsigned>(splitStorage),
        cOrderStorage.empty() ? cOrder : ArrayRef<unsigned>(cOrderStorage));
    return mlir::triton::gpu::BlockedEncodingAttr::get(&ctx, spt, tpw, wpcta,
                                                       order, cta);
  }

  mlir::triton::gpu::NvidiaMmaEncodingAttr mma(ArrayRef<unsigned> version,
                                               ArrayRef<unsigned> warpsPerCTA,
                                               ArrayRef<unsigned> instrShape) {
    auto cta = mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(
        &ctx, warpsPerCTA.size());
    return mlir::triton::gpu::NvidiaMmaEncodingAttr::get(
        &ctx, version[0], version[1], warpsPerCTA, cta, instrShape);
  }

  mlir::triton::gpu::NVMMASharedEncodingAttr
  nvmmaShared(unsigned swizzle, unsigned bitwidth, unsigned rank,
              bool transposed = false) {
    SmallVector<unsigned> cpg(rank, 1), split(rank, 1), order(rank);
    std::iota(order.begin(), order.end(), 0);
    auto cta = mlir::triton::gpu::CGAEncodingAttr::fromSplitParams(
        &ctx, cpg, split, order);
    return mlir::triton::gpu::NVMMASharedEncodingAttr::get(
        &ctx, swizzle, transposed, bitwidth,
        /*fp4Padded=*/false, cta);
  }

  LinearLayout toLL(ArrayRef<int64_t> shape, Attribute attr) {
    return mlir::triton::gpu::toLinearLayout(shape, attr);
  }

  int computeConflicts(const LinearLayout &regLL, const LinearLayout &sharedLL,
                       int bitwidth) {
    return mlir::triton::gpu::bankConflictsMemDesc(regLL, sharedLL, bitwidth);
  }

  int bruteforceBankConflictsPerWavefront(const LinearLayout &regLL,
                                          const LinearLayout &sharedLL,
                                          int bitwidth) {
    // Compute the bank conflicts per wavefront
    // In other words, we compute how many extra memory accesses (bank
    // conflicts) are needed for a given wavefront.
    auto *ctx = sharedLL.getInDimNames().begin()->getContext();
    auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };

    auto kOffset = S("offset");
    auto kReg = S("register");
    auto kLane = S("lane");
    auto kWarp = S("warp");
    auto regToShared = regLL.invertAndCompose(sharedLL);
    auto kBlock = S("block");
    assert(regToShared.isTrivialOver({kBlock}) && "NYI");
    regToShared = regToShared.sublayout({kReg, kLane, kWarp}, {kOffset});

    // Remove broadcasting
    regToShared = actionRemoveBroadcastedRegs(regToShared).apply(regToShared);
    auto [elemsPerVec, permutation] =
        largestVectorisation(ctx, regToShared, bitwidth);
    regToShared = permutation.apply(regToShared);

    int vectorisation = llvm::divideCeil(bitwidth * elemsPerVec, 32);
    assert(vectorisation == 1 || vectorisation == 2 || vectorisation == 4);
    int wavefronts = 0;
    // For all the emitted instructions
    for (int regIdx = 0; regIdx < regToShared.getInDimSize(kReg);
         regIdx += elemsPerVec) {
      for (int warpIdx = 0; warpIdx < regToShared.getInDimSize(kWarp);
           warpIdx++) {
        // For each instruction
        for (int laneIdx = 0; laneIdx < regToShared.getInDimSize(kLane);
             laneIdx += (32 / vectorisation)) {
          // For each wavefront
          llvm::SmallSet<int, 32> uniqueOffsets;
          for (int laneIdx = 0; laneIdx < 32 / vectorisation; laneIdx++) {
            for (int vecIdx = 0; vecIdx < elemsPerVec; vecIdx++) {
              auto offset = regToShared
                                .apply({{kReg, regIdx + vecIdx},
                                        {kLane, laneIdx},
                                        {kWarp, warpIdx}})[0]
                                .second;
              auto offsetB32 = offset * bitwidth / 32;
              uniqueOffsets.insert(offsetB32);
            }
          }
          llvm::SmallVector<int, 32> banks(32, 0);
          for (int offset : uniqueOffsets) {
            banks[offset % 32]++;
          }
          wavefronts += *llvm::max_element(banks);
        }
      }
    }
    auto minWavefronts =
        (regToShared.getInDimSize(kReg) / elemsPerVec) *
        regToShared.getInDimSize(kWarp) *
        (regToShared.getInDimSize(kLane) / (32 / vectorisation));
    // Assert homogeneity
    assert(wavefronts % minWavefronts == 0);
    return wavefronts / minWavefronts - 1;
  }

  LinearLayout
  makeSharedLL(std::mt19937 &rng, ArrayRef<unsigned> outDimSizesLog2,
               int totalOutSizeBits,
               const SmallVector<std::pair<StringAttr, int32_t>> &outDims) {
    // Do not allow broadcast for shared memory layout
    int sharedInBits = totalOutSizeBits;
    return buildRandomLayout(rng,
                             {{S("offset"), sharedInBits}, {S("block"), 0}},
                             outDimSizesLog2, totalOutSizeBits, outDims);
  }

  LinearLayout
  makeRegLL(std::mt19937 &rng, ArrayRef<unsigned> outDimSizesLog2,
            int totalOutSizeBits,
            const SmallVector<std::pair<StringAttr, int32_t>> &outDims) {
    int laneBits = 5;
    int minExtraBits = std::max(0, totalOutSizeBits - laneBits);
    int regBits = sampleInt(rng, 0, minExtraBits);
    int warpBits = minExtraBits - regBits;
    // Add broadcast bases
    int slack = sampleInt(rng, 0, 2);
    for (int i = 0; i < slack; i++) {
      if (sampleInt(rng, 0, 1) == 0) {
        regBits++;
      } else {
        warpBits++;
      }
    }

    return buildRandomLayout(rng,
                             {{S("register"), regBits},
                              {S("lane"), laneBits},
                              {S("warp"), warpBits},
                              {S("block"), 0}},
                             outDimSizesLog2, totalOutSizeBits, outDims);
  }

  struct FuzzCase {
    LinearLayout regLL;
    LinearLayout sharedLL;
    SmallVector<std::pair<StringAttr, int32_t>> outDims;
    int bitwidth;
  };

  FuzzCase generateFuzzCase(std::mt19937 &rng) {
    constexpr static int kBitwidthOptions[] = {8, 16, 32};
    int bitwidth = kBitwidthOptions[sampleInt(rng, 0, 2)];

    int rank = sampleInt(rng, 1, 4);
    SmallVector<unsigned, 8> outDimSizesLog2(rank);
    for (int i = 0; i < rank; ++i) {
      outDimSizesLog2[i] = sampleInt(rng, 0, 5);
    }
    int mustBeNonZero = sampleInt(rng, 0, rank - 1);
    outDimSizesLog2[mustBeNonZero] = sampleInt(rng, 1, 5);
    int totalOutSizeBits =
        std::accumulate(outDimSizesLog2.begin(), outDimSizesLog2.end(), 0);

    SmallVector<std::pair<StringAttr, int32_t>> outDims;
    outDims.reserve(outDimSizesLog2.size());
    for (int i = 0; i < outDimSizesLog2.size(); i++) {
      outDims.push_back(
          {S(("dim" + std::to_string(i)).c_str()), 1 << outDimSizesLog2[i]});
    }

    auto regLL = makeRegLL(rng, outDimSizesLog2, totalOutSizeBits, outDims);
    auto sharedLL =
        makeSharedLL(rng, outDimSizesLog2, totalOutSizeBits, outDims);
    return {std::move(regLL), std::move(sharedLL), std::move(outDims),
            bitwidth};
  }
};

// ——— Tests ———

TEST_F(SwizzleTest, Test128x128Float8Transpose) {
  // 128x128 float8 matrix transpose
  LinearLayout matrix(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}}},
       {S("lane"), {{0, 16}, {0, 32}, {0, 64}, {4, 0}, {8, 0}}},
       {S("warp"), {{16, 0}, {32, 0}, {64, 0}}},
       {S("block"), {}}},
      {{S("dim0"), 128}, {S("dim1"), 128}}, /*requireSurjective=*/true);
  auto matrix_t = transposeLinearLayout(matrix, {1, 0});

  auto smem = optimalSwizzlingLdSt(matrix, matrix_t, /*bitwidth=*/8);
  auto [r, w] = bankConflictsLdSt(matrix, matrix_t, smem, /*bitwidth=*/8);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test16x16Bf16BlockedMma) {
  // 16×16 bf16 MMA
  LinearLayout blocked({{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                        {S("lane"), {{0, 8}, {1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                        {S("warp"), {}},
                        {S("block"), {}}},
                       {{S("dim0"), 16}, {S("dim1"), 16}},
                       /*requireSurjective=*/true);
  LinearLayout mma({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}}},
                   {{S("dim0"), 16}, {S("dim1"), 16}},
                   /*requireSurjective=*/true);

  auto smem = optimalSwizzlingLdSt(blocked, mma, /*bitwidth=*/16);
  auto [r, w] = bankConflictsLdSt(blocked, mma, smem, /*bitwidth=*/16);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test16x256U4Mma) {
  // 16×256 u4 MMA
  LinearLayout blocked(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}, {8, 0}}},
       {S("lane"), {{0, 32}, {0, 64}, {0, 128}, {1, 0}, {2, 0}}},
       {S("warp"), {}},
       {S("block"), {}}},
      {{S("dim0"), 16}, {S("dim1"), 256}}, /*requireSurjective=*/true);
  LinearLayout mma(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}, {0, 64}, {0, 128}}},
       {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {}},
       {S("block"), {}}},
      {{S("dim0"), 16}, {S("dim1"), 256}}, /*requireSurjective=*/true);

  auto smem = optimalSwizzlingLdSt(blocked, mma, /*bitwidth=*/4);
  auto [r, w] = bankConflictsLdSt(blocked, mma, smem, /*bitwidth=*/4);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test32x16F32Transpose) {
  // 32×16 f32 transpose
  LinearLayout matrix({{S("register"), {{4, 0}, {8, 0}, {16, 0}}},
                       {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                       {S("warp"), {{2, 0}}},
                       {S("block"), {}}},
                      {{S("dim0"), 32}, {S("dim1"), 16}},
                      /*requireSurjective=*/true);
  LinearLayout matrix_t({{S("register"), {{0, 2}, {0, 4}, {0, 8}}},
                         {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                         {S("warp"), {{0, 1}}},
                         {S("block"), {}}},
                        {{S("dim0"), 32}, {S("dim1"), 16}},
                        /*requireSurjective=*/true);
  auto smem = optimalSwizzlingLdSt(matrix, matrix_t, /*bitwidth=*/32);
  auto [r, w] = bankConflictsLdSt(matrix, matrix_t, smem, /*bitwidth=*/32);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test128x128F16Transpose) {
  LinearLayout matrix(
      {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {0, 32}, {0, 64}}},
       {S("lane"), {{8, 0}, {16, 0}, {32, 0}, {64, 0}, {0, 1}}},
       {S("warp"), {{0, 2}, {0, 4}, {0, 8}, {0, 16}}},
       {S("block"), {}}},
      {{S("dim0"), 128}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  LinearLayout matrix_t(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {32, 0}, {64, 0}}},
       {S("lane"), {{0, 8}, {0, 16}, {0, 32}, {0, 64}, {1, 0}}},
       {S("warp"), {{2, 0}, {4, 0}, {8, 0}, {16, 0}}},
       {S("block"), {}}},
      {{S("dim0"), 128}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  auto smem = optimalSwizzlingLdSt(matrix, matrix_t, /*bitwidth=*/16);
  auto [r, w] = bankConflictsLdSt(matrix, matrix_t, smem, /*bitwidth=*/16);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(BankConflictTest, bankConflicts) {
  using mlir::triton::gpu::DotOperandEncodingAttr;

  auto mmaV3 = mma({3, 0}, {4, 1}, {16, 32, 16});
  auto mmaV2 = mma({2, 0}, {1, 4}, {16, 8});
  auto mmaV2Large = mma({2, 0}, {2, 4}, {16, 8});

  auto dotA =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, mmaV2, /*kWidth=*/2);
  auto dotB =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, mmaV2, /*kWidth=*/2);
  auto dotBInt8 =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, mmaV2, /*kWidth=*/1);
  auto dotBNoswizzle =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, mmaV2Large, /*kWidth=*/2);

  struct Case {
    Attribute reg;
    Attribute shared;
    SmallVector<int64_t, 3> shape;
    int bitwidth;
  };

  SmallVector<Case, 11> cases = {
      {blocked({1}, {32}, {4}, {0}),
       mlir::triton::gpu::SwizzledSharedEncodingAttr::get(
           &ctx, 1, 1, 1, {0},
           mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, 1)),
       {32},
       32},
      {blocked({1}, {32}, {4}, {0}),
       mlir::triton::gpu::SwizzledSharedEncodingAttr::get(
           &ctx, 1, 1, 1, {0},
           mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, 1)),
       {32},
       16},
      {mmaV3,
       nvmmaShared(/*swizzle=*/128, /*bitwidth=*/16, /*rank=*/2),
       {128, 128},
       16},
      {dotB,
       nvmmaShared(/*swizzle=*/64, /*bitwidth=*/16, /*rank=*/2),
       {64, 32},
       16},
      {dotA,
       nvmmaShared(/*swizzle=*/64, /*bitwidth=*/16, /*rank=*/2,
                   /*transposed=*/true),
       {32, 64},
       16},
      {dotBInt8,
       nvmmaShared(/*swizzle=*/32, /*bitwidth=*/8, /*rank=*/2),
       {8, 32},
       8},
      {mma({2, 0}, {4, 1}, {16, 8}),
       nvmmaShared(/*swizzle=*/64, /*bitwidth=*/16, /*rank=*/2,
                   /*transposed=*/true),
       {64, 64},
       16},
      {mma({3, 0}, {2, 2}, {16, 32, 16}),
       nvmmaShared(/*swizzle=*/64, /*bitwidth=*/16, /*rank=*/2),
       {64, 32},
       16},
      {mma({2, 0}, {4, 1}, {16, 8}),
       nvmmaShared(/*swizzle=*/32, /*bitwidth=*/8, /*rank=*/2),
       {32, 32},
       8},
      {dotBNoswizzle,
       nvmmaShared(/*swizzle=*/0, /*bitwidth=*/16, /*rank=*/2),
       {4, 64},
       16},
      {mma({3, 0}, {4, 1}, {16, 32, 16}),
       nvmmaShared(/*swizzle=*/128, /*bitwidth=*/32, /*rank=*/2),
       {128, 64},
       32},
  };

  for (const auto &c : cases) {
    auto regLL = toLL(c.shape, c.reg);
    auto sharedLL = toLL(c.shape, c.shared);
    EXPECT_EQ(computeConflicts(regLL, sharedLL, c.bitwidth),
              bruteforceBankConflictsPerWavefront(regLL, sharedLL, c.bitwidth))
        << regLL.invertAndCompose(sharedLL) << "\nbitwidth=" << c.bitwidth
        << "\n"
        << attrStr(c.reg) << "\n"
        << attrStr(c.shared);
  }
}

TEST_F(BankConflictTest, bankConflictsFuzz) {
  constexpr int numCases = 512;
  constexpr uint32_t seed = 0x5f3f8f83U;
  std::mt19937 rng(seed);

  for (int i = 0; i < numCases; i++) {
    auto c = generateFuzzCase(rng);
    ASSERT_TRUE(c.regLL.isSurjective());
    ASSERT_TRUE(c.sharedLL.isSurjective());

    int computed = computeConflicts(c.regLL, c.sharedLL, c.bitwidth);
    int brute =
        bruteforceBankConflictsPerWavefront(c.regLL, c.sharedLL, c.bitwidth);
    EXPECT_EQ(computed, brute) << "case_idx=" << i << "\n"
                               << c.regLL.invertAndCompose(c.sharedLL)
                               << "\nbitwidth=" << c.bitwidth << "\n"
                               << c.regLL << "\n"
                               << c.sharedLL;
  }
}

} // namespace

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
