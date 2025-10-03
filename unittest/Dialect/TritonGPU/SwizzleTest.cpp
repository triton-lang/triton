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

  using LinearLayout = mlir::triton::LinearLayout;

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

    auto cta = mlir::triton::gpu::CTALayoutAttr::get(
        &ctx, cpgStorage.empty() ? cpg : ArrayRef<unsigned>(cpgStorage),
        splitStorage.empty() ? split : ArrayRef<unsigned>(splitStorage),
        cOrderStorage.empty() ? cOrder : ArrayRef<unsigned>(cOrderStorage));
    return mlir::triton::gpu::BlockedEncodingAttr::get(&ctx, spt, tpw, wpcta,
                                                       order, cta);
  }

  mlir::triton::gpu::NvidiaMmaEncodingAttr mma(ArrayRef<unsigned> version,
                                               ArrayRef<unsigned> warpsPerCTA,
                                               ArrayRef<unsigned> instrShape) {
    auto cta =
        mlir::triton::gpu::CTALayoutAttr::getDefault(&ctx, warpsPerCTA.size());
    return mlir::triton::gpu::NvidiaMmaEncodingAttr::get(
        &ctx, version[0], version[1], warpsPerCTA, cta, instrShape);
  }

  mlir::triton::gpu::NVMMASharedEncodingAttr
  nvmmaShared(unsigned swizzle, unsigned bitwidth, unsigned rank,
              bool transposed = false) {
    SmallVector<unsigned> cpg(rank, 1), split(rank, 1), order(rank);
    std::iota(order.begin(), order.end(), 0);
    auto cta = mlir::triton::gpu::CTALayoutAttr::get(&ctx, cpg, split, order);
    return mlir::triton::gpu::NVMMASharedEncodingAttr::get(
        &ctx, swizzle, transposed, bitwidth,
        /*fp4Padded=*/false, cta);
  }

  LinearLayout toLL(ArrayRef<int64_t> shape, Attribute attr) {
    return mlir::triton::gpu::toLinearLayout(shape, attr);
  }

  int computeConflicts(ArrayRef<int64_t> shape, Attribute regAttr,
                       Attribute sharedAttr, int bitwidth) {
    auto regLL = toLL(shape, regAttr);
    auto sharedLL = toLL(shape, sharedAttr);
    return mlir::triton::gpu::bankConflictsMemDesc(regLL, sharedLL, bitwidth);
  }

  int bruteforceBankConflictsPerWavefront(ArrayRef<int64_t> shape,
                                          Attribute regAttr,
                                          Attribute sharedAttr, int bitwidth) {
    // Compute the bank conflicts per wavefront
    // In other words, we compute how many extra memory accesses (bank
    // conflicts) are needed for a given wavefront.
    auto regLL = toLL(shape, regAttr);
    auto sharedLL = toLL(shape, sharedAttr);

    auto *ctx = sharedLL.getInDimNames().begin()->getContext();
    auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };

    auto kOffset = S("offset");
    auto kReg = S("register");
    auto kLane = S("lane");
    auto kWarp = S("warp");
    auto regToShared = regLL.invertAndCompose(sharedLL);
    assert(regToShared.isTrivialOver({S("block")}) && "NYI");
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
};

// ——— Tests ———

TEST_F(SwizzleTest, Test128x128Float8Transpose) {
  // 128x128 float8 matrix transpose
  LinearLayout matrix(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}}},
       {S("lane"), {{0, 16}, {0, 32}, {0, 64}, {4, 0}, {8, 0}}},
       {S("warp"), {{16, 0}, {32, 0}, {64, 0}}}},
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
                        {S("warp"), {}}},
                       {{S("dim0"), 16}, {S("dim1"), 16}},
                       /*requireSurjective=*/true);
  LinearLayout mma({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}}},
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
       {S("warp"), {}}},
      {{S("dim0"), 16}, {S("dim1"), 256}}, /*requireSurjective=*/true);
  LinearLayout mma(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}, {0, 64}, {0, 128}}},
       {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {}}},
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
                       {S("warp"), {{2, 0}}}},
                      {{S("dim0"), 32}, {S("dim1"), 16}},
                      /*requireSurjective=*/true);
  LinearLayout matrix_t({{S("register"), {{0, 2}, {0, 4}, {0, 8}}},
                         {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                         {S("warp"), {{0, 1}}}},
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
       {S("warp"), {{0, 2}, {0, 4}, {0, 8}, {0, 16}}}},
      {{S("dim0"), 128}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  LinearLayout matrix_t(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {32, 0}, {64, 0}}},
       {S("lane"), {{0, 8}, {0, 16}, {0, 32}, {0, 64}, {1, 0}}},
       {S("warp"), {{2, 0}, {4, 0}, {8, 0}, {16, 0}}}},
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
           mlir::triton::gpu::CTALayoutAttr::getDefault(&ctx, 1)),
       {32},
       32},
      {blocked({1}, {32}, {4}, {0}),
       mlir::triton::gpu::SwizzledSharedEncodingAttr::get(
           &ctx, 1, 1, 1, {0},
           mlir::triton::gpu::CTALayoutAttr::getDefault(&ctx, 1)),
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
    EXPECT_EQ(computeConflicts(c.shape, c.reg, c.shared, c.bitwidth),
              bruteforceBankConflictsPerWavefront(c.shape, c.reg, c.shared,
                                                  c.bitwidth))

        << toLL(c.shape, c.reg).invertAndCompose(toLL(c.shape, c.shared))
        << "\nbitwidth=" << c.bitwidth << "\n"
        << attrStr(c.reg) << "\n"
        << attrStr(c.shared);
  }
}

} // namespace

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
