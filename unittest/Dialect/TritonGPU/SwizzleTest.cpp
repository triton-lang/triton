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
using mlir::triton::gpu::LocalMemOpTile;
using mlir::triton::gpu::optimalSwizzling;
using mlir::triton::gpu::optimalSwizzlingLdSt;

namespace {

static std::string attrStr(Attribute a) {
  std::string s;
  llvm::raw_string_ostream os(s);
  a.print(os);
  return s;
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

  mlir::triton::gpu::AMDMfmaEncodingAttr
  mfma(unsigned version, ArrayRef<unsigned> warpsPerCTA,
       ArrayRef<unsigned> instrShape, bool isTransposed,
       ArrayRef<unsigned> tilesPerWarp = {}, unsigned bitWidth = 0) {
    auto cta = mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(
        &ctx, warpsPerCTA.size());
    return mlir::triton::gpu::AMDMfmaEncodingAttr::get(
        &ctx, version, warpsPerCTA, instrShape, isTransposed, cta, tilesPerWarp,
        bitWidth);
  }

  mlir::triton::gpu::AMDRotatingSharedEncodingAttr
  AMDRotatingShared(unsigned vec, unsigned perPhase, unsigned maxPhase,
                    ArrayRef<unsigned> order) {
    auto cta =
        mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, order.size());
    return mlir::triton::gpu::AMDRotatingSharedEncodingAttr::get(
        &ctx, vec, perPhase, maxPhase, order, cta);
  }

  LinearLayout toLL(ArrayRef<int64_t> shape, Attribute attr) {
    return mlir::triton::gpu::toLinearLayout(shape, attr);
  }

  int computeConflicts(ArrayRef<int64_t> shape, Attribute regAttr,
                       Attribute sharedAttr, int bitwidth, int numBanks = 32,
                       LocalMemOpTile laneTile = {}) {
    auto regLL = toLL(shape, regAttr);
    auto sharedLL = toLL(shape, sharedAttr);
    return mlir::triton::gpu::bankConflictsMemDesc(regLL, sharedLL, bitwidth,
                                                   numBanks, laneTile);
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

  int bruteforceBankConflictsPerWavefront64(ArrayRef<int64_t> shape,
                                            Attribute regAttr,
                                            Attribute sharedAttr, int bitwidth,
                                            int numBanks,
                                            LocalMemOpTile laneTile) {
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
    int threadsPerPhase = numBanks / vectorisation;
    int numLanes = regToShared.getInDimSize(kLane);
    int numPhases = numLanes / threadsPerPhase;
    int wavefronts = 0;
    // For all the emitted instructions
    for (int regIdx = 0; regIdx < regToShared.getInDimSize(kReg);
         regIdx += elemsPerVec) {
      for (int warpIdx = 0; warpIdx < regToShared.getInDimSize(kWarp);
           warpIdx++) {
        // For each instruction
        llvm::SmallVector<llvm::SmallSet<int, 32>, 8> uniqueOffsetsByPhase(
            numPhases);
        for (int laneIdx = 0; laneIdx < numLanes; laneIdx++) {
          int phaseIdx;
          int maskedLaneIdx;
          if (vectorisation == 4) {
            int t2 = (laneIdx >> 2) & 1;
            int t3 = (laneIdx >> 3) & 1;
            int t4 = (laneIdx >> 4) & 1;
            int t5 = (laneIdx >> 5) & 1;
            if (numBanks == 32) {
              // Group threads into 8 phases of 8 threads each.
              // Logic: 'half' (bit 5) selects even/odd phase indices.
              // Bit 3 (b3) and the XOR of bits 2 and 4 (b2^b4) determine the
              // intra-half grouping.
              // Phase 0: T0-3,   T20-23 | Phase 1: T32-35, T52-55
              // Phase 2: T4-7,   T16-19 | Phase 3: T36-39, T48-51
              // Phase 4: T8-11,  T28-31 | Phase 5: T40-43, T60-63
              // Phase 6: T12-15, T24-27 | Phase 7: T44-47, T56-59
              // phaseIdx is a linear layout on the laneIdx:
              // phase[0] = t[5]
              // phase[1] = t[2] ^ t[4]
              // phase[2] = t[3]
              phaseIdx = (t5 << 0) | ((t2 ^ t4) << 1) | (t3 << 2);
            } else {
              // Group threads into 4 phases of 16 threads each.
              // Logic: 'half' (bit 5) selects Phase 0/1 or 2/3.
              // Parity of bits 2, 3, and 4 (b2^b3^b4) determines the intra-half
              // grouping.
              // Phase 0: T0-3,   T12-15, T20-23, T24-27
              // Phase 1: T32-35, T44-47, T52-55, T56-59
              // Phase 2: T4-7,   T8-11,  T16-19, T28-31
              // Phase 3: T36-39, T40-43, T48-51, T60-63
              // phaseIdx is a linear layout on the laneIdx:
              // phase[0] = t[5]
              // phase[1] = t[2] ^ t[3] ^ t[4]
              phaseIdx = (t5 << 0) | ((t2 ^ t3 ^ t4) << 1);
            }
            maskedLaneIdx = 0;
            for (int shift : laneTile.laneAddr)
              maskedLaneIdx |= laneIdx & (1 << shift);
          } else {
            phaseIdx = laneIdx / threadsPerPhase;
            maskedLaneIdx = laneIdx % threadsPerPhase;
          }
          for (int vecIdx = 0; vecIdx < elemsPerVec; vecIdx++) {
            auto offset = regToShared
                              .apply({{kReg, regIdx + vecIdx},
                                      {kLane, maskedLaneIdx},
                                      {kWarp, warpIdx}})[0]
                              .second;
            auto offsetB32 = offset * bitwidth / 32;
            uniqueOffsetsByPhase[phaseIdx].insert(offsetB32);
          }
        }
        for (auto &uniqueOffsets : uniqueOffsetsByPhase) {
          llvm::SmallVector<int, 64> banks(numBanks, 0);
          for (int offset : uniqueOffsets) {
            banks[offset % numBanks]++;
          }
          wavefronts += *llvm::max_element(banks);
        }
      }
    }
    auto minWavefronts = (regToShared.getInDimSize(kReg) / elemsPerVec) *
                         regToShared.getInDimSize(kWarp) * numPhases;
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

TEST_F(SwizzleTest, Test64x128F16BlockedLinear32Bank) {
  LinearLayout src(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {16, 0}, {32, 0}}},
       {S("lane"), {{0, 8}, {0, 16}, {0, 32}, {0, 64}, {1, 0}, {2, 0}}},
       {S("warp"), {{4, 0}, {8, 0}}},
       {S("block"), {}}},
      {{S("dim0"), 64}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  LinearLayout dst(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 8}, {0, 16}, {0, 32}, {0, 64}, {32, 0}}},
       {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
       {S("warp"), {{0, 0}, {0, 0}}},
       {S("block"), {}}},
      {{S("dim0"), 64}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  auto smem = optimalSwizzlingLdSt(src, dst, /*bitwidth=*/16, /*numBanks*/ 32,
                                   /*srcTile*/ {},
                                   /*dstTile*/ {{}, {0, 1, 4}});
  auto [r, w] = bankConflictsLdSt(src, dst, smem, /*bitwidth=*/16,
                                  /*numBanks*/ 32, /*srcTile*/ {},
                                  /*dstTile*/ {{}, {0, 1, 4}});
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test64x128F16BlockedMfma64Bank) {
  LinearLayout src(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {16, 0}, {32, 0}}},
       {S("lane"), {{0, 8}, {0, 16}, {0, 32}, {0, 64}, {1, 0}, {2, 0}}},
       {S("warp"), {{4, 0}, {8, 0}}},
       {S("block"), {}}},
      {{S("dim0"), 64}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  LinearLayout dst(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 4}, {0, 16}, {0, 32}, {0, 64}, {32, 0}}},
       {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 8}}},
       {S("warp"), {{0, 0}, {0, 0}}},
       {S("block"), {}}},
      {{S("dim0"), 64}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  auto smem = optimalSwizzlingLdSt(src, dst, /*bitwidth=*/16,
                                   /*numBanks*/ 64, /*srcTile*/ {},
                                   /*dstTile*/ {{}, {0, 1, 3, 4}});
  auto [r, w] = bankConflictsLdSt(src, dst, smem, /*bitwidth=*/16,
                                  /*numBanks*/ 64, /*srcTile*/ {},
                                  /*dstTile*/ {{}, {0, 1, 3, 4}});
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
    EXPECT_EQ(computeConflicts(c.shape, c.reg, c.shared, c.bitwidth),
              bruteforceBankConflictsPerWavefront(c.shape, c.reg, c.shared,
                                                  c.bitwidth))

        << toLL(c.shape, c.reg).invertAndCompose(toLL(c.shape, c.shared))
        << "\nbitwidth=" << c.bitwidth << "\n"
        << attrStr(c.reg) << "\n"
        << attrStr(c.shared);
  }
}

TEST_F(BankConflictTest, bankConflictsWavefront64) {
  using mlir::triton::gpu::DotOperandEncodingAttr;

  auto mmaV3 = mfma(3, {4, 1}, {32, 32, 8}, true);
  auto mmaV4 = mfma(4, {4, 1}, {32, 32, 16}, true);

  auto dotAV3 =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, mmaV3, /*kWidth=*/4);
  auto dotAV4 =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, mmaV4, /*kWidth=*/8);
  auto dotBV3 =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, mmaV3, /*kWidth=*/4);
  auto dotBV4 =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, mmaV4, /*kWidth=*/4);

  struct Case {
    Attribute reg;
    Attribute shared;
    SmallVector<int64_t, 3> shape;
    int bitwidth;
    int numBanks;
    LocalMemOpTile laneTile;
  };

  SmallVector<Case, 6> cases = {
      {blocked({1, 8}, {4, 16}, {4, 1}, {1, 0}),
       mlir::triton::gpu::SwizzledSharedEncodingAttr::get(
           &ctx, 8, 1, 16, {1, 0},
           mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, 2)),
       {128, 128},
       16,
       32,
       /*vec=4*/ {{}, {0, 1, 4}}},
      {blocked({1, 8}, {4, 16}, {4, 1}, {1, 0}),
       mlir::triton::gpu::SwizzledSharedEncodingAttr::get(
           &ctx, 8, 1, 16, {1, 0},
           mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, 2)),
       {128, 128},
       16,
       64,
       /*vec=4*/ {{}, {0, 1, 3, 4}}},
      {dotAV3,
       mlir::triton::gpu::SwizzledSharedEncodingAttr::get(
           &ctx, 4, 1, 16, {1, 0},
           mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, 2)),
       {128, 128},
       16,
       32,
       /*vec=2*/ {{}, {}}},
      {dotAV4,
       mlir::triton::gpu::SwizzledSharedEncodingAttr::get(
           &ctx, 8, 1, 16, {1, 0},
           mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, 2)),
       {128, 128},
       16,
       64,
       /*vec=4*/ {{}, {0, 1, 3, 4}}},
      {dotBV3,
       AMDRotatingShared(/*vec=*/4, /*perPhase=*/1, /*maxPhase=*/16,
                         /*order=*/{0, 1}),
       {64, 128},
       16,
       32,
       /*vec=2*/ {{}, {}}},
      {dotBV4,
       AMDRotatingShared(/*vec=*/4, /*perPhase=*/2, /*maxPhase=*/8,
                         /*order=*/{0, 1}),
       {64, 128},
       16,
       64,
       /*vec=2*/ {{}, {}}},
  };

  for (const auto &c : cases) {
    EXPECT_EQ(computeConflicts(c.shape, c.reg, c.shared, c.bitwidth, c.numBanks,
                               c.laneTile),
              bruteforceBankConflictsPerWavefront64(
                  c.shape, c.reg, c.shared, c.bitwidth, c.numBanks, c.laneTile))
        << toLL(c.shape, c.reg).invertAndCompose(toLL(c.shape, c.shared))
        << "\nbitwidth=" << c.bitwidth << "\n"
        << "numBanks=" << c.numBanks << "\n"
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
