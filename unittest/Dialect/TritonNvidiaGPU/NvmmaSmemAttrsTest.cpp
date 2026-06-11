#include "triton/Dialect/TritonNvidiaGPU/IR/NvmmaSmemAttrs.h"

#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include <gtest/gtest.h>

using namespace mlir::triton::nvidia_gpu;

namespace mlir::triton::gpu {
namespace {

LinearLayout getSwizzle0CoreMatrixLinearLayout(MLIRContext *ctx,
                                               ArrayRef<int64_t> shape,
                                               unsigned elementBitWidth,
                                               bool transposed,
                                               bool fp4Padded) {
  SmallVector<int64_t> tiledShape(shape);
  if (transposed)
    std::swap(tiledShape[0], tiledShape[1]);
  auto enc = NVMMASharedEncodingAttr::get(
      ctx, /*swizzlingByteWidth=*/0, /*transposed=*/false, elementBitWidth,
      fp4Padded, CGAEncodingAttr::get1CTALayout(ctx, /*rank=*/2));
  auto layout = ensureLayoutNotSmallerThan(
      getCoreMatrixLinearLayout(enc, /*disableSwizzle=*/false),
      standardOutDimNames(ctx, shape.size()), tiledShape);
  if (transposed)
    layout = transposeLinearLayout(layout, {1, 0});
  return layout;
}

class NvmmaSmemAttrsTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<TritonGPUDialect>();
    ctx.getOrLoadDialect<TritonNvidiaGPUDialect>();
  }

  NVMMASharedEncodingAttr
  nvmmaShared(unsigned swizzleSizeInBytes, bool transposed,
              unsigned elementBitWidth, ArrayRef<unsigned> cpg,
              ArrayRef<unsigned> cSplit, ArrayRef<unsigned> ord,
              ArrayRef<unsigned> cOrd, bool fp4Padded = false) {
    return NVMMASharedEncodingAttr::get(
        &ctx, swizzleSizeInBytes, transposed, elementBitWidth, fp4Padded,
        CGAEncodingAttr::fromSplitParams(&ctx, cpg, cSplit, cOrd));
  }

  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(NvmmaSmemAttrsTest, InferNvmmaSmemAttrsSharedLinear) {
  auto i8Ty = IntegerType::get(&ctx, 8);
  auto smem = SharedMemorySpaceAttr::get(&ctx);

  struct ExpectedLayout {
    unsigned swizzlingByteWidth = 0;
    bool transposed = false;
    bool fp4Padded = false;
  };

  auto checkLayout = [&](SmallVector<int64_t> shape, Type elemTy,
                         NVMMASharedEncodingAttr nvmma,
                         ExpectedLayout expected) {
    auto ll = toLinearLayout(shape, nvmma);
    auto sharedLinear = SharedLinearEncodingAttr::get(&ctx, std::move(ll),
                                                      /*layoutAlignment=*/1024);
    auto memTy = MemDescType::get(shape, elemTy, sharedLinear, smem);

    auto inferred = getNvmmaSmemAttrs(memTy);
    ASSERT_TRUE(inferred) << "sw=" << expected.swizzlingByteWidth
                          << " transposed=" << expected.transposed
                          << " fp4Padded=" << expected.fp4Padded;
    EXPECT_EQ(inferred->swizzlingByteWidth, expected.swizzlingByteWidth);
    EXPECT_EQ(inferred->transposed, expected.transposed);
    EXPECT_EQ(inferred->fp4Padded, expected.fp4Padded);
  };

  for (unsigned sw : {32u, 64u, 128u}) {
    for (bool transposed : {false, true}) {
      for (bool fp4Padded : {false, true}) {
        checkLayout({128, 128}, i8Ty,
                    nvmmaShared(sw, transposed, /*elementBitWidth=*/8, {1, 1},
                                {1, 1}, {1, 0}, {1, 0}, fp4Padded),
                    ExpectedLayout{sw, transposed, fp4Padded});
      }
    }
  }

  SmallVector<int64_t> sw0Shape = {128, 256};
  for (bool transposed : {false, true}) {
    for (bool fp4Padded : {false, true}) {
      auto sw0CoreMatrixTiled =
          getSwizzle0CoreMatrixLinearLayout(&ctx, sw0Shape,
                                            /*elementBitWidth=*/8, transposed,
                                            fp4Padded) *
          LinearLayout::identity1D(1, S("block"), S("dim0"));
      auto sw0SharedLinear = SharedLinearEncodingAttr::get(
          &ctx, std::move(sw0CoreMatrixTiled), /*layoutAlignment=*/1024);
      auto sw0MemTy = MemDescType::get(sw0Shape, i8Ty, sw0SharedLinear, smem);
      auto sw0Inferred = getNvmmaSmemAttrs(sw0MemTy);
      ASSERT_TRUE(sw0Inferred)
          << "transposed=" << transposed << " fp4Padded=" << fp4Padded;
      EXPECT_EQ(sw0Inferred->swizzlingByteWidth, 0u);
      EXPECT_EQ(sw0Inferred->transposed, transposed);
      EXPECT_EQ(sw0Inferred->fp4Padded, fp4Padded);
    }
  }

  struct Case {
    unsigned bitwidth;
    unsigned sw;
    bool transposed;
  };
  for (Case testCase :
       {Case{8, 32, true}, Case{16, 64, false}, Case{32, 128, true}}) {
    checkLayout({128, 128}, IntegerType::get(&ctx, testCase.bitwidth),
                nvmmaShared(testCase.sw, testCase.transposed, testCase.bitwidth,
                            {1, 1}, {1, 1}, {1, 0}, {1, 0}),
                ExpectedLayout{testCase.sw, testCase.transposed,
                               /*fp4Padded=*/false});
  }
}

TEST_F(NvmmaSmemAttrsTest, Fp4PaddedRequiresI8Storage) {
  auto checkLayout = [&](const LinearLayout &ll) {
    bool sawFp4Padded = false;
    for (unsigned bitwidth : {8u, 16u, 32u, 64u}) {
      auto attrsAndCandidate = getNvmmaSmemAttrs(ll, bitwidth);
      if (!attrsAndCandidate)
        continue;
      auto attrs = attrsAndCandidate->first;
      sawFp4Padded |= attrs.fp4Padded;
      EXPECT_TRUE(!attrs.fp4Padded || bitwidth == 8)
          << "fp4Padded with bitwidth=" << bitwidth;
    }
    return sawFp4Padded;
  };

  auto fp4 = nvmmaShared(/*swizzleSizeInBytes=*/128, /*transposed=*/false,
                         /*elementBitWidth=*/8, {1, 1}, {1, 1}, {1, 0}, {1, 0},
                         /*fp4Padded=*/true);
  // Positive control: matcher must produce fp4Padded for i8 storage.
  EXPECT_TRUE(checkLayout(toLinearLayout({128, 128}, fp4).pseudoinvert()));

  struct Case {
    unsigned bitwidth;
    unsigned sw;
    bool transposed;
  };
  for (Case testCase : {Case{8, 32, true}, Case{16, 64, false},
                        Case{32, 128, true}, Case{64, 128, false}}) {
    auto nvmma = nvmmaShared(testCase.sw, testCase.transposed,
                             testCase.bitwidth, {1, 1}, {1, 1}, {1, 0}, {1, 0});
    checkLayout(toLinearLayout({128, 128}, nvmma).pseudoinvert());
  }
}

TEST_F(NvmmaSmemAttrsTest, InferNvmmaSmemAttrsRejectsNearMisses) {
  auto i4Ty = IntegerType::get(&ctx, 4);
  auto smem = SharedMemorySpaceAttr::get(&ctx);
  auto inferSharedLinearInfo = [&](LinearLayout layout) {
    SmallVector<int64_t> shape;
    for (int32_t size : layout.getOutDimSizes())
      shape.push_back(size);
    auto sharedLinear = SharedLinearEncodingAttr::get(&ctx, std::move(layout),
                                                      /*layoutAlignment=*/1024);
    auto memTy = MemDescType::get(shape, i4Ty, sharedLinear, smem);
    return getNvmmaSmemAttrs(memTy);
  };
  auto nonPaddedPrefix = LinearLayout({{S("offset"),
                                        {{0, 1},
                                         {0, 2},
                                         {0, 4},
                                         {0, 8},
                                         {0, 16},
                                         {0, 32},
                                         {1, 8},
                                         {2, 16},
                                         {4, 32}}},
                                       {S("block"), {}}},
                                      {S("dim0"), S("dim1")});
  EXPECT_FALSE(inferSharedLinearInfo(std::move(nonPaddedPrefix)));

  auto shiftedZero = LinearLayout({{S("offset"),
                                    {{0, 1},
                                     {0, 2},
                                     {0, 4},
                                     {0, 8},
                                     {0, 0},
                                     {0, 16},
                                     {0, 32},
                                     {1, 8},
                                     {2, 16},
                                     {4, 32},
                                     {8, 0},
                                     {16, 0}}},
                                   {S("block"), {}}},
                                  {S("dim0"), S("dim1")});
  EXPECT_FALSE(inferSharedLinearInfo(std::move(shiftedZero)));

  // A valid fp4 prefix is not enough; the remaining bases must match.
  auto earlyRowInterleave = LinearLayout({{S("offset"),
                                           {{0, 1},
                                            {0, 2},
                                            {0, 4},
                                            {0, 0},
                                            {1, 8},
                                            {2, 16},
                                            {4, 32},
                                            {0, 8},
                                            {0, 16},
                                            {0, 32},
                                            {8, 0},
                                            {16, 0}}},
                                          {S("block"), {}}},
                                         {{S("dim0"), 32}, {S("dim1"), 64}},
                                         /*requireSurjective=*/true);
  EXPECT_FALSE(inferSharedLinearInfo(std::move(earlyRowInterleave)));
}

} // namespace
} // namespace mlir::triton::gpu
