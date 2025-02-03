#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Signals.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
std::ostream &operator<<(std::ostream &os, StringAttr str) {
  os << str.str();
  return os;
}
} // namespace mlir

namespace mlir::triton::gpu {
namespace {

class LinearLayoutConversionsTest : public ::testing::Test {
public:
  void SetUp() { ctx.getOrLoadDialect<TritonGPUDialect>(); }

  BlockedEncodingAttr blocked(ArrayRef<unsigned> spt, ArrayRef<unsigned> tpw,
                              ArrayRef<unsigned> wpb, ArrayRef<unsigned> cpg,
                              ArrayRef<unsigned> cSplit, ArrayRef<unsigned> ord,
                              ArrayRef<unsigned> cOrd) {
    return BlockedEncodingAttr::get(
        &ctx, spt, tpw, wpb, ord, CTALayoutAttr::get(&ctx, cpg, cSplit, cOrd));
  }

  NvidiaMmaEncodingAttr mma(unsigned versionMaj, unsigned versionMin,
                            ArrayRef<unsigned> instrShape,
                            ArrayRef<unsigned> wbp, ArrayRef<unsigned> cpg,
                            ArrayRef<unsigned> cSplit,
                            ArrayRef<unsigned> cOrd) {
    return NvidiaMmaEncodingAttr::get(
        &ctx, versionMaj, versionMin, wbp,
        CTALayoutAttr::get(&ctx, cpg, cSplit, cOrd), instrShape);
  }

  NvidiaMmaEncodingAttr mma(unsigned versionMaj, unsigned versionMin,
                            ArrayRef<unsigned> instrShape,
                            ArrayRef<unsigned> numWarps) {
    auto ctaLayout = CTALayoutAttr::getDefault(&ctx, numWarps.size());
    return NvidiaMmaEncodingAttr::get(&ctx, versionMaj, versionMin, numWarps,
                                      std::move(ctaLayout), instrShape);
  }

  DotOperandEncodingAttr dot(Attribute parent, int idx, int kWidth) {
    return DotOperandEncodingAttr::get(&ctx, idx, parent, /*kWidth=*/kWidth);
  }

  AMDMfmaEncodingAttr mfma(ArrayRef<unsigned> warps, unsigned mDim,
                           unsigned nDim, bool isTransposed) {
    SmallVector<unsigned> cpg(warps.size(), 1u);
    SmallVector<unsigned> cSplit(warps.size(), 1u);
    SmallVector<unsigned> cOrd(warps.size());
    std::iota(cOrd.begin(), cOrd.end(), 0);
    return AMDMfmaEncodingAttr::get(
        &ctx, /*versionMajor=*/2, /*versionMinor=*/0, warps, mDim, nDim,
        isTransposed, CTALayoutAttr::get(&ctx, cpg, cSplit, cOrd));
  }

  DotOperandEncodingAttr mfmaDotOp(AMDMfmaEncodingAttr mfma, unsigned opIdx,
                                   unsigned kWidth) {
    return DotOperandEncodingAttr::get(&ctx, opIdx, mfma, kWidth);
  }

  AMDWmmaEncodingAttr wmma(ArrayRef<unsigned> warps, int version,
                           bool transposed) {
    SmallVector<unsigned> cpg(warps.size(), 1u);
    SmallVector<unsigned> cSplit(warps.size(), 1u);
    SmallVector<unsigned> cOrd(warps.size());
    std::iota(cOrd.begin(), cOrd.end(), 0);
    return AMDWmmaEncodingAttr::get(
        &ctx, version, transposed, warps,
        CTALayoutAttr::get(&ctx, cpg, cSplit, cOrd));
  }

  DotOperandEncodingAttr wmmaDotOp(AMDWmmaEncodingAttr wmma, unsigned opIdx,
                                   unsigned kWidth) {
    return DotOperandEncodingAttr::get(&ctx, opIdx, wmma, kWidth);
  }

  SliceEncodingAttr slice(Attribute parent, int dim) {
    return SliceEncodingAttr::get(&ctx, dim, parent);
  }

  SwizzledSharedEncodingAttr shared(unsigned vec, unsigned perPhase,
                                    unsigned maxPhase, ArrayRef<unsigned> cpg,
                                    ArrayRef<unsigned> cSplit,
                                    ArrayRef<unsigned> ord,
                                    ArrayRef<unsigned> cOrd) {
    return SwizzledSharedEncodingAttr::get(
        &ctx, vec, perPhase, maxPhase, ord,
        CTALayoutAttr::get(&ctx, cpg, cSplit, cOrd));
  }

  NVMMASharedEncodingAttr
  nvmmaShared(unsigned swizzleSizeInBytes, bool transposed,
              unsigned elementBitWidth, ArrayRef<unsigned> cpg,
              ArrayRef<unsigned> cSplit, ArrayRef<unsigned> ord,
              ArrayRef<unsigned> cOrd, bool fp4Padded = false) {
    return NVMMASharedEncodingAttr::get(
        &ctx, swizzleSizeInBytes, transposed, elementBitWidth, fp4Padded,
        CTALayoutAttr::get(&ctx, cpg, cSplit, cOrd));
  }

  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(LinearLayoutConversionsTest, SimpleBlocked) {
  auto layout =
      toLinearLayout({16}, blocked({1}, {4}, {4}, {1}, {1}, {0}, {0}));
  EXPECT_THAT(layout, LinearLayout(
                          {
                              {S("register"), {}},
                              {S("lane"), {{1}, {2}}},
                              {S("warp"), {{4}, {8}}},
                              {S("block"), {}},
                          },
                          {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, CTADuplication) {
  auto layout = toLinearLayout(
      {32}, blocked({1}, {4}, {4}, /*cpg=*/{4}, /*cSplit=*/{2}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {}},
                            {S("lane"), {{1}, {2}}},
                            {S("warp"), {{4}, {8}}},
                            {S("block"), {{16}, {0}}},
                        },
                        {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, CTABroadcast) {
  auto layout =
      toLinearLayout({64, 128}, blocked({8, 1}, {8, 4}, {1, 4}, {1, 2}, {1, 2},
                                        {0, 1}, {1, 0}));
  EXPECT_EQ(
      layout,
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {0, 16}, {0, 32}}},
                    {S("lane"), {{8, 0}, {16, 0}, {32, 0}, {0, 1}, {0, 2}}},
                    {S("warp"), {{0, 4}, {0, 8}}},
                    {S("block"), {{0, 64}}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, ShapeLargerThanLayout) {
  // The layout is 16 elements, but the shape is 128, so it's repeated 128/16 =
  // 8 times.
  auto layout =
      toLinearLayout({128}, blocked({1}, {4}, {4}, {1}, {1}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {{16}, {32}, {64}}},
                            {S("lane"), {{1}, {2}}},
                            {S("warp"), {{4}, {8}}},
                            {S("block"), {}},
                        },
                        {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, ShapeLargerThanLayout2DDegenerate) {
  auto layout = toLinearLayout({128, 1}, blocked({1, 1}, {4, 1}, {4, 1}, {1, 1},
                                                 {1, 1}, {0, 1}, {1, 0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {{16, 0}, {32, 0}, {64, 0}}},
                            {S("lane"), {{1, 0}, {2, 0}}},
                            {S("warp"), {{4, 0}, {8, 0}}},
                            {S("block"), {}},
                        },
                        {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, ShapeSmallerThanLayout) {
  // The shape is 8 elements, but the layout is 4*4*4 = 64 elems.  Therefore the
  // log2(64/8) = 3 most major bases are 0.
  auto layout = toLinearLayout({8}, blocked({4}, {4}, {4}, {1}, {1}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {{1}, {2}}},
                            {S("lane"), {{4}, {0}}},
                            {S("warp"), {{0}, {0}}},
                            {S("block"), {}},
                        },
                        {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, ReversedOrder) {
  auto layout = toLinearLayout({1, 64}, blocked({1, 1}, {32, 1}, {1, 8}, {1, 1},
                                                {1, 1}, {0, 1}, {1, 0}));
  EXPECT_EQ(layout,
            LinearLayout(
                {
                    {S("register"), {{0, 8}, {0, 16}, {0, 32}}},
                    {S("lane"), {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}},
                    {S("warp"), {{0, 1}, {0, 2}, {0, 4}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, ReplicateInRegisterDim) {
  auto layout =
      toLinearLayout({32}, blocked({2}, {4}, {1}, {1}, {1}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {{1}, {8}, {16}}},
                            {S("lane"), {{2}, {4}}},
                            {S("warp"), {}},
                            {S("block"), {}},
                        },
                        {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, OneDimTooLargeAnotherTooSmall) {
  auto blockedLayout =
      blocked({1, 4}, {8, 4}, {4, 1}, {2, 2}, {2, 1}, {1, 0}, {1, 0});
  auto ll = toLinearLayout({128, 16}, blockedLayout);
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {{0, 1}, {0, 2}, {32, 0}}},
                        {S("lane"), {{0, 4}, {0, 8}, {1, 0}, {2, 0}, {4, 0}}},
                        {S("warp"), {{8, 0}, {16, 0}}},
                        {S("block"), {{0, 0}, {64, 0}}},
                    },
                    {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, RepeatInCTGDimFirst) {
  // We have a 4-element shape and an 8-element layout (4 elems per CTA).  So
  // the layout will map two inputs to each output.  The question is, which two
  // inputs?  The answer is, we split between CTAs first, so the two CTAs have
  // distinct elements.
  auto blockedLayout = blocked({1}, {1}, {4}, {2}, {2}, {0}, {0});
  auto ll = toLinearLayout({4}, blockedLayout);
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {}},
                        {S("lane"), {}},
                        {S("warp"), {{1}, {0}}},
                        {S("block"), {{2}}},
                    },
                    {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, SmallerThanCTALayout) {
  auto blockedLayout = blocked({1}, {1}, {1}, {4}, {4}, {0}, {0});
  auto ll = toLinearLayout({2}, blockedLayout);
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {}},
                        {S("lane"), {}},
                        {S("warp"), {}},
                        {S("block"), {{1}, {0}}},
                    },
                    {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, Skinny) {
  auto blockedLayout =
      blocked({8, 1}, {8, 4}, {1, 4}, {1, 2}, {1, 2}, {0, 1}, {0, 1});
  auto ll = toLinearLayout({64, 1}, blockedLayout);
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                        {S("lane"), {{8, 0}, {16, 0}, {32, 0}, {0, 0}, {0, 0}}},
                        {S("warp"), {{0, 0}, {0, 0}}},
                        {S("block"), {{0, 0}}},
                    },
                    {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedOrder) {
  auto ll = toLinearLayout({1024, 128}, blocked({2, 2}, {4, 8}, {2, 2}, {2, 2},
                                                {2, 2}, {1, 0}, {1, 0}));
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"),
                         {
                             {0, 1},
                             {1, 0},
                             {0, 32},
                             {16, 0},
                             {32, 0},
                             {64, 0},
                             {128, 0},
                             {256, 0},
                         }},
                        {S("lane"), {{0, 2}, {0, 4}, {0, 8}, {2, 0}, {4, 0}}},
                        {S("warp"), {{0, 16}, {8, 0}}},
                        {S("block"), {{0, 64}, {512, 0}}},
                    },
                    {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, Blocked4D) {
  auto ll = toLinearLayout({2, 1, 1, 1},
                           blocked({1, 1, 1, 4}, {2, 1, 1, 16}, {1, 2, 4, 1},
                                   {1, 1, 1, 1}, {1, 1, 1, 1}, {3, 0, 1, 2},
                                   {3, 2, 1, 0}));
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {{0, 0, 0, 0}, {0, 0, 0, 0}}},
                        {S("lane"),
                         {{0, 0, 0, 0},
                          {0, 0, 0, 0},
                          {0, 0, 0, 0},
                          {0, 0, 0, 0},
                          {1, 0, 0, 0}}},
                        {S("warp"), {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
                        {S("block"), {}},
                    },
                    {S("dim0"), S("dim1"), S("dim2"), S("dim3")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedDotOperandLhs) {
  auto parent = blocked(/*size*/ {2, 4}, /*threads*/ {8, 4}, /*warps*/ {2, 4},
                        /*ctas*/ {1, 1}, /*splits*/ {1, 1}, /*order*/ {1, 0},
                        /*cta order*/ {1, 0});
  auto dotOperand = dot(parent, /*idx*/ 0, /*kWidth*/ 0);
  EXPECT_EQ(
      toLinearLayout({32, 16}, dotOperand),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                    {S("lane"), {{0, 0}, {0, 0}, {2, 0}, {4, 0}, {8, 0}}},
                    {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedDot3dOperandLhs) {
  auto parent =
      blocked(/*size*/ {2, 2, 4}, /*threads*/ {2, 4, 4}, /*warps*/ {2, 2, 2},
              /*ctas*/ {1, 1, 1}, /*splits*/ {1, 1, 1}, /*order*/ {2, 1, 0},
              /*cta order*/ {2, 1, 0});
  auto dotOperand = dot(parent, /*idx*/ 0, /*kWidth*/ 0);
  EXPECT_EQ(
      toLinearLayout({16, 32, 4}, dotOperand),
      LinearLayout(
          {{S("register"),
            {{0, 0, 1},
             {0, 0, 2},
             {0, 1, 0},
             {1, 0, 0},
             {0, 16, 0},
             {8, 0, 0}}},
           {S("lane"), {{0, 0, 0}, {0, 0, 0}, {0, 2, 0}, {0, 4, 0}, {2, 0, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 8, 0}, {4, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedDotOperandRhs) {
  auto parent = blocked(/*size*/ {2, 4}, /*threads*/ {8, 4}, /*warps*/ {2, 4},
                        /*ctas*/ {1, 1}, /*splits*/ {1, 1}, /*order*/ {1, 0},
                        /*cta order*/ {1, 0});
  auto dotOperand = dot(parent, /*idx*/ 1, /*kWidth*/ 0);
  EXPECT_EQ(toLinearLayout({16, 64}, dotOperand),
            LinearLayout({{S("register"),
                           {{0, 1}, {0, 2}, {1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                          {S("lane"), {{0, 4}, {0, 8}, {0, 0}, {0, 0}, {0, 0}}},
                          {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedDot3dOperandRhs) {
  auto parent =
      blocked(/*size*/ {2, 2, 4}, /*threads*/ {2, 4, 4}, /*warps*/ {2, 2, 2},
              /*ctas*/ {1, 1, 1}, /*splits*/ {1, 1, 1}, /*order*/ {2, 1, 0},
              /*cta order*/ {2, 1, 0});
  auto dotOperand = dot(parent, /*idx*/ 1, /*kWidth*/ 0);
  EXPECT_EQ(
      toLinearLayout({16, 4, 64}, dotOperand),
      LinearLayout(
          {{S("register"),
            {{0, 0, 1},
             {0, 0, 2},
             {0, 1, 0},
             {0, 2, 0},
             {1, 0, 0},
             {0, 0, 32},
             {8, 0, 0}}},
           {S("lane"), {{0, 0, 4}, {0, 0, 8}, {0, 0, 0}, {0, 0, 0}, {2, 0, 0}}},
           {S("warp"), {{0, 0, 16}, {0, 0, 0}, {4, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_16x16) {
  EXPECT_EQ(toLinearLayout({16, 16},
                           mma(2, 0, {16, 8}, {1, 1}, {1, 1}, {1, 1}, {0, 1})),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_32x32) {
  EXPECT_EQ(toLinearLayout({32, 32},
                           mma(2, 0, {16, 8}, {1, 1}, {1, 1}, {1, 1}, {0, 1})),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {16, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_ExtendDim2) {
  EXPECT_EQ(toLinearLayout({16, 128},
                           mma(2, 0, {16, 8}, {1, 1}, {1, 1}, {1, 1}, {0, 1})),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {0, 32}, {0, 64}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_Cga) {
  EXPECT_EQ(
      toLinearLayout({64, 128, 128}, mma(2, 0, {1, 16, 8}, {16, 1, 1},
                                         {4, 2, 2}, {4, 2, 1}, {2, 1, 0})),
      LinearLayout(
          {
              {S("register"),
               {
                   {0, 0, 1},
                   {0, 8, 0},
                   {0, 0, 8},
                   {0, 0, 16},
                   {0, 0, 32},
                   {0, 0, 64},
                   {0, 16, 0},
                   {0, 32, 0},
               }},
              {S("lane"),
               {{0, 0, 2}, {0, 0, 4}, {0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
              {S("warp"), {{1, 0, 0}, {2, 0, 0}, {4, 0, 0}, {8, 0, 0}}},
              {S("block"), {{0, 0, 0}, {0, 64, 0}, {16, 0, 0}, {32, 0, 0}}},
          },
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_Small3D) {
  EXPECT_EQ(toLinearLayout({1, 128, 128}, mma(2, 0, {1, 16, 8}, {16, 1, 1},
                                              {4, 2, 2}, {4, 2, 1}, {2, 1, 0})),
            LinearLayout(
                {
                    {S("register"),
                     {
                         {0, 0, 1},
                         {0, 8, 0},
                         {0, 0, 8},
                         {0, 0, 16},
                         {0, 0, 32},
                         {0, 0, 64},
                         {0, 16, 0},
                         {0, 32, 0},
                     }},
                    {S("lane"),
                     {{0, 0, 2}, {0, 0, 4}, {0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
                    {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
                    {S("block"), {{0, 0, 0}, {0, 64, 0}, {0, 0, 0}, {0, 0, 0}}},
                },
                {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv3_64x16) {
  SmallVector<SmallVector<unsigned>, 2> instrShapes = {{16, 16, 8}, {16, 8, 8}};
  for (auto instrShape : instrShapes) {
    SCOPED_TRACE(triton::join(instrShape, ","));
    EXPECT_EQ(toLinearLayout({64, 16}, mma(3, 0, instrShape, {4, 1}, {1, 1},
                                           {1, 1}, {1, 0})),
              LinearLayout(
                  {
                      {S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                      {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                      {S("warp"), {{16, 0}, {32, 0}}},
                      {S("block"), {}},
                  },
                  {S("dim0"), S("dim1")}));
  }
}

TEST_F(LinearLayoutConversionsTest, MMAv3_128x16) {
  EXPECT_EQ(toLinearLayout({128, 16}, mma(3, 0, {16, 16, 8}, {4, 1}, {1, 1},
                                          {1, 1}, {1, 0})),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {64, 0}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv3_1024x1024) {
  EXPECT_EQ(toLinearLayout({1024, 1024}, mma(3, 0, {16, 16, 8}, {4, 1}, {1, 1},
                                             {1, 1}, {1, 0})),
            LinearLayout({{S("register"),
                           {{0, 1},
                            {8, 0},
                            {0, 8},
                            {0, 16},
                            {0, 32},
                            {0, 64},
                            {0, 128},
                            {0, 256},
                            {0, 512},
                            {64, 0},
                            {128, 0},
                            {256, 0},
                            {512, 0}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv3_4x2Warps) {
  auto legacy = mma(3, 0, {16, 32, 16}, {4, 2}, {1, 1}, {1, 1}, {1, 0});

  EXPECT_EQ(toLinearLayout({64, 32}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 64}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 32}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({128, 64}, legacy),
      LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {64, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}, {0, 32}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({256, 64}, legacy),
      LinearLayout({{S("register"),
                     {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {64, 0}, {128, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}, {0, 32}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv3_4x4Warps) {
  auto legacy = mma(3, 0, {16, 16, 8}, {4, 4}, {1, 1}, {1, 1}, {1, 0});

  EXPECT_EQ(toLinearLayout({16, 16}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 16}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 16}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({128, 16}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {64, 0}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 32}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {0, 0}, {0, 16}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 32}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 16}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv2_tile_kwidth8) {
  auto parent = mma(2, 0, {16, 8}, {1, 1});
  EXPECT_EQ(toLinearLayout({16, 64}, dot(parent, 0, 8)),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}}},
                    {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 8}, dot(parent, 1, 8)),
            LinearLayout(
                {
                    {S("register"), {{1, 0}, {2, 0}, {4, 0}, {32, 0}}},
                    {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv2_large_warp4_kwidth8) {
  auto parent = mma(2, 0, {16, 8}, {4, 1});
  EXPECT_EQ(
      toLinearLayout({128, 128}, dot(parent, 0, 8)),
      LinearLayout(
          {
              {S("register"),
               {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}, {0, 64}, {64, 0}}},
              {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
              {S("warp"), {{16, 0}, {32, 0}}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({128, 64}, dot(parent, 1, 8)),
            LinearLayout(
                {
                    {S("register"),
                     {{1, 0},
                      {2, 0},
                      {4, 0},
                      {32, 0},
                      {64, 0},
                      {0, 8},
                      {0, 16},
                      {0, 32}}},
                    {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
                    {
                        S("warp"),
                        {{0, 0}, {0, 0}},
                    },
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, dot(parent, 1, 8)),
            LinearLayout(
                {
                    {S("register"),
                     {{1, 0},
                      {2, 0},
                      {4, 0},
                      {32, 0},
                      {0, 8},
                      {0, 16},
                      {0, 32},
                      {0, 64}}},
                    {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
                    {
                        S("warp"),
                        {{0, 0}, {0, 0}},
                    },
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv2_3D) {
  // We implement one that exercises all the paths
  auto parent = mma(2, 0, {1, 16, 8}, {2, 4, 2});
  EXPECT_EQ(toLinearLayout({16, 128, 128}, dot(parent, 0, 8)),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 0, 1},
                      {0, 0, 2},
                      {0, 0, 4},
                      {0, 8, 0},
                      {0, 0, 32},
                      {0, 0, 64},
                      {0, 64, 0},
                      {2, 0, 0},
                      {4, 0, 0},
                      {8, 0, 0}}},
                    {S("lane"),
                     {{0, 0, 8}, {0, 0, 16}, {0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
                    {S("warp"), {{0, 0, 0}, {0, 16, 0}, {0, 32, 0}, {1, 0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(toLinearLayout({8, 128, 64}, dot(parent, 1, 8)),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 1, 0},
                      {0, 2, 0},
                      {0, 4, 0},
                      {0, 32, 0},
                      {0, 64, 0},
                      {0, 0, 16},
                      {0, 0, 32},
                      {2, 0, 0},
                      {4, 0, 0}}},
                    {S("lane"),
                     {{0, 8, 0}, {0, 16, 0}, {0, 0, 1}, {0, 0, 2}, {0, 0, 4}}},
                    {
                        S("warp"),
                        {{0, 0, 8}, {0, 0, 0}, {0, 0, 0}, {1, 0, 0}},
                    },
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv3_warp4_kwidth2) {
  auto parent = mma(3, 0, {16, 16, 8}, {4, 1});
  auto dotOp = dot(parent, 0, 2);

  EXPECT_EQ(toLinearLayout({64, 16}, dotOp),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({128, 16}, dotOp),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}, {64, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({128, 32}, dotOp),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {64, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv3_mixed_warp_kwidth4) {
  // Testing dot with MMAv3 encoding for opIdx = 0 and kWidth = 4
  auto parent = mma(3, 0, {16, 16, 8}, {4, 2});
  auto dotOp = dot(parent, 0, 4);

  EXPECT_EQ(toLinearLayout({128, 64}, dotOp),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 1}, {0, 2}, {8, 0}, {0, 16}, {0, 32}, {64, 0}}},
                    {S("lane"), {{0, 4}, {0, 8}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}, {0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv2_split_warp_kwidth8) {
  auto parent = mma(2, 0, {16, 8}, {2, 2});
  EXPECT_EQ(
      toLinearLayout({32, 64}, dot(parent, 0, 8)),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}}},
                    {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{0, 0}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({64, 16}, dot(parent, 1, 8)),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {32, 0}}},
                    {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
                    {S("warp"), {{0, 8}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, dot(parent, 0, 8)),
            LinearLayout(
                {{S("register"),
                  {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}, {0, 64}, {32, 0}}},
                 {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
                 {S("warp"), {{0, 0}, {16, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({128, 32}, dot(parent, 1, 8)),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {32, 0}, {64, 0}, {0, 16}}},
           {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
           {S("warp"), {{0, 8}, {0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SliceDot) {
  // Slice layout with a DotOperand (MMAv2) as the parent.
  auto parentV2 = dot(mma(2, 0, {16, 8}, {1, 1}), /*opIdx=*/0, /*kWidth=*/8);
  auto sliceV2 = slice(parentV2, /*dim=*/1);

  EXPECT_EQ(toLinearLayout({16}, sliceV2),
            LinearLayout(
                {
                    {S("register"), {{8}}},
                    {S("lane"), {{0}, {0}, {1}, {2}, {4}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0")}));

  // Slice layout with a DotOperand (MMAv3) as the parent.
  auto parentV3 =
      dot(mma(3, 0, {16, 16, 8}, {4, 1}), /*opIdx=*/0, /*kWidth=*/2);
  auto sliceV3 = slice(parentV3, /*dim=*/0);

  EXPECT_EQ(toLinearLayout({16}, sliceV3),
            LinearLayout(
                {
                    {S("register"), {{1}, {8}}},
                    {S("lane"), {{2}, {4}, {0}, {0}, {0}}},
                    {S("warp"), {{0}, {0}}},
                    {S("block"), {}},
                },
                {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, MFMA32_2x4Warps) {
  auto mfmaNT = mfma(/*warps=*/{2, 4}, /*mDim=*/32, /*nDim=*/32,
                     /*isTransposed=*/false);

  EXPECT_EQ(toLinearLayout({32, 32}, mfmaNT),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {8, 0}, {16, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 32}, mfmaNT),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {8, 0}, {16, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({128, 128}, mfmaNT),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {8, 0}, {16, 0}, {64, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
                 {S("warp"), {{0, 32}, {0, 64}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  auto mfmaT = mfma(/*warps=*/{2, 4}, /*mDim=*/32, /*nDim=*/32,
                    /*isTransposed=*/true);

  EXPECT_EQ(toLinearLayout({32, 32}, mfmaT),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 8}, {0, 16}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 32}, mfmaT),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 8}, {0, 16}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
                 {S("warp"), {{0, 0}, {0, 0}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({128, 128}, mfmaT),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 8}, {0, 16}, {64, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
                 {S("warp"), {{0, 32}, {0, 64}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MFMA16_2x4Warps) {
  auto mfmaNT = mfma(/*warps=*/{2, 4}, /*mDim=*/16, /*nDim=*/16,
                     /*isTransposed=*/false);
  EXPECT_EQ(toLinearLayout({16, 16}, mfmaNT),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MFMA32_2x4x1Warps) {
  auto mfmaNT = mfma(/*warps=*/{2, 4, 1}, /*mDim=*/32, /*nDim=*/32,
                     /*isTransposed=*/false);

  EXPECT_EQ(toLinearLayout({1, 128, 128}, mfmaNT),
            LinearLayout({{S("register"),
                           {{0, 1, 0},
                            {0, 2, 0},
                            {0, 8, 0},
                            {0, 16, 0},
                            {0, 0, 32},
                            {0, 0, 64}}},
                          {S("lane"),
                           {{0, 0, 1},
                            {0, 0, 2},
                            {0, 0, 4},
                            {0, 0, 8},
                            {0, 0, 16},
                            {0, 4, 0}}},
                          {S("warp"), {{0, 32, 0}, {0, 64, 0}, {0, 0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(toLinearLayout({2, 32, 32}, mfmaNT),
            LinearLayout(
                {{S("register"), {{0, 1, 0}, {0, 2, 0}, {0, 8, 0}, {0, 16, 0}}},
                 {S("lane"),
                  {{0, 0, 1},
                   {0, 0, 2},
                   {0, 0, 4},
                   {0, 0, 8},
                   {0, 0, 16},
                   {0, 4, 0}}},
                 {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(toLinearLayout({2, 64, 32}, mfmaNT),
            LinearLayout(
                {{S("register"), {{0, 1, 0}, {0, 2, 0}, {0, 8, 0}, {0, 16, 0}}},
                 {S("lane"),
                  {{0, 0, 1},
                   {0, 0, 2},
                   {0, 0, 4},
                   {0, 0, 8},
                   {0, 0, 16},
                   {0, 4, 0}}},
                 {S("warp"), {{0, 32, 0}, {0, 0, 0}, {1, 0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1"), S("dim2")}));

  auto mfmaT = mfma(/*warps=*/{2, 4, 1}, /*mDim=*/32, /*nDim=*/32,
                    /*isTransposed=*/true);

  EXPECT_EQ(toLinearLayout({1, 128, 128}, mfmaT),
            LinearLayout({{S("register"),
                           {{0, 0, 1},
                            {0, 0, 2},
                            {0, 0, 8},
                            {0, 0, 16},
                            {0, 0, 32},
                            {0, 0, 64}}},
                          {S("lane"),
                           {{0, 1, 0},
                            {0, 2, 0},
                            {0, 4, 0},
                            {0, 8, 0},
                            {0, 16, 0},
                            {0, 0, 4}}},
                          {S("warp"), {{0, 32, 0}, {0, 64, 0}, {0, 0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(toLinearLayout({2, 32, 32}, mfmaT),
            LinearLayout(
                {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 8}, {0, 0, 16}}},
                 {S("lane"),
                  {{0, 1, 0},
                   {0, 2, 0},
                   {0, 4, 0},
                   {0, 8, 0},
                   {0, 16, 0},
                   {0, 0, 4}}},
                 {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(toLinearLayout({2, 64, 32}, mfmaT),
            LinearLayout(
                {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 8}, {0, 0, 16}}},
                 {S("lane"),
                  {{0, 1, 0},
                   {0, 2, 0},
                   {0, 4, 0},
                   {0, 8, 0},
                   {0, 16, 0},
                   {0, 0, 4}}},
                 {S("warp"), {{0, 32, 0}, {0, 0, 0}, {1, 0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, warp1onK_mfma32_lhs_kwidth8) {
  auto parentMfma_1_8 = mfma(/*warps=*/{1, 8}, /*mDim=*/32, /*nDim=*/32,
                             /*isTransposed=*/false);
  auto mfmaDot_1_8 = mfmaDotOp(parentMfma_1_8, /*opIdx=*/0, /*kWidth=*/8);
  EXPECT_EQ(toLinearLayout({128, 128}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 4},
                   {0, 16},
                   {0, 32},
                   {0, 64},
                   {32, 0},
                   {64, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({128, 256}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 4},
                   {0, 16},
                   {0, 32},
                   {0, 64},
                   {0, 128},
                   {32, 0},
                   {64, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({32, 64}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 16}, {0, 32}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({256, 256}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 4},
                   {0, 16},
                   {0, 32},
                   {0, 64},
                   {0, 128},
                   {32, 0},
                   {64, 0},
                   {128, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, warp1onK_mfma32_rhs_kwidth8) {
  auto parentMfma_1_8 = mfma(/*warps=*/{1, 8}, /*mDim=*/32, /*nDim=*/32,
                             /*isTransposed=*/false);
  auto mfmaDot_1_8 = mfmaDotOp(parentMfma_1_8, /*opIdx=*/1, /*kWidth=*/8);
  EXPECT_EQ(
      toLinearLayout({128, 128}, mfmaDot_1_8),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {16, 0}, {32, 0}, {64, 0}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {8, 0}}},
           {S("warp"), {{0, 32}, {0, 64}, {0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(
      toLinearLayout({128, 256}, mfmaDot_1_8),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {16, 0}, {32, 0}, {64, 0}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {8, 0}}},
           {S("warp"), {{0, 32}, {0, 64}, {0, 128}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({32, 64}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {16, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {8, 0}}},
                 {S("warp"), {{0, 32}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(
      toLinearLayout({256, 256}, mfmaDot_1_8),
      LinearLayout(
          {{S("register"),
            {{1, 0}, {2, 0}, {4, 0}, {16, 0}, {32, 0}, {64, 0}, {128, 0}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {8, 0}}},
           {S("warp"), {{0, 32}, {0, 64}, {0, 128}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 0}, {8, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  auto parentMfma_1_4 = mfma(/*warps=*/{1, 4}, /*mDim=*/32, /*nDim=*/32,
                             /*isTransposed=*/false);
  auto mfmaDot_1_4 = mfmaDotOp(parentMfma_1_4, /*opIdx=*/1, /*kWidth=*/8);
  EXPECT_EQ(toLinearLayout({256, 256}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"),
                  {{1, 0},
                   {2, 0},
                   {4, 0},
                   {16, 0},
                   {32, 0},
                   {64, 0},
                   {128, 0},
                   {0, 128}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {8, 0}}},
                 {S("warp"), {{0, 32}, {0, 64}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, warp1onK_mfma16_lhs_kwidth8) {
  auto parentMfma_1_4 = mfma(/*warps=*/{1, 4}, /*mDim=*/16, /*nDim=*/16,
                             /*isTransposed=*/false);
  auto mfmaDot_1_4 = mfmaDotOp(parentMfma_1_4, /*opIdx=*/0, /*kWidth=*/8);
  EXPECT_EQ(toLinearLayout({128, 128}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 4},
                   {0, 32},
                   {0, 64},
                   {16, 0},
                   {32, 0},
                   {64, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}, {0, 16}}},
                 {S("warp"), {{0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({1, 128}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"),
                  {
                      {0, 1},
                      {0, 2},
                      {0, 4},
                      {0, 32},
                      {0, 64},
                  }},
                 {S("lane"), {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 8}, {0, 16}}},
                 {S("warp"), {{0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(
      toLinearLayout({128, 1}, mfmaDot_1_4),
      LinearLayout(
          {{S("register"), {{0, 0}, {0, 0}, {0, 0}, {16, 0}, {32, 0}, {64, 0}}},
           {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}, {0, 0}}},
           {S("warp"), {{0, 0}, {0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({256, 256}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 4},
                   {0, 32},
                   {0, 64},
                   {0, 128},
                   {16, 0},
                   {32, 0},
                   {64, 0},
                   {128, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}, {0, 16}}},
                 {S("warp"), {{0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}, {0, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  auto parentMfma_1_8 = mfma(/*warps=*/{1, 8}, /*mDim=*/16, /*nDim=*/16,
                             /*isTransposed=*/false);
  auto mfmaDot_1_8 = mfmaDotOp(parentMfma_1_8, /*opIdx=*/0, /*kWidth=*/8);
  EXPECT_EQ(toLinearLayout({256, 256}, mfmaDot_1_8),
            LinearLayout(
                {{S("register"),
                  {{0, 1},
                   {0, 2},
                   {0, 4},
                   {0, 32},
                   {0, 64},
                   {0, 128},
                   {16, 0},
                   {32, 0},
                   {64, 0},
                   {128, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}, {0, 16}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  auto parentMfma_1_8_1 = mfma(/*warps=*/{1, 1, 8}, /*mDim=*/16, /*nDim=*/16,
                               /*isTransposed=*/false);
  auto mfmaDot_1_8_1 = mfmaDotOp(parentMfma_1_8_1, /*opIdx=*/0, /*kWidth=*/8);

  EXPECT_EQ(toLinearLayout({1, 256, 256}, mfmaDot_1_8_1),
            LinearLayout({{S("register"),
                           {{0, 0, 1},
                            {0, 0, 2},
                            {0, 0, 4},
                            {0, 0, 32},
                            {0, 0, 64},
                            {0, 0, 128},
                            {0, 16, 0},
                            {0, 32, 0},
                            {0, 64, 0},
                            {0, 128, 0}}},
                          {S("lane"),
                           {{0, 1, 0},
                            {0, 2, 0},
                            {0, 4, 0},
                            {0, 8, 0},
                            {0, 0, 8},
                            {0, 0, 16}}},
                          {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, warp1onK_mfma16_rhs_kwidth8) {
  auto parentMfma_1_4 = mfma(/*warps=*/{1, 4}, /*mDim=*/16, /*nDim=*/16,
                             /*isTransposed=*/false);
  auto mfmaDot_1_4 = mfmaDotOp(parentMfma_1_4, /*opIdx=*/1, /*kWidth=*/8);
  EXPECT_EQ(
      toLinearLayout({128, 128}, mfmaDot_1_4),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {32, 0}, {64, 0}, {0, 64}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}, {16, 0}}},
           {S("warp"), {{0, 16}, {0, 32}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({1, 128}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"), {{0, 0}, {0, 0}, {0, 0}, {0, 64}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 0}, {0, 0}}},
                 {S("warp"), {{0, 16}, {0, 32}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({128, 1}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {32, 0}, {64, 0}}},
                 {S("lane"), {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {8, 0}, {16, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({256, 256}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"),
                  {{1, 0},
                   {2, 0},
                   {4, 0},
                   {32, 0},
                   {64, 0},
                   {128, 0},
                   {0, 64},
                   {0, 128}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}, {16, 0}}},
                 {S("warp"), {{0, 16}, {0, 32}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDot_1_4),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}, {0, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  auto parentMfma_1_8 = mfma(/*warps=*/{1, 8}, /*mDim=*/16, /*nDim=*/16,
                             /*isTransposed=*/false);
  auto mfmaDot_1_8 = mfmaDotOp(parentMfma_1_8, /*opIdx=*/1, /*kWidth=*/8);
  EXPECT_EQ(
      toLinearLayout({256, 256}, mfmaDot_1_8),
      LinearLayout(
          {{S("register"),
            {{1, 0}, {2, 0}, {4, 0}, {32, 0}, {64, 0}, {128, 0}, {0, 128}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}, {16, 0}}},
           {S("warp"), {{0, 16}, {0, 32}, {0, 64}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));

  auto parentMfma_1_8_1 = mfma(/*warps=*/{1, 1, 8}, /*mDim=*/16, /*nDim=*/16,
                               /*isTransposed=*/false);
  auto mfmaDot_1_8_1 = mfmaDotOp(parentMfma_1_8_1, /*opIdx=*/1, /*kWidth=*/8);

  EXPECT_EQ(toLinearLayout({1, 256, 256}, mfmaDot_1_8_1),
            LinearLayout({{S("register"),
                           {{0, 1, 0},
                            {0, 2, 0},
                            {0, 4, 0},
                            {0, 32, 0},
                            {0, 64, 0},
                            {0, 128, 0},
                            {0, 0, 128}}},
                          {S("lane"),
                           {{0, 0, 1},
                            {0, 0, 2},
                            {0, 0, 4},
                            {0, 0, 8},
                            {0, 8, 0},
                            {0, 16, 0}}},
                          {S("warp"), {{0, 0, 16}, {0, 0, 32}, {0, 0, 64}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, mfma32_dot_op_lhs_kwidth4) {
  auto parentMfma32 = mfma(/*warps=*/{2, 4}, /*mDim=*/32, /*nDim=*/32,
                           /*isTransposed=*/false);
  auto mfmaDotOp0_32 = mfmaDotOp(parentMfma32, /*opIdx=*/0, /*kWidth=*/4);
  EXPECT_EQ(toLinearLayout({128, 128}, mfmaDotOp0_32),
            LinearLayout(
                {{S("register"),
                  {{0, 1}, {0, 2}, {0, 8}, {0, 16}, {0, 32}, {0, 64}, {64, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
                 {S("warp"), {{0, 0}, {0, 0}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 32}, mfmaDotOp0_32),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 8}, {0, 16}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
                 {S("warp"), {{0, 0}, {0, 0}, {32, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDotOp0_32),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 8}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}, {0, 4}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  // Dot operand based on transposed mfma layout has same layout as ordinary
  auto parentTMfma32 = mfma(/*warps=*/{2, 4}, /*mDim=*/32, /*nDim=*/32,
                            /*isTransposed=*/true);
  auto tmfmaDotOp0_32 = mfmaDotOp(parentTMfma32, /*opIdx=*/0, /*kWidth=*/4);

  EXPECT_EQ(toLinearLayout({128, 128}, tmfmaDotOp0_32),
            toLinearLayout({128, 128}, mfmaDotOp0_32));
  EXPECT_EQ(toLinearLayout({64, 32}, tmfmaDotOp0_32),
            toLinearLayout({64, 32}, mfmaDotOp0_32));
  EXPECT_EQ(toLinearLayout({16, 16}, tmfmaDotOp0_32),
            toLinearLayout({16, 16}, mfmaDotOp0_32));
}

TEST_F(LinearLayoutConversionsTest, mfma16_dot_op_lhs_kwidth4) {
  auto parentMfma16 = mfma(/*warps=*/{2, 4}, /*mDim=*/16, /*nDim=*/16,
                           /*isTransposed=*/false);
  auto mfmaDotOp0_16 = mfmaDotOp(parentMfma16, /*opIdx=*/0, /*kWidth=*/4);
  EXPECT_EQ(
      toLinearLayout({128, 128}, mfmaDotOp0_16),
      LinearLayout(
          {{S("register"),
            {{0, 1}, {0, 2}, {0, 16}, {0, 32}, {0, 64}, {32, 0}, {64, 0}}},
           {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}, {0, 8}}},
           {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 32}, mfmaDotOp0_16),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}, {0, 16}, {32, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDotOp0_16),
            LinearLayout(
                {{S("register"), {{0, 1}, {0, 2}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  // Dot operand based on transposed mfma layout has same layout as ordinary
  auto parentTMfma16 = mfma(/*warps=*/{2, 4}, /*mDim=*/16, /*nDim=*/16,
                            /*isTransposed=*/true);
  auto tmfmaDotOp0_16 = mfmaDotOp(parentTMfma16, /*opIdx=*/0, /*kWidth=*/4);

  EXPECT_EQ(toLinearLayout({128, 128}, tmfmaDotOp0_16),
            toLinearLayout({128, 128}, mfmaDotOp0_16));
  EXPECT_EQ(toLinearLayout({64, 32}, tmfmaDotOp0_16),
            toLinearLayout({64, 32}, mfmaDotOp0_16));
  EXPECT_EQ(toLinearLayout({16, 16}, tmfmaDotOp0_16),
            toLinearLayout({16, 16}, mfmaDotOp0_16));
}

TEST_F(LinearLayoutConversionsTest, mfma32_dot_op_rhs_kwidth4) {
  auto parentMfma32 = mfma(/*warps=*/{2, 4}, /*mDim=*/32, /*nDim=*/32,
                           /*isTransposed=*/false);
  auto mfmaDotOp1_32 = mfmaDotOp(parentMfma32, /*opIdx=*/1, /*kWidth=*/4);
  EXPECT_EQ(
      toLinearLayout({128, 128}, mfmaDotOp1_32),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {8, 0}, {16, 0}, {32, 0}, {64, 0}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
           {S("warp"), {{0, 32}, {0, 64}, {0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 64}, mfmaDotOp1_32),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {8, 0}, {16, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}}},
                 {S("warp"), {{0, 32}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDotOp1_32),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {8, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 0}, {4, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  // Dot operand based on transposed mfma layout has same layout as ordinary
  auto parentTMfma32 = mfma(/*warps=*/{2, 4}, /*mDim=*/32, /*nDim=*/32,
                            /*isTransposed=*/true);
  auto tmfmaDotOp1_32 = mfmaDotOp(parentTMfma32, /*opIdx=*/1, /*kWidth=*/4);

  EXPECT_EQ(toLinearLayout({128, 128}, tmfmaDotOp1_32),
            toLinearLayout({128, 128}, mfmaDotOp1_32));
  EXPECT_EQ(toLinearLayout({64, 32}, tmfmaDotOp1_32),
            toLinearLayout({64, 32}, mfmaDotOp1_32));
  EXPECT_EQ(toLinearLayout({16, 16}, tmfmaDotOp1_32),
            toLinearLayout({16, 16}, mfmaDotOp1_32));
}

TEST_F(LinearLayoutConversionsTest, mfma16_dot_op_rhs_kwidth4) {
  auto parentMfma16 = mfma(/*warps=*/{2, 4}, /*mDim=*/16, /*nDim=*/16,
                           /*isTransposed=*/false);
  auto mfmaDotOp1_16 = mfmaDotOp(parentMfma16, /*opIdx=*/1, /*kWidth=*/4);
  EXPECT_EQ(toLinearLayout({128, 128}, mfmaDotOp1_16),
            LinearLayout(
                {{S("register"),
                  {{1, 0}, {2, 0}, {16, 0}, {32, 0}, {64, 0}, {0, 64}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}},
                 {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 64}, mfmaDotOp1_16),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}, {16, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}},
                 {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({16, 16}, mfmaDotOp1_16),
            LinearLayout(
                {{S("register"), {{1, 0}, {2, 0}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}},
                 {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  // Dot operand based on transposed mfma layout has same layout as ordinary
  auto parentTMfma16 = mfma(/*warps=*/{2, 4}, /*mDim=*/16, /*nDim=*/16,
                            /*isTransposed=*/true);
  auto tmfmaDotOp1_16 = mfmaDotOp(parentTMfma16, /*opIdx=*/1, /*kWidth=*/4);

  EXPECT_EQ(toLinearLayout({128, 128}, tmfmaDotOp1_16),
            toLinearLayout({128, 128}, mfmaDotOp1_16));
  EXPECT_EQ(toLinearLayout({64, 32}, tmfmaDotOp1_16),
            toLinearLayout({64, 32}, mfmaDotOp1_16));
  EXPECT_EQ(toLinearLayout({16, 16}, tmfmaDotOp1_16),
            toLinearLayout({16, 16}, mfmaDotOp1_16));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v1_2x4Warps) {
  auto legacy = wmma(/*warps=*/{2, 4}, /*version=*/1, /*transposed=*/false);

  EXPECT_EQ(toLinearLayout({16, 16}, legacy),
            LinearLayout({{S("register"), {{2, 0}, {4, 0}, {8, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  // For 32x16, we need 2x1 WMMA instances. We have 2x4 warps, so we are
  // broadcasted along the warp N dimension, distributed along the warp M
  // dimension.
  EXPECT_EQ(toLinearLayout({32, 16}, legacy),
            LinearLayout({{S("register"), {{2, 0}, {4, 0}, {8, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  // For 16x32, we need 1x2 WMMA instances. We have 2x4 warps, so along the warp
  // N dimension, warp 0/2 gets the first distributed instance, warp 1/3 gets
  // the second distributed instance. Along the warp M dimension, all are
  // broadcasted.
  EXPECT_EQ(toLinearLayout({16, 32}, legacy),
            LinearLayout({{S("register"), {{2, 0}, {4, 0}, {8, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                          {S("warp"), {{0, 16}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  // For 128x128, we need 8x8 WMMA instances. Given that we have 2x4 warps, each
  // warp handles 4x2 instances. So for both the warp M and N dimension, we
  // distribute. The register dimension will handle (8 x 4x2 =) 64 values--those
  // additional base vectors after the intrinsic shape are next power of two
  // values following the warp dimension, given that we are tiling cyclically
  // among warps.
  EXPECT_EQ(toLinearLayout({128, 128}, legacy),
            LinearLayout({{S("register"),
                           {{2, 0}, {4, 0}, {8, 0}, {0, 64}, {32, 0}, {64, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                          {S("warp"), {{0, 16}, {0, 32}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v1_2x4x1Warps) {
  auto legacy = wmma(/*warps=*/{2, 4, 1}, /*version=*/1, /*transposed=*/false);

  EXPECT_EQ(
      toLinearLayout({1, 16, 16}, legacy),
      LinearLayout(
          {{S("register"), {{0, 2, 0}, {0, 4, 0}, {0, 8, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 1, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 16, 16}, legacy),
      LinearLayout(
          {{S("register"), {{0, 2, 0}, {0, 4, 0}, {0, 8, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 1, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({8, 16, 16}, legacy),
      LinearLayout(
          {{S("register"),
            {{0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {2, 0, 0}, {4, 0, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 1, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v1_2x4Warps_lhs) {
  auto dot = wmma(/*warps=*/{2, 4}, /*version=*/1, /*transposed=*/false);
  auto wmmaOperand = wmmaDotOp(dot, 0, 16);

  EXPECT_EQ(toLinearLayout({16, 16}, wmmaOperand),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 16}, wmmaOperand),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 64}, wmmaOperand),
            LinearLayout({{S("register"),
                           {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {0, 32}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, wmmaOperand),
            LinearLayout({{S("register"),
                           {{0, 1},
                            {0, 2},
                            {0, 4},
                            {0, 8},
                            {0, 16},
                            {0, 32},
                            {0, 64},
                            {32, 0}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v1_2x4Warps_rhs) {
  auto dot = wmma(/*warps=*/{2, 4}, /*version=*/1, /*transposed=*/false);
  auto wmmaOperand = wmmaDotOp(dot, 1, 16);

  EXPECT_EQ(toLinearLayout({16, 16}, wmmaOperand),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({32, 16}, wmmaOperand),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 0}}},
                    {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({32, 64}, wmmaOperand),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 0}}},
                    {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, wmmaOperand),
            LinearLayout(
                {{S("register"),
                  {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {32, 0}, {0, 64}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 0}}},
                 {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v1_2x4x1Warps_lhs) {
  auto dot = wmma(/*warps=*/{2, 4, 1}, /*version=*/1, /*transposed=*/false);
  auto wmmaOperand = wmmaDotOp(dot, 0, 16);

  EXPECT_EQ(
      toLinearLayout({1, 16, 16}, wmmaOperand),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 32, 16}, wmmaOperand),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 0}}},
           {S("warp"), {{0, 16, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 64, 16}, wmmaOperand),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 0}}},
           {S("warp"), {{0, 16, 0}, {0, 32, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({4, 128, 32}, wmmaOperand),
      LinearLayout(
          {{S("register"),
            {{0, 0, 1},
             {0, 0, 2},
             {0, 0, 4},
             {0, 0, 8},
             {0, 0, 16},
             {0, 64, 0},
             {2, 0, 0}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 0}}},
           {S("warp"), {{0, 16, 0}, {0, 32, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v1_2x4x1Warps_rhs) {
  auto dot = wmma(/*warps=*/{2, 4, 1}, /*version=*/1, /*transposed=*/false);
  auto wmmaOperand = wmmaDotOp(dot, 1, 16);

  EXPECT_EQ(
      toLinearLayout({1, 16, 16}, wmmaOperand),
      LinearLayout(
          {{S("register"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 0, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 32, 16}, wmmaOperand),
      LinearLayout(
          {{S("register"),
            {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 16, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 0, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 64, 16}, wmmaOperand),
      LinearLayout(
          {{S("register"),
            {{0, 1, 0},
             {0, 2, 0},
             {0, 4, 0},
             {0, 8, 0},
             {0, 16, 0},
             {0, 32, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 0, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({4, 128, 32}, wmmaOperand),
      LinearLayout(
          {{S("register"),
            {{0, 1, 0},
             {0, 2, 0},
             {0, 4, 0},
             {0, 8, 0},
             {0, 16, 0},
             {0, 32, 0},
             {0, 64, 0},
             {0, 0, 16},
             {2, 0, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 0, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v2_2x4Warps) {
  auto layout = wmma(/*warps=*/{2, 4}, /*version=*/2, /*transposed=*/false);

  EXPECT_EQ(toLinearLayout({16, 16}, layout),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 16}, layout),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({16, 32}, layout),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 16}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({64, 128}, layout),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {0, 64}, {32, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                    {S("warp"), {{0, 16}, {0, 32}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v2_2x2x2Warps) {
  auto layout = wmma(/*warps=*/{2, 2, 2}, /*version=*/2, /*transposed=*/false);

  EXPECT_EQ(
      toLinearLayout({1, 16, 16}, layout),
      LinearLayout(
          {{S("register"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 8, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 16, 16}, layout),
      LinearLayout(
          {{S("register"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 8, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({4, 64, 64}, layout),
      LinearLayout(
          {{S("register"),
            {{0, 1, 0},
             {0, 2, 0},
             {0, 4, 0},
             {0, 0, 32},
             {0, 32, 0},
             {2, 0, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 8, 0}}},
           {S("warp"), {{0, 0, 16}, {0, 16, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, TWMMA_v2_2x4Warps) {
  auto layout = wmma(/*warps=*/{2, 4}, /*version=*/2, /*transposed=*/true);

  EXPECT_EQ(toLinearLayout({16, 16}, layout),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 16}, layout),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({16, 32}, layout),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                          {S("warp"), {{0, 16}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({64, 128}, layout),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 64}, {32, 0}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                    {S("warp"), {{0, 16}, {0, 32}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, TWMMA_v2_2x2x2Warps) {
  auto layout = wmma(/*warps=*/{2, 2, 2}, /*version=*/2, /*transposed=*/true);

  EXPECT_EQ(
      toLinearLayout({1, 16, 16}, layout),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 8}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 16, 16}, layout),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 8}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({4, 64, 64}, layout),
      LinearLayout(
          {{S("register"),
            {{0, 0, 1},
             {0, 0, 2},
             {0, 0, 4},
             {0, 0, 32},
             {0, 32, 0},
             {2, 0, 0}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 8}}},
           {S("warp"), {{0, 0, 16}, {0, 16, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v2_2x4Warps_lhs) {
  auto dot = wmma(/*warps=*/{2, 4}, /*version=*/2, /*transposed=*/false);

  auto wmmaOperandK8 = wmmaDotOp(dot, 0, 8);
  EXPECT_EQ(toLinearLayout({16, 16}, wmmaOperandK8),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 16}, wmmaOperandK8),
            LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                          {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                          {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({32, 64}, wmmaOperandK8),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 16}, {0, 32}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                    {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, wmmaOperandK8),
            LinearLayout(
                {{S("register"),
                  {{0, 1}, {0, 2}, {0, 4}, {0, 16}, {0, 32}, {0, 64}, {32, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}}},
                 {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));

  auto wmmaOperandK16 = wmmaDotOp(dot, 0, 16);
  EXPECT_EQ(
      toLinearLayout({16, 32}, wmmaOperandK16),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 16}}},
                    {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({32, 32}, wmmaOperandK16),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 16}}},
                    {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({32, 128}, wmmaOperandK16),
      LinearLayout(
          {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 32}, {0, 64}}},
           {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 16}}},
           {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, wmmaOperandK16),
            LinearLayout(
                {{S("register"),
                  {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 32}, {0, 64}, {32, 0}}},
                 {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 16}}},
                 {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v2_2x4Warps_rhs) {
  auto dot = wmma(/*warps=*/{2, 4}, /*version=*/2, /*transposed=*/false);

  auto wmmaOperandK8 = wmmaDotOp(dot, 1, 8);
  EXPECT_EQ(toLinearLayout({16, 16}, wmmaOperandK8),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 16}, wmmaOperandK8),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {16, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 64}, wmmaOperandK8),
            LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {16, 0}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, wmmaOperandK8),
            LinearLayout({{S("register"),
                           {{1, 0}, {2, 0}, {4, 0}, {16, 0}, {32, 0}, {0, 64}}},
                          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}}},
                          {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));

  auto wmmaOperandK16 = wmmaDotOp(dot, 1, 16);
  EXPECT_EQ(
      toLinearLayout({32, 16}, wmmaOperandK16),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {16, 0}}},
                    {S("warp"), {{0, 0}, {0, 0}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({32, 32}, wmmaOperandK16),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {16, 0}}},
                    {S("warp"), {{0, 16}, {0, 0}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({64, 64}, wmmaOperandK16),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {32, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {16, 0}}},
                    {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({128, 128}, wmmaOperandK16),
            LinearLayout(
                {{S("register"),
                  {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {32, 0}, {64, 0}, {0, 64}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {16, 0}}},
                 {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v2_2x4x1Warps_lhs) {
  auto dot = wmma(/*warps=*/{2, 4, 1}, /*version=*/2, /*transposed=*/false);
  auto wmmaOperandK8 = wmmaDotOp(dot, 0, 8);

  EXPECT_EQ(
      toLinearLayout({1, 16, 16}, wmmaOperandK8),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 8}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 32, 16}, wmmaOperandK8),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 8}}},
           {S("warp"), {{0, 16, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 64, 16}, wmmaOperandK8),
      LinearLayout(
          {{S("register"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 8}}},
           {S("warp"), {{0, 16, 0}, {0, 32, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({4, 128, 32}, wmmaOperandK8),
      LinearLayout(
          {{S("register"),
            {{0, 0, 1},
             {0, 0, 2},
             {0, 0, 4},
             {0, 0, 16},
             {0, 64, 0},
             {2, 0, 0}}},
           {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}, {0, 0, 8}}},
           {S("warp"), {{0, 16, 0}, {0, 32, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, WMMA_v2_2x4x1Warps_rhs) {
  auto dot = wmma(/*warps=*/{2, 4, 1}, /*version=*/2, /*transposed=*/false);
  auto wmmaOperandK8 = wmmaDotOp(dot, 1, 8);

  EXPECT_EQ(
      toLinearLayout({1, 16, 16}, wmmaOperandK8),
      LinearLayout(
          {{S("register"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 8, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 32, 16}, wmmaOperandK8),
      LinearLayout(
          {{S("register"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 16, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 8, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({2, 64, 16}, wmmaOperandK8),
      LinearLayout(
          {{S("register"),
            {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 16, 0}, {0, 32, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 8, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(
      toLinearLayout({4, 128, 32}, wmmaOperandK8),
      LinearLayout(
          {{S("register"),
            {{0, 1, 0},
             {0, 2, 0},
             {0, 4, 0},
             {0, 16, 0},
             {0, 32, 0},
             {0, 64, 0},
             {0, 0, 16},
             {2, 0, 0}}},
           {S("lane"), {{0, 0, 1}, {0, 0, 2}, {0, 0, 4}, {0, 0, 8}, {0, 8, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, SliceOfBlocked) {
  auto parent = blocked({2, 4}, {4, 2}, {2, 2}, {2, 2}, {2, 2}, {1, 0}, {1, 0});
  EXPECT_EQ(toLinearLayout({128}, slice(parent, 0)),
            LinearLayout({{S("register"), {{1}, {2}, {16}, {32}}},
                          {S("lane"), {{4}, {0}, {0}}},
                          {S("warp"), {{8}, {0}}},
                          {S("block"), {{64}, {0}}}},
                         {S("dim0")}));
  EXPECT_EQ(toLinearLayout({128}, slice(parent, 1)),
            LinearLayout({{S("register"), {{1}, {16}, {32}}},
                          {S("lane"), {{0}, {2}, {4}}},
                          {S("warp"), {{0}, {8}}},
                          {S("block"), {{0}, {64}}}},
                         {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, SliceWithShape1) {
  auto parent = blocked({1, 4}, {8, 4}, {2, 2}, {1, 1}, {1, 1}, {0, 1}, {1, 0});
  EXPECT_EQ(toLinearLayout({1}, slice(parent, 0)),
            LinearLayout({{S("register"), {{0}, {0}}},
                          {S("lane"), {{0}, {0}, {0}, {0}, {0}}},
                          {S("warp"), {{0}, {0}}},
                          {S("block"), {}}},
                         {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, Slice4D) {
  auto parent = blocked({1, 1, 1, 4}, {2, 1, 1, 16}, {1, 2, 4, 1}, {1, 1, 1, 1},
                        {1, 1, 1, 1}, {3, 0, 1, 2}, {3, 2, 1, 0});
  EXPECT_EQ(toLinearLayout({2, 1, 1}, slice(parent, 3)),
            LinearLayout(
                {
                    {S("register"), {}},
                    {S("lane"),
                     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
                    {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, SliceOfMmaV2) {
  auto parent = mma(2, 0, {16, 8}, {2, 2}, {1, 1}, {1, 1}, {0, 1});
  EXPECT_EQ(toLinearLayout({16}, slice(parent, 0)),
            LinearLayout({{S("register"), {{1}}},
                          {S("lane"), {{2}, {4}, {0}, {0}, {0}}},
                          {S("warp"), {{8}, {0}}},
                          {S("block"), {}}},
                         {S("dim0")}));
  EXPECT_EQ(toLinearLayout({128}, slice(parent, 0)),
            LinearLayout({{S("register"), {{1}, {16}, {32}, {64}}},
                          {S("lane"), {{2}, {4}, {0}, {0}, {0}}},
                          {S("warp"), {{8}, {0}}},
                          {S("block"), {}}},
                         {S("dim0")}));
  EXPECT_EQ(toLinearLayout({8}, slice(parent, 1)),
            LinearLayout({{S("register"), {{0}}},
                          {S("lane"), {{0}, {0}, {1}, {2}, {4}}},
                          {S("warp"), {{0}, {0}}},
                          {S("block"), {}}},
                         {S("dim0")}));
  EXPECT_EQ(toLinearLayout({128}, slice(parent, 1)),
            LinearLayout({{S("register"), {{8}, {32}, {64}}},
                          {S("lane"), {{0}, {0}, {1}, {2}, {4}}},
                          {S("warp"), {{0}, {16}}},
                          {S("block"), {}}},
                         {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSimple1D) {
  EXPECT_EQ(toLinearLayout({1024}, shared(1, 1, 1, {1}, {1}, {0}, {0})),
            LinearLayout::identity1D(1024, S("offset"), S("dim0")) *
                LinearLayout::identity1D(1, S("block"), S("dim0")));
}

TEST_F(LinearLayoutConversionsTest, SharedSimple2D) {
  EXPECT_EQ(toLinearLayout({128, 128},
                           shared(1, 1, 1, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
            (LinearLayout::identity1D(128, S("offset"), S("dim1")) *
             LinearLayout::identity1D(128, S("offset"), S("dim0")) *
             LinearLayout::identity1D(1, S("block"), S("dim0")))
                .transposeOuts({S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSimple2D_Order01) {
  EXPECT_EQ(toLinearLayout({128, 128},
                           shared(1, 1, 1, {1, 1}, {1, 1}, {0, 1}, {1, 0})),
            LinearLayout::identity1D(128, S("offset"), S("dim0")) *
                LinearLayout::identity1D(128, S("offset"), S("dim1")) *
                LinearLayout::identity1D(1, S("block"), S("dim0")));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_MaxPhaseOnly) {
  EXPECT_EQ(
      toLinearLayout({32, 32}, shared(1, 1, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 1},
                      {0, 2},
                      {0, 4},
                      {0, 8},
                      {0, 16},
                      {1, 1},
                      {2, 2},
                      {4, 0},
                      {8, 0},
                      {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_PerPhaseMaxPhase) {
  EXPECT_EQ(
      toLinearLayout({32, 32}, shared(1, 2, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 1},
                      {0, 2},
                      {0, 4},
                      {0, 8},
                      {0, 16},
                      {1, 0},
                      {2, 1},
                      {4, 2},
                      {8, 0},
                      {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_Vec) {
  EXPECT_EQ(
      toLinearLayout({4, 8}, shared(2, 1, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"), {{0, 1}, {0, 2}, {0, 4}, {1, 2}, {2, 4}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_PerPhaseMaxPhaseVec) {
  EXPECT_EQ(
      toLinearLayout({32, 32}, shared(2, 2, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 1},
                      {0, 2},
                      {0, 4},
                      {0, 8},
                      {0, 16},
                      {1, 0},
                      {2, 2},
                      {4, 4},
                      {8, 0},
                      {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled4D) {
  EXPECT_EQ(
      toLinearLayout({2, 4, 32, 32}, shared(2, 2, 4, {1, 1, 1, 1}, {1, 1, 1, 1},
                                            {3, 2, 1, 0}, {3, 2, 1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 0, 0, 1},
                      {0, 0, 0, 2},
                      {0, 0, 0, 4},
                      {0, 0, 0, 8},
                      {0, 0, 0, 16},
                      {0, 0, 1, 0},
                      {0, 0, 2, 2},
                      {0, 0, 4, 4},
                      {0, 0, 8, 0},
                      {0, 0, 16, 0},
                      {0, 1, 0, 0},
                      {0, 2, 0, 0},
                      {1, 0, 0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1"), S("dim2"), S("dim3")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_Order01) {
  EXPECT_EQ(
      toLinearLayout({4, 8}, shared(1, 1, 4, {1, 1}, {1, 1}, {0, 1}, {0, 1})),
      LinearLayout({{S("offset"), {{1, 0}, {2, 0}, {1, 1}, {2, 2}, {0, 4}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_8x16_4_2) {
  EXPECT_EQ(
      toLinearLayout(
          {8, 16}, nvmmaShared(32, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}, {4, 8}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_128x16_4_2) {
  EXPECT_EQ(toLinearLayout({128, 16}, nvmmaShared(32, false, 16, {1, 1}, {1, 1},
                                                  {1, 0}, {1, 0})),
            LinearLayout({{S("offset"),
                           {{0, 1},
                            {0, 2},
                            {0, 4},
                            {0, 8},
                            {1, 0},
                            {2, 0},
                            {4, 8},
                            {8, 0},
                            {16, 0},
                            {32, 0},
                            {64, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_8x32_2_4) {
  EXPECT_EQ(
      toLinearLayout(
          {8, 32}, nvmmaShared(64, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout(
          {{S("offset"),
            {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {1, 0}, {2, 8}, {4, 16}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_8x64_1_8) {
  EXPECT_EQ(toLinearLayout({8, 64}, nvmmaShared(128, false, 16, {1, 1}, {1, 1},
                                                {1, 0}, {1, 0})),
            LinearLayout({{S("offset"),
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
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_8x64_1_8_32b) {
  EXPECT_EQ(toLinearLayout({8, 64}, nvmmaShared(128, false, 32, {1, 1}, {1, 1},
                                                {1, 0}, {1, 0})),
            LinearLayout({{S("offset"),
                           {{0, 1},
                            {0, 2},
                            {0, 4},
                            {0, 8},
                            {0, 16},
                            {1, 4},
                            {2, 8},
                            {4, 16},
                            {0, 32}}},
                          {S("block"), {}}},
                         {{S("dim0"), 8}, {S("dim1"), 64}},
                         /*requireSurjective=*/false));
}

TEST_F(LinearLayoutConversionsTest, Shared1DSwizzle) {
  EXPECT_EQ(
      toLinearLayout({64, 1}, shared(2, 2, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout::identity1D(64, S("offset"), S("dim0")) *
          LinearLayout::identity1D(1, S("offset"), S("dim1")) *
          LinearLayout::identity1D(1, S("block"), S("dim0")));
}

TEST_F(LinearLayoutConversionsTest, ChooseShmemLayout) {
  LinearLayout ll = LinearLayout({{S("register"), {{1}, {2}, {2}, {8}}},
                                  {S("lane"), {{8}, {4}, {1}}},
                                  {S("warp"), {{16}, {32}, {0}}},
                                  {S("block"), {}}},
                                 {S("dim0")});
  EXPECT_EQ(chooseShemLayoutForRegToRegConversion(&ctx, /*tensorShape=*/{64},
                                                  /*repShape=*/{64},
                                                  /*order=*/{0}),
            LinearLayout({{S("offset"), {{1}, {2}, {4}, {8}, {16}, {32}}},
                          {S("iteration"), {}},
                          {S("block"), {}}},
                         {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, ChooseShmemLayout_Empty) {
  LinearLayout ll = LinearLayout({{S("register"), {{0}}},
                                  {S("lane"), {{0}}},
                                  {S("warp"), {{0}}},
                                  {S("block"), {}}},
                                 {S("dim0")});
  EXPECT_EQ(
      chooseShemLayoutForRegToRegConversion(&ctx, /*tensorShape=*/{},
                                            /*repShape=*/{}, /*order=*/{}),
      LinearLayout({{S("offset"), {}}, {S("iteration"), {}}, {S("block"), {}}},
                   {}));
}

TEST_F(LinearLayoutConversionsTest, ChooseShmemLayout_Multidim) {
  LinearLayout src(
      {{S("register"), {}},
       {S("lane"),
        {{0, 0, 1, 0}, {0, 0, 2, 0}, {1, 0, 0, 0}, {2, 0, 0, 0}, {0, 0, 0, 1}}},
       {S("warp"), {{0, 0, 0, 2}, {0, 1, 0, 0}, {0, 2, 0, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1"), S("dim2"), S("dim3")});
  EXPECT_EQ(
      chooseShemLayoutForRegToRegConversion(&ctx, /*tensorShape=*/{4, 4, 4, 4},
                                            /*repShape=*/{2, 2, 2, 2},
                                            /*order=*/{3, 2, 1, 0}),
      LinearLayout({{S("offset"),
                     {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}},
                    {S("iteration"),
                     {{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 2}}},
                    {S("block"), {}}},
                   {S("dim3"), S("dim2"), S("dim1"), S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv5Fp4Padded) {
  auto ll = toLinearLayout({32, 64}, nvmmaShared(128, false, 8, {1, 1}, {1, 1},
                                                 {1, 0}, {1, 0}, true));
  EXPECT_EQ(ll, LinearLayout(
                    {{S("offset"),
                      {{0, 1},
                       {0, 2},
                       {0, 4},
                       {0, 0}, // offset 8 maps to the same indices as offset 0
                       {0, 8},
                       {0, 16},
                       {0, 32},
                       {1, 8},
                       {2, 16},
                       {4, 32},
                       {8, 0},
                       {16, 0}}},
                     {S("block"), {}}},
                    {S("dim0"), S("dim1")}));
}

} // anonymous namespace
} // namespace mlir::triton::gpu

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
