#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

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

  SliceEncodingAttr slice(Attribute parent, int dim) {
    return SliceEncodingAttr::get(&ctx, dim, parent);
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
  EXPECT_EQ(toLinearLayout({64, 16}, mma(3, 0, {16, 16, 8}, {4, 1}, {1, 1},
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
            LinearLayout({{S("register"), {{4}}},
                          {S("lane"), {{0}, {0}, {1}, {2}, {0}}},
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

} // anonymous namespace
} // namespace mlir::triton::gpu
