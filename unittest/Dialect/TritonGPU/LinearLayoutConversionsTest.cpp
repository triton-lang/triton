#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

#include "mlir/IR/MLIRContext.h"
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

  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(LinearLayoutConversionsTest, SimpleBlocked) {
  auto layout =
      toLinearLayout({16}, blocked({1}, {4}, {4}, {1}, {1}, {0}, {0}));
  EXPECT_THAT(layout, LinearLayout({
                          {S("register"), {{S("dim0"), {}}}},
                          {S("lane"), {{S("dim0"), {1, 2}}}},
                          {S("warp"), {{S("dim0"), {4, 8}}}},
                          {S("block"), {{S("dim0"), {}}}},
                      }));
}

TEST_F(LinearLayoutConversionsTest, CTADuplication) {
  auto layout = toLinearLayout(
      {32}, blocked({1}, {4}, {4}, /*cpg=*/{4}, /*cSplit=*/{2}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout({
                        {S("register"), {{S("dim0"), {}}}},
                        {S("lane"), {{S("dim0"), {1, 2}}}},
                        {S("warp"), {{S("dim0"), {4, 8}}}},
                        {S("block"), {{S("dim0"), {16, 0}}}},
                    }));
}

TEST_F(LinearLayoutConversionsTest, CTABroadcast) {
  auto layout =
      toLinearLayout({64, 128}, blocked({8, 1}, {8, 4}, {1, 4}, {1, 2}, {1, 2},
                                        {0, 1}, {1, 0}));
  EXPECT_EQ(
      layout,
      LinearLayout({
          {S("register"),
           {{S("dim0"), {1, 2, 4, 0, 0}}, {S("dim1"), {0, 0, 0, 16, 32}}}},
          {S("lane"),
           {{S("dim0"), {8, 16, 32, 0, 0}}, {S("dim1"), {0, 0, 0, 1, 2}}}},
          {S("warp"), {{S("dim0"), {0, 0}}, {S("dim1"), {4, 8}}}},
          {S("block"), {{S("dim0"), {0}}, {S("dim1"), {64}}}},
      }));
}

TEST_F(LinearLayoutConversionsTest, ShapeLargerThanLayout) {
  // The layout is 16 elements, but the shape is 128, so it's repeated 128/16 =
  // 8 times.
  auto layout =
      toLinearLayout({128}, blocked({1}, {4}, {4}, {1}, {1}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout({
                        {S("register"), {{S("dim0"), {16, 32, 64}}}},
                        {S("lane"), {{S("dim0"), {1, 2}}}},
                        {S("warp"), {{S("dim0"), {4, 8}}}},
                        {S("block"), {{S("dim0"), {}}}},
                    }));
}

TEST_F(LinearLayoutConversionsTest, ShapeLargerThanLayout2DDegenerate) {
  auto layout = toLinearLayout({128, 1}, blocked({1, 1}, {4, 1}, {4, 1}, {1, 1},
                                                 {1, 1}, {0, 1}, {1, 0}));
  EXPECT_EQ(layout, LinearLayout({
                        {S("register"),
                         {{S("dim0"), {16, 32, 64}}, {S("dim1"), {0, 0, 0}}}},
                        {S("lane"), {{S("dim0"), {1, 2}}, {S("dim1"), {0, 0}}}},
                        {S("warp"), {{S("dim0"), {4, 8}}, {S("dim1"), {0, 0}}}},
                        {S("block"), {{S("dim0"), {}}, {S("dim1"), {}}}},
                    }));
}

TEST_F(LinearLayoutConversionsTest, ShapeSmallerThanLayout) {
  // The shape is 8 elements, but the layout is 4*4*4 = 64 elems.  Therefore the
  // log2(64/8) = 3 most major bases are 0.
  auto layout = toLinearLayout({8}, blocked({4}, {4}, {4}, {1}, {1}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout({
                        {S("register"), {{S("dim0"), {1, 2}}}},
                        {S("lane"), {{S("dim0"), {4, 0}}}},
                        {S("warp"), {{S("dim0"), {0, 0}}}},
                        {S("block"), {{S("dim0"), {}}}},
                    }));
}

TEST_F(LinearLayoutConversionsTest, ReversedOrder) {
  auto layout = toLinearLayout({1, 64}, blocked({1, 1}, {32, 1}, {1, 8}, {1, 1},
                                                {1, 1}, {0, 1}, {1, 0}));
  EXPECT_EQ(
      layout,
      LinearLayout({
          {S("register"), {{S("dim0"), {0, 0, 0}}, {S("dim1"), {8, 16, 32}}}},
          {S("lane"),
           {{S("dim0"), {0, 0, 0, 0, 0}}, {S("dim1"), {0, 0, 0, 0, 0}}}},
          {S("warp"), {{S("dim0"), {0, 0, 0}}, {S("dim1"), {1, 2, 4}}}},
          {S("block"), {{S("dim0"), {}}, {S("dim1"), {}}}},
      }));
}

TEST_F(LinearLayoutConversionsTest, ReplicateInRegisterDim) {
  auto layout =
      toLinearLayout({32}, blocked({2}, {4}, {1}, {1}, {1}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout({
                        {S("register"), {{S("dim0"), {1, 8, 16}}}},
                        {S("lane"), {{S("dim0"), {2, 4}}}},
                        {S("warp"), {{S("dim0"), {}}}},
                        {S("block"), {{S("dim0"), {}}}},
                    }));
}

TEST_F(LinearLayoutConversionsTest, OneDimTooLargeAnotherTooSmall) {
  auto blockedLayout =
      blocked({1, 4}, {8, 4}, {4, 1}, {2, 2}, {2, 1}, {1, 0}, {1, 0});
  auto ll = toLinearLayout({128, 16}, blockedLayout);
  EXPECT_EQ(ll, LinearLayout({
                    {S("register"),
                     {
                         {S("dim0"), {0, 0, 32}},
                         {S("dim1"), {1, 2, 0}},
                     }},
                    {S("lane"),
                     {
                         {S("dim0"), {0, 0, 1, 2, 4}},
                         {S("dim1"), {4, 8, 0, 0, 0}},
                     }},
                    {S("warp"),
                     {
                         {S("dim0"), {8, 16}},
                         {S("dim1"), {0, 0}},
                     }},
                    {S("block"),
                     {
                         {S("dim0"), {0, 64}},
                         {S("dim1"), {0, 0}},
                     }},
                }));
}

TEST_F(LinearLayoutConversionsTest, RepeatInCTGDimFirst) {
  // We have a 4-element shape and an 8-element layout (4 elems per CTA).  So
  // the layout will map two inputs to each output.  The question is, which two
  // inputs?  The answer is, we split between CTAs first, so the two CTAs have
  // distinct elements.
  auto blockedLayout = blocked({1}, {1}, {4}, {2}, {2}, {0}, {0});
  auto ll = toLinearLayout({4}, blockedLayout);
  EXPECT_EQ(ll, LinearLayout({
                    {S("register"), {{S("dim0"), {}}}},
                    {S("lane"), {{S("dim0"), {}}}},
                    {S("warp"), {{S("dim0"), {1, 0}}}},
                    {S("block"), {{S("dim0"), {2}}}},
                }));
}

} // anonymous namespace
} // namespace mlir::triton::gpu
