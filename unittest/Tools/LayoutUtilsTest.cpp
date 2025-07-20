#include "triton/Tools/LayoutUtils.h"

#include "mlir/Support/LLVM.h"
#include "llvm/Support/Signals.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir::triton {
namespace {

class LayoutUtilsTest : public ::testing::Test {
public:
  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(LayoutUtilsTest, SquareSublayoutIsIdentity) {
  EXPECT_TRUE(squareSublayoutIsIdentity(
      LinearLayout::identity1D(4, S("in"), S("in")), {S("in")}));
  EXPECT_TRUE(squareSublayoutIsIdentity(
      LinearLayout::identity1D(4, S("in"), S("in")), {}));

  LinearLayout l1(
      {{S("in1"), {{1, 1}, {2, 2}, {4, 4}}}, {S("in2"), {{2, 1}, {1, 2}}}},
      {{S("in1"), 8}, {S("in2"), 8}}, /*requireSurjective=*/false);
  EXPECT_TRUE(squareSublayoutIsIdentity(l1, {S("in1")}));
  EXPECT_FALSE(squareSublayoutIsIdentity(l1, {S("in2")}));

  LinearLayout l2 = LinearLayout::identity1D(4, S("in1"), S("in1")) *
                    LinearLayout::identity1D(8, S("in2"), S("in2")) *
                    LinearLayout({{S("in3"), {{1, 1, 1}}}},
                                 {{S("in1"), 2}, {S("in2"), 2}, {S("in3"), 2}},
                                 /*requireSurjective=*/false);
  EXPECT_FALSE(squareSublayoutIsIdentity(l2, {S("in1")}));
  EXPECT_FALSE(squareSublayoutIsIdentity(l2, {S("in2")}));
  EXPECT_TRUE(squareSublayoutIsIdentity(l2, {S("in3")}));
  EXPECT_FALSE(squareSublayoutIsIdentity(l2, {S("in1"), S("in2")}));

  LinearLayout l3 = LinearLayout::identity1D(4, S("in1"), S("in1")) *
                    LinearLayout::identity1D(8, S("in2"), S("in2"));
  EXPECT_TRUE(squareSublayoutIsIdentity(l3, {S("in1")}));
  EXPECT_TRUE(squareSublayoutIsIdentity(l3, {S("in2")}));
  EXPECT_TRUE(squareSublayoutIsIdentity(l3, {S("in1"), S("in2")}));
}

TEST_F(LayoutUtilsTest, BasisPermutationLayout) {
  LinearLayout src1(
      {{S("in1"), {{1, 0}, {0, 0}, {0, 2}}}, {S("in2"), {{2, 0}, {0, 1}}}},
      {S("out1"), S("out2")});
  LinearLayout dst1(
      {{S("in2"), {{1, 0}, {0, 0}}}, {S("in1"), {{2, 0}, {0, 1}, {0, 2}}}},
      {S("out2"), S("out1")});
  LinearLayout P1(
      {{S("in1"), {{2, 0}, {0, 2}, {1, 0}}}, {S("in2"), {{4, 0}, {0, 1}}}},
      {S("in1"), S("in2")});
  EXPECT_EQ(P1, basisPermutationLayout(src1, dst1));
  EXPECT_EQ(src1, reorder_like(P1.compose(dst1), src1));
  LinearLayout src2({{S("in3"), {{2, 0}, {4, 0}, {8, 0}, {0, 0}}},
                     {S("in2"), {{0, 0}, {16, 0}, {0, 0}, {0, 1}}},
                     {S("in1"), {{0, 2}, {0, 0}, {0, 4}}}},
                    {{S("out1"), 32}, {S("out2"), 8}},
                    /*requireSurjective=*/false);
  LinearLayout dst2({{S("in1"), {{0, 0}, {0, 16}, {2, 0}}},
                     {S("in2"), {{0, 4}, {0, 8}, {0, 0}, {4, 0}}},
                     {S("in3"), {{0, 0}, {0, 0}, {0, 2}, {1, 0}}}},
                    {{S("out2"), 8}, {S("out1"), 32}},
                    /*requireSurjective=*/false);
  LinearLayout P2({{S("in3"), {{4, 0, 0}, {0, 1, 0}, {0, 2, 0}, {1, 0, 0}}},
                   {S("in2"), {{2, 0, 0}, {0, 0, 2}, {0, 4, 0}, {8, 0, 0}}},
                   {S("in1"), {{0, 0, 4}, {0, 0, 1}, {0, 8, 0}}}},
                  {S("in3"), S("in2"), S("in1")});
  EXPECT_EQ(P2, basisPermutationLayout(src2, dst2));
  EXPECT_EQ(src2, reorder_like(P2.compose(dst2), src2));
}

} // namespace
} // namespace mlir::triton
