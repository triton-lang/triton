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

TEST_F(LayoutUtilsTest, OutputBasisMask) {
  LinearLayout layout({{S("x"), {{1, 8}, {2, 16}}},
                       {S("y"), {{4, 1}}},
                       {S("broadcast"), {{0, 0}}}},
                      {{S("out0"), 8}, {S("out1"), 32}},
                      /*requireSurjective=*/false);

  EXPECT_EQ(getOutputBasisMask(layout, {S("x")}, S("out0")), 0b0011);
  EXPECT_EQ(getOutputBasisMask(layout, {S("x"), S("y")}, S("out0")), 0b0111);
  EXPECT_EQ(getOutputBasisMask(layout, {S("x"), S("broadcast")}, S("out1")),
            0b11000);
  EXPECT_EQ(getOutputBasisMask(layout, {}, S("out0")), 0);
}

TEST_F(LayoutUtilsTest, InputBasisMask) {
  LinearLayout layout(
      {{S("x"), {{1, 0}, {0, 2}, {0, 0}, {4, 8}}}, {S("y"), {{2, 0}}}},
      {{S("out0"), 8}, {S("out1"), 16}},
      /*requireSurjective=*/false);

  EXPECT_EQ(getInputBasisMask(layout, S("x"), {S("out0")}), 0b1001);
  EXPECT_EQ(getInputBasisMask(layout, S("x"), {S("out1")}), 0b1010);
  EXPECT_EQ(getInputBasisMask(layout, S("x"), {S("out0"), S("out1")}), 0b1011);
  EXPECT_EQ(getInputBasisMask(layout, S("x"), {}), 0);
}

TEST_F(LayoutUtilsTest, FactorMaximalIdentityPrefix) {
  LinearLayout layout({{S("x"), {{1}, {2}, {8}}}, {S("other"), {{4}}}},
                      {{S("out"), 16}}, /*requireSurjective=*/false);

  auto [size, quotient] =
      factorMaximalIdentityPrefix(layout, S("x"), S("out"), 8);
  EXPECT_EQ(size, 4);
  EXPECT_EQ(LinearLayout::identity1D(size, S("x"), S("out")) * quotient,
            layout);
}

TEST_F(LayoutUtilsTest, RenameDimsPreservesOrder) {
  LinearLayout layout({{S("a"), {{1, 0}}}, {S("b"), {{0, 1}}}},
                      {{S("x"), 2}, {S("y"), 2}}, /*requireSurjective=*/true);

  auto renamed =
      renameLinearLayoutDims(layout, {{S("a"), S("b")}, {S("b"), S("a")}},
                             {{S("x"), S("y")}, {S("y"), S("x")}});
  EXPECT_THAT(llvm::to_vector(renamed.getInDimNames()),
              ::testing::ElementsAre(S("b"), S("a")));
  EXPECT_THAT(llvm::to_vector(renamed.getOutDimNames()),
              ::testing::ElementsAre(S("y"), S("x")));
  EXPECT_EQ(renamed, LinearLayout({{S("b"), {{1, 0}}}, {S("a"), {{0, 1}}}},
                                  {{S("y"), 2}, {S("x"), 2}},
                                  /*requireSurjective=*/true));

  EXPECT_THAT(
      llvm::to_vector(layout.renameInDim(S("a"), S("c")).getInDimNames()),
      ::testing::ElementsAre(S("c"), S("b")));
}

TEST_F(LayoutUtilsTest, GetRepsWithSubsetOfOutputDims) {
  LinearLayout cvt({{S("in"), {{1, 0}, {0, 1}}}}, {S("out0"), S("out1")});
  LinearLayout tile({{S("in"), {{1}}}}, {S("out0")});

  auto reps = getReps(cvt, tile);
  ASSERT_TRUE(reps);
  EXPECT_EQ(*reps, LinearLayout({{S("in"), {{0, 0}, {0, 1}}}},
                                {{S("out0"), 2}, {S("out1"), 2}},
                                /*requireSurjective=*/false));
}

} // namespace
} // namespace mlir::triton
