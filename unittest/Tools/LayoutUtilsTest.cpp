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

} // namespace
} // namespace mlir::triton
