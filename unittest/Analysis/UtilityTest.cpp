#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "llvm/Support/Signals.h"
#include <gtest/gtest.h>

namespace mlir {
namespace {

class ReduceOpHelperTest : public ::testing::Test {
protected:
  StringAttr S(StringRef name) { return StringAttr::get(&ctx, name); }

  MLIRContext ctx;
};

TEST(Analysis, reorder) {
  SmallVector<int> shape({10, 20, 30});
  {
    SmallVector<unsigned> order({2, 1, 0});
    auto reordered = triton::applyPermutation(shape, order);
    EXPECT_EQ(reordered[0], 30);
    EXPECT_EQ(reordered[1], 20);
    EXPECT_EQ(reordered[2], 10);
  }
  {
    SmallVector<unsigned> order({1, 0, 2});
    auto reordered = triton::applyPermutation(shape, order);
    EXPECT_EQ(reordered[0], 20);
    EXPECT_EQ(reordered[1], 10);
    EXPECT_EQ(reordered[2], 30);
  }
}

TEST_F(ReduceOpHelperTest, MoveAxisBasesToFront) {
  auto reg = S("register");
  auto x = S("x");
  auto y = S("y");
  triton::LinearLayout layout({{reg, {{0, 1}, {1, 0}, {0, 2}, {2, 0}}}},
                              {{x, 4}, {y, 4}},
                              /*requireSurjective=*/false);

  auto action = ReduceOpHelper::moveAxisBasesToFront(layout, /*axis=*/0);
  EXPECT_EQ(action.apply(layout),
            triton::LinearLayout({{reg, {{1, 0}, {2, 0}, {0, 1}, {0, 2}}}},
                                 {{x, 4}, {y, 4}},
                                 /*requireSurjective=*/false));

  auto vectorized = ReduceOpHelper::moveAxisBasesToFront(layout, /*axis=*/0,
                                                         /*isVectorized=*/true);
  EXPECT_EQ(vectorized.apply(layout),
            triton::LinearLayout({{reg, {{0, 1}, {1, 0}, {2, 0}, {0, 2}}}},
                                 {{x, 4}, {y, 4}},
                                 /*requireSurjective=*/false));
}

TEST_F(ReduceOpHelperTest, GetInterLayoutUsesFreeLanes) {
  auto lane = S("lane");
  auto warp = S("warp");
  auto block = S("block");
  auto x = S("x");
  auto y = S("y");
  triton::LinearLayout layout(
      {{lane, {{0, 1}, {0, 0}, {0, 0}}}, {warp, {{1, 0}}}, {block, {{2, 0}}}},
      {{x, 4}, {y, 2}}, /*requireSurjective=*/false);

  EXPECT_EQ(ReduceOpHelper::getInterLayout(layout, /*axis=*/0),
            triton::LinearLayout({{lane, {{0, 1}, {1, 0}, {2, 0}}},
                                  {warp, {{0, 0}}},
                                  {block, {{0, 0}}}},
                                 {{x, 4}, {y, 2}},
                                 /*requireSurjective=*/false));
}

TEST_F(ReduceOpHelperTest, GetInterLayoutReplacesOccupiedLanes) {
  auto lane = S("lane");
  auto warp = S("warp");
  auto block = S("block");
  auto x = S("x");
  auto y = S("y");
  triton::LinearLayout layout(
      {{lane, {{0, 1}, {0, 2}}}, {warp, {{1, 0}}}, {block, {{2, 0}}}},
      {{x, 4}, {y, 4}},
      /*requireSurjective=*/false);

  EXPECT_EQ(ReduceOpHelper::getInterLayout(layout, /*axis=*/0),
            triton::LinearLayout(
                {{lane, {{1, 0}, {2, 0}}}, {warp, {{0, 1}}}, {block, {{0, 2}}}},
                {{x, 4}, {y, 4}},
                /*requireSurjective=*/false));
}

TEST_F(ReduceOpHelperTest, GetInterLayoutPrioritizesWarps) {
  auto lane = S("lane");
  auto warp = S("warp");
  auto block = S("block");
  auto x = S("x");
  auto y = S("y");
  triton::LinearLayout layout({{lane, {{0, 1}, {0, 2}}},
                               {warp, {{1, 0}, {2, 0}, {4, 0}}},
                               {block, {{8, 0}}}},
                              {{x, 16}, {y, 4}}, /*requireSurjective=*/false);

  EXPECT_EQ(ReduceOpHelper::getInterLayout(layout, /*axis=*/0),
            triton::LinearLayout({{lane, {{1, 0}, {2, 0}}},
                                  {warp, {{0, 1}, {0, 2}, {4, 0}}},
                                  {block, {{8, 0}}}},
                                 {{x, 16}, {y, 4}},
                                 /*requireSurjective=*/false));
}

} // namespace
} // namespace mlir

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
