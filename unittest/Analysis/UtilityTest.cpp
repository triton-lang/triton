#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Analysis/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

#include "mlir/Support/LLVM.h"
#include "llvm/Support/Signals.h"
#include <gtest/gtest.h>

namespace mlir {

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

TEST(Analysis, isCvtDimSync) {
  MLIRContext ctx;
  auto S = [&](StringRef str) { return StringAttr::get(&ctx, str); };

  auto srcLayout = triton::LinearLayout(
      {{S("register"), {}},
       {S("lane"), {{0, 1}, {1, 0}, {2, 0}, {4, 0}, {8, 0}}},
       {S("warp"), {{16, 0}, {32, 0}}},
       {S("block"), {{0, 0}, {64, 0}}}},
      {S("dim0"), S("dim1")});

  auto dstLayout = triton::LinearLayout(
      {{S("register"), {{0, 1}}},
       {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {32, 0}}},
       {S("warp"), {{0, 0}, {16, 0}}},
       {S("block"), {{0, 0}, {64, 0}}}},
      {S("dim0"), S("dim1")});

  EXPECT_TRUE(isCvtDimSync(srcLayout, dstLayout, S("block"),
                           /*hasDistSharedMem=*/false));
  EXPECT_FALSE(isCvtDimSync(srcLayout, dstLayout, S("block"), true));
  EXPECT_TRUE(isCvtTrivialOverDims(srcLayout, dstLayout, S("block")));
}

} // namespace mlir

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
