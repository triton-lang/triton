#include "triton/Dialect/Triton/IR/Utility.h"

#include "mlir/Parser/Parser.h"
#include "triton/Analysis/Utility.h"
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

TEST(Analysis, ScanIgnoreBroadcastBasis) {
  // Exercise elem_count, num_blocks, and stride helpers when a broadcasted
  // register basis appears before a non-broadcast basis in the register order.
  mlir::DialectRegistry registry;
  registry
      .insert<mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect,
              mlir::arith::ArithDialect>();
  mlir::MLIRContext ctx(registry);
  ctx.getOrLoadDialect<mlir::triton::TritonDialect>();
  ctx.getOrLoadDialect<mlir::triton::gpu::TritonGPUDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();

  // sizePerThread[2] exceeds shape[2], so the register base for dim 2 is
  // broadcasted and should not affect contiguity or stride computation.
  const char *mlirText = R"mlir(
#layout_axis_bcast_3d = #ttg.blocked<{sizePerThread = [2, 2, 2], threadsPerWarp = [2, 4, 1], warpsPerCTA = [2, 1, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 8 : i32} {
  tt.func public @axis_scan(%arg0: tensor<8x8x1xf32, #layout_axis_bcast_3d>) -> tensor<8x8x1xf32, #layout_axis_bcast_3d> {
    %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.scan.return %1 : f32
    }) : (tensor<8x8x1xf32, #layout_axis_bcast_3d>) -> tensor<8x8x1xf32, #layout_axis_bcast_3d>
    tt.return %0 : tensor<8x8x1xf32, #layout_axis_bcast_3d>
  }
}
)mlir";

  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlirText, &ctx);
  ASSERT_TRUE(module);

  mlir::triton::ScanOp scanOp;
  module->walk([&](mlir::triton::ScanOp op) { scanOp = op; });
  ASSERT_TRUE(scanOp);

  mlir::ScanLoweringHelper helper(scanOp);
  EXPECT_EQ(helper.getAxisNumElementsPerThread(), 2u);
  EXPECT_EQ(helper.getNonAxisNumElementsPerThread(), 2u);
  EXPECT_EQ(helper.getAxisNumBlocks(), 1u);
  EXPECT_EQ(helper.getNonAxisNumBlocks(), 1u);
  EXPECT_EQ(helper.getAxisElementStride(), 2u);
  EXPECT_EQ(helper.getAxisBlockStride(), 1u);
}

} // namespace mlir

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
