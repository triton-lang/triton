#include "mlir/IR/Verifier.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace mlir;

static std::string getIRParseError(std::string ir) {
  DialectRegistry registry;
  registry.insert<triton::TritonDialect, triton::gpu::TritonGPUDialect>();
  MLIRContext context(registry);

  std::optional<std::string> error;
  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_TRUE(!error.has_value());
    error = diag.str();
  });

  Block block;
  ParserConfig config(&context);
  LogicalResult result = parseSourceString(ir, &block, config);

  if (succeeded(result)) {
    assert(!error.has_value());
  }
  return error.value_or("");
}

TEST(VerifierTest, InvalidBlockedTensor_ThreadsPerWarp) {
  std::string ir = R"(
    #blocked = #triton_gpu.blocked<{
      sizePerThread=[1, 1],
      threadsPerWarp=[16, 1],
      warpsPerCTA=[4, 1],
      order=[0, 1],
      CTAsPerCGA=[2, 1],
      CTASplitNum=[1, 1],
      CTAOrder=[0, 1]
    }>
    module attributes {
      "triton_gpu.num-warps" = 4 : i32,
      "triton_gpu.num-ctas" = 2 : i32,
      "triton_gpu.threads-per-warp" = 32 : i32
    } {
      tt.func public @fn(%arg0: !tt.ptr<i32, 1>) {
          %t = tt.splat %arg0 : (!tt.ptr<i32,1>) -> tensor<8x1x!tt.ptr<i32,1>, #blocked>
          tt.return
      }
   })";
  EXPECT_THAT(getIRParseError(ir), ::testing::HasSubstr("threads per warp"));
}

TEST(VerifierTest, InvalidBlockedTensor_WarpsPerCTA) {
  std::string ir = R"(
    #blocked = #triton_gpu.blocked<{
      sizePerThread=[1, 1],
      threadsPerWarp=[32, 1],
      warpsPerCTA=[4, 2],
      order=[0, 1],
      CTAsPerCGA=[2, 1],
      CTASplitNum=[1, 1],
      CTAOrder=[0, 1]
    }>
    module attributes {
      "triton_gpu.num-warps" = 4 : i32,
      "triton_gpu.num-ctas" = 2 : i32,
      "triton_gpu.threads-per-warp" = 32 : i32
    } {
      tt.func public @fn(%arg0: !tt.ptr<i32, 1>) {
          %t = tt.splat %arg0 : (!tt.ptr<i32,1>) -> tensor<8x1x!tt.ptr<i32,1>, #blocked>
          tt.return
      }
   })";
  EXPECT_THAT(getIRParseError(ir), ::testing::HasSubstr("warps per CTA"));
}

TEST(VerifierTest, InvalidBlockedTensor_CTAsPerCGA) {
  std::string ir = R"(
    #blocked = #triton_gpu.blocked<{
      sizePerThread=[1, 1],
      threadsPerWarp=[32, 1],
      warpsPerCTA=[4, 1],
      order=[0, 1],
      CTAsPerCGA=[1, 1],
      CTASplitNum=[1, 1],
      CTAOrder=[0, 1]
    }>
    module attributes {
      "triton_gpu.num-warps" = 4 : i32,
      "triton_gpu.num-ctas" = 2 : i32,
      "triton_gpu.threads-per-warp" = 32 : i32
    } {
      tt.func public @fn(%arg0: !tt.ptr<i32, 1>) {
          %t = tt.splat %arg0 : (!tt.ptr<i32,1>) -> tensor<8x1x!tt.ptr<i32,1>, #blocked>
          tt.return
      }
   })";
  EXPECT_THAT(getIRParseError(ir), ::testing::HasSubstr("CTAs per CGA"));
}

TEST(VerifierTest, InvalidBlockedTensor_Rank) {
  std::string ir = R"(
    #blocked = #triton_gpu.blocked<{
      sizePerThread=[1, 1],
      threadsPerWarp=[32, 1],
      warpsPerCTA=[4, 1],
      order=[0, 1],
      CTAsPerCGA=[1, 2],
      CTASplitNum=[1, 1],
      CTAOrder=[0, 1]
    }>
    module attributes {
      "triton_gpu.num-warps" = 4 : i32,
      "triton_gpu.num-ctas" = 2 : i32,
      "triton_gpu.threads-per-warp" = 32 : i32
    } {
      tt.func public @fn(%arg0: !tt.ptr<i32, 1>) {
          // Note it's a 3d tensor here, but #blocked is 2D.
          %t = tt.splat %arg0 : (!tt.ptr<i32,1>) -> tensor<8x1x1x!tt.ptr<i32,1>, #blocked>
          tt.return
      }
   })";
  EXPECT_THAT(getIRParseError(ir), ::testing::HasSubstr("rank"));
}
