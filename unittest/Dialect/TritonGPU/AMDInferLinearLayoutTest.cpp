#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/lib/TritonAMDGPUDialectToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Signals.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
std::ostream &operator<<(std::ostream &os, StringAttr str) {
  os << str.str();
  return os;
}
} // namespace mlir

namespace mlir {
namespace {

class AMDInferLinearLayoutTest : public ::testing::Test {
public:
  void SetUp() {
    ctx.getOrLoadDialect<mlir::triton::amdgpu::TritonAMDGPUDialect>();
  }

  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(AMDInferLinearLayoutTest, ExtractSlice) {
  auto ll1D = triton::LinearLayout(
      {
          {S("register"), {{1}, {256}, {512}}},
          {S("lane"), {{2}, {4}, {8}, {16}, {32}, {64}}},
          {S("warp"), {{128}}},
          {S("block"), {}},
      },
      {S("dim0")});

  auto elemTy = Float32Type::get(&ctx);
  auto tensorType = RankedTensorType::get(
      {1024}, elemTy, triton::gpu::LinearEncodingAttr::get(&ctx, ll1D));
  auto sliced = LLVM::AMD::inferLinearLayoutFromExtractSlice(&ctx, tensorType);
  auto expected = triton::LinearLayout(
      {
          {S("register"), {{1}}},
          {S("lane"), {{2}, {4}, {8}, {16}, {32}, {64}}},
          {S("warp"), {{128}}},
          {S("block"), {}},
      },
      {S("dim0")});

  auto ll2D = triton::LinearLayout(
      {
          {S("register"), {{1, 0}, {2, 0}, {4, 0}, {0, 16}, {0, 32}, {0, 64}}},
          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}, {16, 0}}},
          {S("warp"), {{32, 0}, {64, 0}, {128, 0}}},
          {S("block"), {}},
      },
      {S("dim0")});
  tensorType = RankedTensorType::get(
      {256, 128}, elemTy, triton::gpu::LinearEncodingAttr::get(&ctx, ll2D));
  sliced = LLVM::AMD::inferLinearLayoutFromExtractSlice(&ctx, tensorType);
  expected = triton::LinearLayout(
      {
          {S("register"), {{1, 0}, {2, 0}, {4, 0}}},
          {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}, {16, 0}}},
          {S("warp"), {{32, 0}, {64, 0}, {128, 0}}},
          {S("block"), {}},
      },
      {S("dim0"), S("dim1")});
  EXPECT_EQ(expected, sliced);
}

} // anonymous namespace
} // namespace mlir

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
