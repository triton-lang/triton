#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Signals.h"
#include <gtest/gtest.h>

using namespace mlir;
using mlir::triton::gpu::SwizzledSharedEncodingAttr;

struct swizzleParams {
  int vec;
  int perPhase;
  int maxPhase;
};

struct ParamT {
  std::array<int64_t, 2> shape;
  int opIdx;
  int typeWidth;
  swizzleParams refSwizzle;
};

class SwizzleDotOperandTestFixture : public ::testing::TestWithParam<ParamT> {
protected:
  ParamType param;
};

TEST_P(SwizzleDotOperandTestFixture, DotOperands) {
  auto params = GetParam();
  // init context
  MLIRContext ctx;
  ctx.loadDialect<triton::gpu::TritonGPUDialect>();

  auto CTALayout =
      triton::gpu::CTALayoutAttr::get(&ctx, {1, 1}, {1, 1}, {0, 1});

  // create encoding
  auto parent = triton::gpu::NvidiaMmaEncodingAttr::get(
      &ctx, 2, 0, {1, 1}, CTALayout, {16, 64, 16});
  auto encoding = triton::gpu::DotOperandEncodingAttr::get(
      &ctx, params.opIdx, parent, 32 / params.typeWidth);

  // create element type
  Type eltType = IntegerType::get(&ctx, params.typeWidth);
  auto layout = SwizzledSharedEncodingAttr::get(&ctx, encoding, params.shape,
                                                {1, 0}, CTALayout, eltType);

  ASSERT_EQ(layout.getVec(), params.refSwizzle.vec);
  ASSERT_EQ(layout.getPerPhase(), params.refSwizzle.perPhase);
  ASSERT_EQ(layout.getMaxPhase(), params.refSwizzle.maxPhase);
}

INSTANTIATE_TEST_SUITE_P(TestDotOperands, SwizzleDotOperandTestFixture,
                         ::testing::Values(ParamT{{128, 64}, 0, 16, {8, 1, 8}},
                                           ParamT{{64, 256}, 1, 16, {8, 1, 8}},
                                           ParamT{{128, 32}, 0, 16, {8, 2, 4}},
                                           ParamT{{32, 128}, 1, 16, {8, 1, 8}},
                                           ParamT{{32, 32}, 0, 16, {8, 2, 4}},
                                           ParamT{{32, 32}, 1, 16, {8, 2, 4}},
                                           ParamT{{16, 16}, 0, 16, {8, 4, 2}},
                                           ParamT{{16, 16}, 1, 16, {8, 4, 2}}));

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
