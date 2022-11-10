#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <gtest/gtest.h>

using namespace mlir;
using mlir::triton::gpu::SharedEncodingAttr;

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
  // create encoding
  auto parent = triton::gpu::MmaEncodingAttr::get(&ctx, 2, {1, 1});
  auto encoding =
      triton::gpu::DotOperandEncodingAttr::get(&ctx, params.opIdx, parent);

  // create element type
  Type eltType = IntegerType::get(&ctx, params.typeWidth);
  auto layout = SharedEncodingAttr::get(&ctx, encoding, params.shape, eltType);

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