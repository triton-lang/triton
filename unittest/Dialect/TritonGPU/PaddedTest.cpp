#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

class PaddedTest : public ::testing::Test {
public:
  PaddedTest() {
    ctx.loadDialect<mlir::triton::TritonDialect,
                    mlir::triton::gpu::TritonGPUDialect>();
  }

  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(PaddedTest, TestMultiCTA) {
  std::pair<unsigned, unsigned> intervalPads(64, 8);
  unsigned order[2] = {1, 0};
  int64_t shape[2] = {16, 128};

  {
    auto cgaLL = LinearLayout({{S("block"), {{0, 1}}}}, {S("dim0"), S("dim1")});
    CGAEncodingAttr cgaLayout = CGAEncodingAttr::get(&ctx, cgaLL);

    auto attr = PaddedSharedEncodingAttr::get(&ctx, intervalPads, order, shape,
                                              cgaLayout);
    auto ll = attr.getLinearComponent();
    auto ofstLayout = ll.sublayout(S("offset"), to_vector(ll.getOutDimNames()));

    EXPECT_TRUE(ofstLayout.isInjective());
  }

  {
    auto cgaLL = LinearLayout({{S("block"), {{0, 0}}}}, {S("dim0"), S("dim1")});
    CGAEncodingAttr cgaLayout = CGAEncodingAttr::get(&ctx, cgaLL);

    auto attr = PaddedSharedEncodingAttr::get(&ctx, intervalPads, order, shape,
                                              cgaLayout);
    auto ll = attr.getLinearComponent();
    auto ofstLayout = ll.sublayout(S("offset"), to_vector(ll.getOutDimNames()));
    EXPECT_TRUE(ofstLayout.isInjective());
  }
}

TEST_F(PaddedTest, CopyVecIsCappedByThePaddingInterval) {
  unsigned order[2] = {1, 0};
  int64_t shape[2] = {128, 32};
  auto cga = CGAEncodingAttr::get1CTALayout(&ctx, /*rank=*/2);
  auto blocked =
      BlockedEncodingAttr::get(&ctx, {1, 4}, {4, 8}, {4, 1}, order, cga);
  auto regTy = RankedTensorType::get(shape, Float16Type::get(&ctx), blocked);
  auto vecBytes = [&](unsigned interval) {
    auto padded = PaddedSharedEncodingAttr::get(
        &ctx, std::pair<unsigned, unsigned>(interval, 8), order, shape, cga);
    return getCopyVecBytes(regTy, cast<SharedEncodingTrait>(Attribute(padded)));
  };
  // Four contiguous f16 elements fit inside an interval of 32, but a copy
  // cannot run past a shorter one.
  EXPECT_EQ(vecBytes(32), 8);
  EXPECT_EQ(vecBytes(2), 4);
  EXPECT_EQ(vecBytes(1), 2);
}
