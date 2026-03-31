#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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
