#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
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

TEST_F(PaddedTest, TestPartitionedMultiCTAIsCTALocal) {
  std::pair<unsigned, unsigned> intervalPads(128, 8);
  unsigned order[2] = {1, 0};
  int64_t pieceShape[2] = {128, 128};
  int64_t fullShape[2] = {512, 128};

  // Split dim0 over two CTAs and multicast along the second block bit.
  auto cgaLL =
      LinearLayout({{S("block"), {{1, 0}, {0, 0}}}}, {S("dim0"), S("dim1")});
  auto cgaLayout = CGAEncodingAttr::get(&ctx, cgaLL);
  auto inner = PaddedSharedEncodingAttr::get(&ctx, intervalPads, order,
                                             pieceShape, cgaLayout);
  auto partitioned = PartitionedSharedEncodingAttr::get(
      &ctx, /*numPartitions=*/2, /*numGroups=*/2, /*partitionDim=*/0, inner);

  auto ll = paddedLinearLayout(fullShape, partitioned);

  // The four partition/group pieces span one 256-row CTA tile. The CGA block
  // bit selects the outer 256-row tile rather than interleaving within pieces.
  EXPECT_EQ(ll.getBasis(S("partition"), 0, S("dim0")), 64);
  EXPECT_EQ(ll.getBasis(S("offset"), 13, S("dim0")), 128);
  EXPECT_EQ(ll.getBasis(S("block"), 0, S("dim0")), 256);
  EXPECT_EQ(ll.getBasis(S("block"), 1, S("dim0")), 0);
  EXPECT_TRUE(ll.isSurjective());
}

TEST_F(PaddedTest, TestPartitionedSwizzledMultiCTAIsCTALocalInDim1) {
  unsigned order[2] = {0, 1};
  int64_t fullShape[2] = {128, 512};

  // Split dim1 over two CTAs and multicast along the second block bit.
  auto cgaLL =
      LinearLayout({{S("block"), {{0, 1}, {0, 0}}}}, {S("dim0"), S("dim1")});
  auto cgaLayout = CGAEncodingAttr::get(&ctx, cgaLL);
  auto inner = SwizzledSharedEncodingAttr::get(
      &ctx, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1, order, cgaLayout);
  auto partitioned = PartitionedSharedEncodingAttr::get(
      &ctx, /*numPartitions=*/2, /*numGroups=*/2, /*partitionDim=*/1, inner);

  auto ll = toLinearLayout(fullShape, partitioned);

  EXPECT_EQ(ll.getBasis(S("partition"), 0, S("dim1")), 64);
  EXPECT_EQ(ll.getBasis(S("offset"), 13, S("dim1")), 128);
  EXPECT_EQ(ll.getBasis(S("block"), 0, S("dim1")), 256);
  EXPECT_EQ(ll.getBasis(S("block"), 1, S("dim1")), 0);
  EXPECT_TRUE(ll.isSurjective());
}
