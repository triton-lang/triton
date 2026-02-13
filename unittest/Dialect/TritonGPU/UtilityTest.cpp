#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include <gtest/gtest.h>

namespace mlir::triton::gpu {
namespace {

class MMAInstrShapeTest : public ::testing::Test {
protected:
  MLIRContext ctx;
};

TEST_F(MMAInstrShapeTest, Version2Rank3Shape) {
  Type f16 = Float16Type::get(&ctx);
  auto instrShape = mmaVersionToInstrShape(/*version=*/2, {32, 64, 128}, f16,
                                           /*numWarps=*/4);
  ASSERT_TRUE(instrShape.has_value());
  ASSERT_EQ(instrShape->size(), 3u);
  EXPECT_EQ((*instrShape)[0], 1u);
  EXPECT_EQ((*instrShape)[1], 16u);
  EXPECT_EQ((*instrShape)[2], 8u);
}

TEST_F(MMAInstrShapeTest, Version3RejectsUnsupportedShape) {
  Type f16 = Float16Type::get(&ctx);
  auto instrShape = mmaVersionToInstrShape(/*version=*/3, {96, 64}, f16,
                                           /*numWarps=*/4);
  EXPECT_FALSE(instrShape.has_value());
}

TEST_F(MMAInstrShapeTest, Version3ReturnsExpectedShape) {
  Type f16 = Float16Type::get(&ctx);
  auto instrShape = mmaVersionToInstrShape(/*version=*/3, {128, 64}, f16,
                                           /*numWarps=*/8);
  ASSERT_TRUE(instrShape.has_value());
  ASSERT_EQ(instrShape->size(), 3u);
  EXPECT_EQ((*instrShape)[0], 16u);
  EXPECT_EQ((*instrShape)[1], 64u);
  EXPECT_EQ((*instrShape)[2], 16u);
}

TEST_F(MMAInstrShapeTest, RejectsInsufficientRank) {
  Type i8 = IntegerType::get(&ctx, 8);
  EXPECT_FALSE(mmaVersionToInstrShape(/*version=*/2, {128}, i8, /*numWarps=*/4)
                   .has_value());
  EXPECT_FALSE(mmaVersionToInstrShape(/*version=*/3, {128}, i8, /*numWarps=*/4)
                   .has_value());
  EXPECT_FALSE(mmaVersionToInstrShape(/*version=*/5, {128}, i8, /*numWarps=*/4)
                   .has_value());
}

TEST_F(MMAInstrShapeTest, RejectsUnsupportedVersion) {
  Type f16 = Float16Type::get(&ctx);
  EXPECT_FALSE(
      mmaVersionToInstrShape(/*version=*/4, {128, 64}, f16, /*numWarps=*/4)
          .has_value());
}

} // namespace
} // namespace mlir::triton::gpu
