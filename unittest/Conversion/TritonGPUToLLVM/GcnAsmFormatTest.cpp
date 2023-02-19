#include "triton/Conversion/TritonGPUToLLVM/GCNAsmFormat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <gtest/gtest.h>

namespace mlir {
namespace triton {
class GcnAsmFormatTest : public ::testing::Test {
protected:
  static constexpr int numValues = 4;

  GcnAsmFormatTest() {
    ctx.loadDialect<arith::ArithDialect>();

    createValues();
  }

  // Creates the test values.
  void createValues() {
    OpBuilder builder(&ctx);
    builder.setInsertionPointToStart(&block);

    // a b1 value for predicate.
    v[0] = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 1, 1);
    for (int i = 0; i < numValues; i++) {
      v[i + 1] =
          builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), i, 32);
    }
  }

  MLIRContext ctx;
  Block block;
  Value v[numValues + 1];
};

TEST_F(GcnAsmFormatTest, basic) {
  GCNBuilder builder;

  // Create the operands needed by the instructions in the GCNs code.
  auto *cst = builder.newConstantOperand(1);
  auto *val = builder.newOperand(v[1], "=v");

  // create an instruction
  auto &mov = *builder.create("v_mov_b32");

  mov(val, cst);
  ASSERT_EQ(builder.dump(), "v_mov_b32 $0, 0x1");

  auto constraints = builder.getConstraints();
  ASSERT_EQ(constraints, "=v");
}

TEST_F(GcnAsmFormatTest, complexInstruction) {
  GCNBuilder builder;
  int offset = 128;
  int width = 16;

  Value addrVal = v[0];
  auto dst = builder.newOperand("=v");
  auto addr = builder.newAddrOperand(addrVal, "v");
  auto offOpr = builder.newConstantOperand("off");

  auto offsetMod = builder.newModifier("offset", std::to_string(offset));

  auto &ld = builder.create<GCNMemInstr>("global_load")->load_type(width);

  // Link the instruction to operands
  ld({addr, dst, offOpr}, {offsetMod});

  EXPECT_EQ(builder.dump(), "global_load_ushort $1, $0, off offset:128");

  auto constraints = builder.getConstraints();
  ASSERT_EQ(constraints, "=v,v");

  auto values = builder.getAllMLIRArgs();
  EXPECT_EQ(values[0], addrVal);
  EXPECT_EQ(builder.getConstraints(), "=v,v");
}

} // namespace triton
} // namespace mlir
