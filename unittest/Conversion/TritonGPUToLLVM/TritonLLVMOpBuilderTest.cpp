#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>

using namespace mlir;
using namespace mlir::triton;

namespace {

class TritonLLVMOpBuilderTest : public ::testing::Test {
protected:
  MLIRContext ctx;
  void SetUp() override {
    ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    ctx.getOrLoadDialect<mlir::triton::TritonDialect>();
  }
};

TEST_F(TritonLLVMOpBuilderTest, testCache) {
  OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  auto func = builder.create<mlir::triton::FuncOp>(
      loc, "test", builder.getFunctionType({}, {}));
  auto block = func.addEntryBlock();
  auto build = [&builder]() {
    TritonLLVMOpBuilder ttb(builder.getUnknownLoc(), builder);
    ttb.i32_val(0);
    ttb.i32_val(0);
    ttb.i32_val(1);
    ttb.i32_val(2);
    SmallVector<Attribute, 16> values;
    for (double v : {0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5,
                     -2., -3., -4., -6.})
      values.push_back(builder.getFloatAttr(builder.getBF16Type(), v));
    ttb.dense_val(VectorType::get({16}, builder.getBF16Type()), values);
    ttb.dense_val(VectorType::get({16}, builder.getBF16Type()), values);
  };
  auto check = [&](bool empty = false) {
    ASSERT_TRUE(mlir::verify(module).succeeded());

    if (empty) {
      ASSERT_TRUE(
          isa<mlir::triton::ReturnOp>(*func.getBody().getOps().begin()));
      return;
    }

    unsigned counter = 0;
    for (auto &op : func.getBody().getOps()) {
      if (auto c = dyn_cast<LLVM::ConstantOp>(op)) {
        if (c.getType() == builder.getIntegerType(32)) {
          auto v = dyn_cast<IntegerAttr>(c.getValueAttr()).getInt();
          if (v >= 0 && v <= 2) {
            ++counter;
            continue;
          }
        } else if (auto vecType = dyn_cast<VectorType>(c.getType())) {
          if (vecType.getElementType() == builder.getBF16Type() &&
              vecType.getNumElements() == 16) {
            ++counter;
            continue;
          }
        }
      } else if (isa<mlir::triton::ReturnOp>(op)) {
        ++counter;
        continue;
      }

      op.dump();
      FAIL() << "Unexpected operation!";
    }
    ASSERT_EQ(counter, 5) << "Unexpected number of operations!"
                          << " Expected 5, actual " << counter;
  };

  module.push_back(func);
  builder.setInsertionPointToEnd(block);
  builder.create<mlir::triton::ReturnOp>(loc);
  builder.setInsertionPointToStart(block);

  build();
  check();

  mlir::PassManager pm(&ctx);
  pm.addPass(mlir::createCanonicalizerPass());
  ASSERT_TRUE(pm.run(module).succeeded());
  // Check if the constants has been removed by the canonicalizer
  check(true);

  build();
  check();
}

// Fill vectors with constants and print timings
TEST_F(TritonLLVMOpBuilderTest, testFillPerf) {
  GTEST_SKIP();
  OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();
  mlir::PassManager pm(&ctx);
  pm.addPass(mlir::createCanonicalizerPass());
  auto module = ModuleOp::create(loc);

  unsigned size = 1024;
  auto i32Type = builder.getIntegerType(32);
  auto ptrType = LLVM::LLVMPointerType::get(&ctx);
  auto intVecType = VectorType::get(size, i32Type);
  auto ptrVecType = VectorType::get(size, ptrType);
  auto func = builder.create<mlir::triton::FuncOp>(
      loc, "test", builder.getFunctionType({ptrVecType}, {}));
  module.push_back(func);
  builder.setInsertionPointToStart(func.addEntryBlock());
  auto ptrVec = func.getArgument(0);

  TritonLLVMOpBuilder ttb(loc, builder);
  auto start = std::chrono::high_resolution_clock::now();
  for (int row = 0; row < 1024; ++row) {
    Value vec = ttb.undef(intVecType);
    for (int col = 0; col < 1024; ++col) {
      vec = ttb.insert_element(vec, ttb.i32_val(col), ttb.i32_val(col));
    }
    Value ptr = ttb.extract_element(ptrVec, ttb.i32_val(row));
    ttb.store(vec, ptr);
  }
  std::chrono::duration<double> time =
      std::chrono::high_resolution_clock::now() - start;
  std::cout << "Fill| Insertion time: " << time.count() << std::endl;

  builder.create<mlir::triton::ReturnOp>(loc);
  ASSERT_TRUE(mlir::verify(module).succeeded());
  unsigned counter = 0;
  for (auto &op : func.getBody().getOps()) {
    if (auto c = dyn_cast<LLVM::ConstantOp>(op)) {
      ASSERT_EQ(c.getType(), i32Type) << "Unexpected constant type";
      ASSERT_GE(dyn_cast<IntegerAttr>(c.getValueAttr()).getInt(), 0);
      ASSERT_LT(dyn_cast<IntegerAttr>(c.getValueAttr()).getInt(), size);
      counter++;
    }
  }
  ASSERT_EQ(counter, size);

  start = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(pm.run(module).succeeded());
  time = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Fill| Canonicalizer time: " << time.count() << std::endl;
  ASSERT_TRUE(mlir::verify(module).succeeded());
}

// Simillar to the above, but only creates the constants
TEST_F(TritonLLVMOpBuilderTest, testCreatePerf) {
  GTEST_SKIP();
  OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();
  mlir::PassManager pm(&ctx);
  pm.addPass(mlir::createCanonicalizerPass());
  auto module = ModuleOp::create(loc);

  unsigned size = 1024;
  auto i32Type = builder.getIntegerType(32);
  auto func = builder.create<mlir::triton::FuncOp>(
      loc, "test", builder.getFunctionType({}, {}));
  module.push_back(func);
  builder.setInsertionPointToStart(func.addEntryBlock());

  TritonLLVMOpBuilder ttb(loc, builder);
  auto start = std::chrono::high_resolution_clock::now();
  for (int row = 0; row < 1024; ++row) {
    for (int col = 0; col < 1024; ++col) {
      ttb.i32_val(col);
      ttb.i32_val(col);
    }
    ttb.i32_val(row);
  }
  std::chrono::duration<double> time =
      std::chrono::high_resolution_clock::now() - start;
  std::cout << "Create| Insertion time: " << time.count() << std::endl;

  builder.create<mlir::triton::ReturnOp>(loc);
  ASSERT_TRUE(mlir::verify(module).succeeded());
  unsigned counter = 0;
  for (auto &op : func.getBody().getOps()) {
    if (auto c = dyn_cast<LLVM::ConstantOp>(op)) {
      ASSERT_EQ(c.getType(), i32Type) << "Unexpected constant type";
      ASSERT_GE(dyn_cast<IntegerAttr>(c.getValueAttr()).getInt(), 0);
      ASSERT_LT(dyn_cast<IntegerAttr>(c.getValueAttr()).getInt(), size);
      counter++;
    }
  }
  ASSERT_EQ(counter, size);

  start = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(pm.run(module).succeeded());
  time = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Create| Canonicalizer time: " << time.count() << std::endl;
  ASSERT_TRUE(isa<mlir::triton::ReturnOp>(*func.getBody().getOps().begin()));
}
} // namespace
