//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "xpu/lib/Conversion/TritonXPUToLLVM/Utility.h"

namespace mlir::LLVM::XPU {

Value llGetPid(Location loc, ConversionPatternRewriter &rewriter,
               ModuleOp moduleOp, int axis) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);
  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};

  // TODO[dyq]: add Dimension:y & Dimension:z mapping
  Value blockId;
  switch (axis) {
  case 0: {
    blockId = rewriter.create<::mlir::gpu::BlockIdOp>(loc, dims[axis]);
    break;
  }
  case 1:
  case 2: {
    blockId = i32_val(0);
    break;
  }
  default: {
    llvm_unreachable("ProgramIdOp Get Invalid Axis");
  }
  }

  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockId);
}

Type getFunctionType(mlir::OpBuilder &builder, ValueRange operands) {
  SmallVector<Type> operandTypes(operands.getTypes());
  mlir::MLIRContext *ctx = builder.getContext();
  auto voidTy = mlir::LLVM::LLVMVoidType::get(ctx);
  return LLVM::LLVMFunctionType::get(voidTy, operandTypes);
}

Value createDeviceCall(StringRef funcName, ConversionPatternRewriter &rewriter,
                       Operation *op, Type &elemTy, ValueRange &operands,
                       Location &loc) {
  Type funcType = mlir::triton::gpu::getFunctionType(elemTy, operands);
  LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
      rewriter, op, funcName, funcType, "", "");
  return rewriter.create<LLVM::CallOp>(loc, funcOp, operands).getResult();
}

void createDeviceCall(StringRef funcName, ConversionPatternRewriter &rewriter,
                      Operation *op, ValueRange &operands, Location &loc) {
  OpBuilder builder(op);
  Type funcType = getFunctionType(builder, operands);
  LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
      rewriter, op, funcName, funcType, "", "");
  rewriter.create<LLVM::CallOp>(loc, funcOp, operands);
  return;
}

} // namespace mlir::LLVM::XPU
