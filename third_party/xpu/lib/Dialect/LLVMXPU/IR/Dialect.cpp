//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/LLVMXPU/IR/Dialect.h" // before cpp.inc

#include "triton/Dialect/LLVMXPU/IR/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect Initialization
//===----------------------------------------------------------------------===//

void ::mlir::LLVM::XPU::LLVMXPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST // declare
#include "triton/Dialect/LLVMXPU/IR/Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES // define
#include "triton/Dialect/LLVMXPU/IR/Ops.cpp.inc"

mlir::LogicalResult
mlir::LLVM::XPU::LLVMXPUDialect::verifyOperationAttribute(Operation *op,
                                                          NamedAttribute attr) {
  // Kernel function attribute should be attached to functions.
  if (attr.getName() == LLVMXPUDialect::getKernelFuncAttrName()) {
    if (!isa<LLVM::LLVMFuncOp>(op)) {
      return op->emitError() << "'" << LLVMXPUDialect::getKernelFuncAttrName()
                             << "' attribute attached to unexpected op";
    }
  }
  return success();
}
