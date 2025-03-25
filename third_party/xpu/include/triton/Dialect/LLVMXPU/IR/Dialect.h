//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_LLVMXPU_IR_DIALECT_H_
#define MLIR_DIALECT_LLVMXPU_IR_DIALECT_H_

// LLVMXPUDialect
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/LLVMXPU/IR/Dialect.h.inc"

// LLVMXPUOps
#define GET_OP_CLASSES
#include "triton/Dialect/LLVMXPU/IR/Ops.h.inc"

namespace mlir {
namespace LLVM {
namespace XPU {} // namespace XPU
} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMXPU_IR_DIALECT_H_
