#ifndef TRITON_THIRD_PARTY_METAL_INCLUDE_DIALECT_TRITONMETALGPU_IR_DIALECT_H_
#define TRITON_THIRD_PARTY_METAL_INCLUDE_DIALECT_TRITONMETALGPU_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// clang-format off
#include "metal/include/Dialect/TritonMetalGPU/IR/Dialect.h.inc"
// clang-format on

#define GET_OP_CLASSES
#include "metal/include/Dialect/TritonMetalGPU/IR/Ops.h.inc"

#endif // TRITON_THIRD_PARTY_METAL_INCLUDE_DIALECT_TRITONMETALGPU_IR_DIALECT_H_
