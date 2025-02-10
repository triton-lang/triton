#ifndef DIALECT_PROTON_GPU_IR_DIALECT_H_
#define DIALECT_PROTON_GPU_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/dialect/include/Dialect/ProtonGPU/IR/Dialect.h.inc"
#include "proton/dialect/include/Dialect/ProtonGPU/IR/OpsEnums.h.inc"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "proton/dialect/include/Dialect/ProtonGPU/IR/Ops.h.inc"

#endif // DIALECT_PROTON_GPU_IR_DIALECT_H_
