#ifndef TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// TritonGPU depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonGPU/IR/Ops.h.inc"

#endif // TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_
