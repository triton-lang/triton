#ifndef TRITON_IR_DIALECT_H_
#define TRITON_IR_DIALECT_H_


#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"

#include "triton/Dialect/Triton/IR/Traits.h"
#include "triton/Dialect/Triton/IR/Dialect.h.inc"
#include "triton/Dialect/Triton/IR/OpsEnums.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/Triton/IR/Ops.h.inc"

#endif // TRITON_IR_DIALECT_H_
