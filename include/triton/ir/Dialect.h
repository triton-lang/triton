#ifndef TRITON_IR_DIALECT_H_
#define TRITON_IR_DIALECT_H_

#include "triton/ir/Traits.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"

#include "triton/ir/Dialect.h.inc"

#include "triton/ir/OpsEnums.h.inc"

#define GET_OP_CLASSES
#include "triton/ir/Ops.h.inc"

#endif // TRITON_IR_DIALECT_H_
