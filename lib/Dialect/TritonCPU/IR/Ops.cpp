#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonCPU/IR/Ops.cpp.inc"

// enum attribute definitions
#include "triton/Dialect/TritonCPU/IR/OpsEnums.cpp.inc"

namespace mlir::triton::cpu {

LogicalResult PrintOp::verify() {
  if (getOperands().size() > 1)
    return emitOpError("expects at most one operand");
  return success();
}

} // namespace mlir::triton::cpu
