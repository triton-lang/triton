#ifndef TRITON_DIALECT_TRITON_IR_DIALECT_H_
#define TRITON_DIALECT_TRITON_IR_DIALECT_H_

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "triton/Dialect/Triton/IR/Dialect.h.inc"
#include "triton/Dialect/Triton/IR/OpsEnums.h.inc"
#include "triton/Dialect/Triton/IR/Traits.h"
#include "triton/Dialect/Triton/IR/Types.h"

#define GET_OP_CLASSES
#include "triton/Dialect/Triton/IR/Ops.h.inc"

namespace mlir {
namespace triton {

class DialectInferLayoutInterface
    : public DialectInterface::Base<DialectInferLayoutInterface> {
public:
  DialectInferLayoutInterface(Dialect *dialect) : Base(dialect) {}

  virtual LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding) const = 0;

  virtual LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            Optional<Location> location) const = 0;

  // Note: this function only verify operand encoding but doesn't infer result
  // encoding
  virtual LogicalResult
  inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                     Attribute retEncoding,
                     Optional<Location> location) const = 0;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_IR_DIALECT_H_
