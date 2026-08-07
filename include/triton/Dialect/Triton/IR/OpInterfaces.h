#ifndef TRITON_IR_OP_INTERFACES_H_
#define TRITON_IR_OP_INTERFACES_H_

#include <cstdint>

#include "mlir/IR/OpDefinition.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir {

namespace triton {

enum class MemSemantic : uint32_t;

namespace impl {

LogicalResult verifyTransposeOpInterface(Operation *op);

LogicalResult verifyDotOpInterface(Operation *op);

} // namespace impl

} // namespace triton
} // namespace mlir

#include "triton/Dialect/Triton/IR/OpInterfaces.h.inc"

#endif // TRITON_IR_OP_INTERFACES_H_
