#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"

#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir {
namespace triton {
namespace impl {

LogicalResult verifyTransposeOpInterface(Operation *op) {
  TransposeOpInterface transposeOp = cast<TransposeOpInterface>(op);
  auto rank = cast<ShapedType>(transposeOp.getSrc().getType()).getRank();
  auto order = transposeOp.getOrder();
  if (rank != order.size()) {
    return op->emitError(
        "order must have the same size as the rank of the operand and result");
  }

  SmallVector<int32_t, 8> sortedOrder(order);
  llvm::sort(sortedOrder);
  for (int32_t i = 0; i < sortedOrder.size(); i++) {
    if (sortedOrder[i] != i) {
      return op->emitError("order must be a permutation of [0, ..., rank - 1]");
    }
  }

  return success();
}

} // namespace impl
} // namespace triton
} // namespace mlir
