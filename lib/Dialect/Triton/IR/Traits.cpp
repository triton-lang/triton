#include "triton/Dialect/Triton/IR/Traits.h"

static mlir::LogicalResult verifySameEncoding(mlir::Type tyA, mlir::Type tyB) {
  using namespace mlir;
  auto encA = tyA.dyn_cast<RankedTensorType>();
  auto encB = tyA.dyn_cast<RankedTensorType>();
  if (!encA || !encB)
    return success();
  return encA.getEncoding() == encB.getEncoding() ? success() : failure();
}

mlir::LogicalResult
mlir::OpTrait::impl::verifySameOperandsAndResultEncoding(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)) ||
      failed(verifyAtLeastNResults(op, 1)))
    return failure();

  auto type = op->getOperand(0).getType();
  for (auto resultType : op->getResultTypes())
    if (failed(verifySameEncoding(resultType, type)))
      return op->emitOpError()
             << "requires the same shape for all operands and results";
  return verifySameOperandsEncoding(op);
}

mlir::LogicalResult
mlir::OpTrait::impl::verifySameOperandsEncoding(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)))
    return failure();

  auto type = op->getOperand(0).getType();
  for (auto opType : llvm::drop_begin(op->getOperandTypes(), 1))
    if (failed(verifySameEncoding(opType, type)))
      return op->emitOpError() << "requires the same encoding for all operands";

  return success();
}

mlir::LogicalResult mlir::OpTrait::impl::verifyTensorSize(Operation *op) {
  for (auto opType : op->getOperandTypes()) {
    if (auto tensorType = opType.dyn_cast<RankedTensorType>()) {
      int64_t numElements = 1;
      for (int64_t s : tensorType.getShape())
        numElements *= s;
      if (numElements > maxTensorNumElements)
        return op->emitError("Maximum allowed number of elements is ")
               << maxTensorNumElements << ", but " << *op
               << " has more than that";
      if ((numElements & (numElements - 1)) != 0)
        return op->emitError("Number of elements must be power-of-two, but ")
               << *op << " doesn't follow the rule (" << numElements << ")"
               << " elements";
    }
  }
  for (auto opType : op->getResultTypes()) {
    if (auto tensorType = opType.dyn_cast<RankedTensorType>()) {
      int64_t numElements = 1;
      for (int64_t s : tensorType.getShape())
        numElements *= s;
      if (numElements > maxTensorNumElements)
        return op->emitError("Maximum allowed number of elements is ")
               << maxTensorNumElements << ", but " << *op
               << " has more than that";
      if ((numElements & (numElements - 1)) != 0)
        return op->emitError("Number of elements must be power-of-two, but ")
               << *op << " doesn't follow the rule (" << numElements << ")"
               << " elements";
    }
  }
  return success();
}
