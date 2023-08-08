#include "triton/Dialect/Triton/IR/Traits.h"

#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;

static LogicalResult verifySameEncoding(Type typeA, Type typeB,
                                        bool allowTensorPointerType) {
  // TODO(Keren): the allowTensorPointerType argument is a hack to allow.
  // The type checking code is kind of a mess with the current design.
  auto getEncoding = [=](Type type) -> Attribute {
    Attribute ret;
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      ret = tensorType.getEncoding();
    }
    if (!allowTensorPointerType) {
      assert(!triton::isTensorPointerType(type));
    }
    return ret;
  };
  auto encodingA = getEncoding(typeA);
  auto encodingB = getEncoding(typeB);
  if (!encodingA || !encodingB)
    return success();
  return encodingA == encodingB ? success() : failure();
}

LogicalResult
OpTrait::impl::verifySameOperandsEncoding(Operation *op,
                                          bool allowTensorPointerType) {
  if (failed(verifyAtLeastNOperands(op, 1)))
    return failure();

  auto type = op->getOperand(0).getType();
  for (auto opType : llvm::drop_begin(op->getOperandTypes(), 1))
    if (failed(verifySameEncoding(opType, type, allowTensorPointerType)))
      return op->emitOpError() << "requires the same encoding for all operands";

  return success();
}

LogicalResult OpTrait::impl::verifySameOperandsAndResultEncoding(
    Operation *op, bool allowTensorPointerType) {
  if (op->getNumOperands() == 0)
    return success();

  if (failed(verifyAtLeastNOperands(op, 1)) ||
      failed(verifyAtLeastNResults(op, 1)))
    return failure();

  auto type = op->getOperand(0).getType();
  for (auto resultType : op->getResultTypes())
    if (failed(verifySameEncoding(resultType, type, allowTensorPointerType)))
      return op->emitOpError()
             << "requires the same encoding for all operands and results";

  return verifySameOperandsEncoding(op, allowTensorPointerType);
}

LogicalResult OpTrait::impl::verifyTensorSize(Operation *op) {
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

static ArrayRef<int64_t> getTypeShape(Type type) {
  auto rankedType = type.dyn_cast<RankedTensorType>();
  if (auto ptrType = type.dyn_cast<triton::PointerType>())
    rankedType = ptrType.getPointeeType().dyn_cast<RankedTensorType>();
  return rankedType ? rankedType.getShape() : ArrayRef<int64_t>();
}

LogicalResult OpTrait::impl::verifySameLoadStoreOperandsShape(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)))
    return failure();

  auto firstOperandShape = getTypeShape(op->getOperand(0).getType());
  for (auto type : llvm::drop_begin(op->getOperandTypes(), 1))
    if (failed(verifyCompatibleShape(getTypeShape(type), firstOperandShape)))
      return op->emitOpError() << "requires the same shape for all operands";

  return success();
}

LogicalResult
OpTrait::impl::verifySameLoadStoreOperandsAndResultShape(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)) ||
      failed(verifyAtLeastNResults(op, 1)))
    return failure();

  auto firstOperandShape = getTypeShape(op->getOperand(0).getType());
  for (auto type : op->getResultTypes())
    if (failed(verifyCompatibleShape(getTypeShape(type), firstOperandShape)))
      return op->emitOpError()
             << "requires the same shape for all operands and results";

  return verifySameLoadStoreOperandsShape(op);
}

bool OpTrait::impl::verifyLoadStorePointerAndValueType(Type valueType,
                                                       Type ptrType) {
  if (triton::isTensorPointerType(ptrType)) {
    // The encoding of tensor pointers is meaningless, we only check shapes and
    // the type of elements
    auto tensorAType = ptrType.cast<triton::PointerType>()
                           .getPointeeType()
                           .cast<RankedTensorType>();
    if (!isa<RankedTensorType>(valueType))
      return false;
    auto tensorBType = valueType.cast<RankedTensorType>();
    return tensorAType.getShape() == tensorBType.getShape() &&
           tensorAType.getElementType() == tensorBType.getElementType();
  } else if (auto rankedType = ptrType.dyn_cast<RankedTensorType>()) {
    if (auto elementPtrType =
            dyn_cast<triton::PointerType>(rankedType.getElementType())) {
      auto inferValueType = RankedTensorType::get(
          rankedType.getShape(), elementPtrType.getPointeeType(),
          rankedType.getEncoding());
      return inferValueType == valueType;
    }
  } else if (auto scalarPtrType = ptrType.dyn_cast<triton::PointerType>()) {
    return scalarPtrType.getPointeeType() == valueType;
  }
  return false;
}
