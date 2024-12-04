#ifndef TRITON_IR_TRAITS_H_
#define TRITON_IR_TRAITS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir {
namespace OpTrait {

// These functions are out-of-line implementations of the methods in the
// corresponding trait classes. This avoids them being template
// instantiated/duplicated.
namespace impl {
// The rationale for this trait is to prevent users from creating programs
// that would have catastrophic register pressure and cause the compiler to
// hang.
// Since H100 has 256KB registers, we should allow users to create tensors
// of size up to 256K elements. It will spill for datatypes wider than 1B,
// but we probably should limit number of elements (rather than bytes) to
// keep specs simple
int constexpr maxTensorNumElements = 1048576;

LogicalResult verifyTensorSize(Operation *op);
LogicalResult verifyTensorLayouts(Operation *op);

LogicalResult verifySameOperandsEncoding(Operation *op,
                                         bool allowTensorPointerType = false);

LogicalResult
verifySameOperandsAndResultEncoding(Operation *op,
                                    bool allowTensorPointerType = false);

LogicalResult verifySameLoadStoreOperandsShape(Operation *op);

LogicalResult verifySameLoadStoreOperandsAndResultShape(Operation *op);

} // namespace impl

template <class ConcreteType>
class TensorSizeTrait : public TraitBase<ConcreteType, TensorSizeTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyTensorSize(op);
  }
};

// Trait applied to all Triton MLIR ops.  Checks that the layouts of tensors are
// valid.
template <class ConcreteType>
class VerifyTensorLayoutsTrait
    : public TraitBase<ConcreteType, VerifyTensorLayoutsTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyTensorLayouts(op);
  }
};

// Verify if the op is a dot-like operation.
// A dot-like operation should have three operands.
// The first two operands should share a common dimension, and the result
// should have the dimensions of the two operands that are not shared.
// A dot-like operation can be either 2d or 3d.
// In the 3d case, the first dimension of operands is the batch dimension.
template <class ConcreteType>
class DotLike : public TraitBase<ConcreteType, DotLike> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    if (op->getNumOperands() < 3)
      return op->emitOpError("expected at least 3 operands");
    auto aTy = cast<ShapedType>(op->getOperand(0).getType());
    auto bTy = cast<ShapedType>(op->getOperand(1).getType());
    auto cTy = cast<ShapedType>(op->getOperand(2).getType());
    auto aShape = aTy.getShape();
    auto bShape = bTy.getShape();
    auto cShape = cTy.getShape();
    // Check if all 3d or all 2d
    if (aShape.size() != 2 && aShape.size() != 3)
      return op->emitOpError("expected operands to be 2d or 3d");
    if (aShape.size() != bShape.size() || aShape.size() != cShape.size())
      return op->emitOpError("expected all operands to have the same rank");
    // Check if the first two operands share a common dimension
    // TODO: enable back with an interface to support scaled dot.
    // if (aShape[aShape.size() - 1] != bShape[aShape.size() - 2])
    //   return op->emitOpError("expected the last dimension of the first
    //   operand "
    //                          "to be equal to the second-to-last dimension of
    //                          " "the second operand");
    // Check the batch dimension
    if (aShape.size() == 3 &&
        (aShape[0] != cShape[0] || bShape[0] != cShape[0]))
      return op->emitOpError("expected the first dimension of the first "
                             "operand to be equal to the first dimension of "
                             "the result");
    // Check the output shape
    if (cShape[cShape.size() - 2] != aShape[aShape.size() - 2] ||
        cShape[cShape.size() - 1] != bShape[aShape.size() - 1])
      return op->emitOpError(
          "expected the output shape to be the concatenation of the last "
          "dimension of the first operand and the last dimension of the "
          "second ");
    return success();
  }
};

template <typename ConcreteType>
class SameOperandsAndResultEncoding
    : public TraitBase<ConcreteType, SameOperandsAndResultEncoding> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsAndResultEncoding(op);
  }
};

template <typename ConcreteType>
class SameOperandsEncoding
    : public TraitBase<ConcreteType, SameOperandsEncoding> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsEncoding(op);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsShape
    : public TraitBase<ConcreteType, SameLoadStoreOperandsShape> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameLoadStoreOperandsShape(op);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsAndResultShape
    : public TraitBase<ConcreteType, SameLoadStoreOperandsAndResultShape> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameLoadStoreOperandsAndResultShape(op);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsEncoding
    : public TraitBase<ConcreteType, SameLoadStoreOperandsEncoding> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsEncoding(op,
                                            /*allowTensorPointerType=*/true);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsAndResultEncoding
    : public TraitBase<ConcreteType, SameLoadStoreOperandsAndResultEncoding> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsAndResultEncoding(
        op, /*allowTensorPointerType=*/true);
  }
};

} // namespace OpTrait
} // namespace mlir

#endif
