#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"

namespace mlir {
namespace triton {

// Type inference
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i1Type, tensorType.getEncoding());
  return Type();
}

static Type getI32SameShape(Type type) {
  auto i32Type = IntegerType::get(type.getContext(), 32);
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i32Type, tensorType.getEncoding());
  return Type();
}

static Type getPointerTypeFromTensor(Type type) {
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    Type elementType = tensorType.getElementType();
    auto shape = tensorType.getShape();
    PointerType ptrType = PointerType::get(elementType, 1);
    return RankedTensorType::get(shape, ptrType, tensorType.getEncoding());
  }
  return Type();
}

}
}

#define GET_OP_CLASSES
#include "triton/Dialect/Triton/IR/Ops.cpp.inc"

// enum attribute definitions
#include "triton/Dialect/Triton/IR/OpsEnums.cpp.inc"

namespace mlir {
namespace triton {

//-- StoreOp --
// Default mask
void StoreOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state, ::mlir::Value ptr, ::mlir::Value value) {
  TensorType ptrType = ptr.getType().dyn_cast<TensorType>();
  auto shape = ptrType.getShape();
  ::mlir::Value mask = builder.create<arith::ConstantOp>(
    ptr.getLoc(),
    RankedTensorType::get(shape, builder.getI1Type()),
    DenseIntElementsAttr::get(
      RankedTensorType::get(shape, builder.getI1Type()), true
    )
  );
  state.addOperands(ptr);
  state.addOperands(value);
  state.addOperands(mask);
}

//-- LoadOp --
void LoadOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state, ::mlir::Value ptr,
                   ::mlir::triton::CacheModifier cache, ::mlir::triton::EvictionPolicy evict, bool isVolatile) {
  TensorType ptrType = ptr.getType().dyn_cast<TensorType>();
  Type elementType = ptrType.getElementType().dyn_cast<PointerType>().getPointeeType();
  auto shape = ptrType.getShape();
  // mask
  ::mlir::Value mask = builder.create<arith::ConstantOp>(
    ptr.getLoc(),
    RankedTensorType::get(shape, builder.getI1Type()),
    DenseIntElementsAttr::get(
      RankedTensorType::get(shape, builder.getI1Type()), true
    )
  );
  // other
  Type resultType = RankedTensorType::get(shape, elementType);
  ::mlir::Value other = builder.create<arith::ConstantOp>(
    ptr.getLoc(),
    resultType,
    DenseElementsAttr::get(
      resultType, builder.getZeroAttr(elementType)
    )
  );
  state.addOperands(ptr);
  state.addOperands(mask);
  state.addOperands(other);
  state.addAttribute(cacheAttrName(state.name), ::mlir::triton::CacheModifierAttr::get(builder.getContext(), cache));
  state.addAttribute(evictAttrName(state.name), ::mlir::triton::EvictionPolicyAttr::get(builder.getContext(), evict));
  state.addAttribute(isVolatileAttrName(state.name), builder.getBoolAttr(isVolatile));
  state.addTypes({resultType});
}

//-- DotOp --

//-- BroadcastOp --
OpFoldResult BroadcastOp::fold(ArrayRef<Attribute> operands) {
  auto constOperand = src().getDefiningOp<arith::ConstantOp>();
  if (!constOperand)
    return {};

  auto shapedType = getType().cast<ShapedType>();

  return SplatElementsAttr::get(shapedType, {constOperand.getValue()});
}

} // namespace triton
} // namespace mlir
