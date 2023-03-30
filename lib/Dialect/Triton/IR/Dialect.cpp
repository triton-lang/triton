#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "triton/Dialect/Triton/IR/AttrInterfaces.h.inc"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/DialectImplementation.h"

#include "mlir/Transforms/InliningUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton;

//===----------------------------------------------------------------------===//
// TritonDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct TritonInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       IRMapping &) const final {
    return true;
  }
};
} // namespace

void TritonDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/Triton/IR/Ops.cpp.inc"
      >();

  // We can also add interface here.
  addInterfaces<TritonInlinerInterface>();
}

Operation *TritonDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return builder.create<arith::ConstantOp>(loc, type, value);
}

namespace mlir {

namespace triton {

// Type inference
Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i1Type,
                                 tensorType.getEncoding());
  return Type();
}

Type getPointeeType(Type type) {
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    // Tensor of pointers
    auto shape = tensorType.getShape();
    auto ptrType = tensorType.getElementType().dyn_cast<PointerType>();
    Type pointeeType = ptrType.getPointeeType();
    return RankedTensorType::get(shape, pointeeType, tensorType.getEncoding());
  } else if (auto ptrType = type.dyn_cast<PointerType>()) {
    // scalar pointer
    Type pointeeType = ptrType.getPointeeType();
    return pointeeType;
  }
  return Type();
}

Type getI32SameShape(Type type) {
  auto i32Type = IntegerType::get(type.getContext(), 32);
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i32Type,
                                 tensorType.getEncoding());
  return i32Type;
}

Type getPointerTypeSameShape(Type type) {
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    Type elementType = tensorType.getElementType();
    auto shape = tensorType.getShape();
    PointerType ptrType = PointerType::get(elementType, 1);
    return RankedTensorType::get(shape, ptrType, tensorType.getEncoding());
  } else {
    return PointerType::get(type, 1);
  }
}

Type getPointerType(Type type) { return PointerType::get(type, 1); }

Type getElementTypeOfTensorPointerType(Type type) {
  if (auto ptrType = type.dyn_cast<PointerType>())
    if (auto tensorType = ptrType.getPointeeType().dyn_cast<RankedTensorType>())
      return tensorType.getElementType();
  return {};
}

} // namespace triton
} // namespace mlir
