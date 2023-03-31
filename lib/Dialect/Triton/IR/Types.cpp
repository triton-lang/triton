#include "triton/Dialect/Triton/IR/Types.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton;

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/Triton/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// Triton Dialect
//===----------------------------------------------------------------------===//
void TritonDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/Triton/IR/Types.cpp.inc"
      >();
}

Type PointerType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  Type pointeeType;
  if (parser.parseType(pointeeType))
    return Type();

  if (parser.parseGreater())
    return Type();

  // TODO: also print address space?
  return PointerType::get(pointeeType, 1);
}

void PointerType::print(AsmPrinter &printer) const {
  printer << "<" << getPointeeType() << ">";
}

namespace mlir {

namespace triton {

unsigned getPointeeBitWidth(Type type) {
  auto pointeeType = getPointeeType(type);
  if (auto tensorTy = pointeeType.dyn_cast<RankedTensorType>())
    return tensorTy.getElementType().getIntOrFloatBitWidth();
  return pointeeType.getIntOrFloatBitWidth();
}

Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorTy = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorTy.getShape(), i1Type,
                                 tensorTy.getEncoding());
  return i1Type;
}

Type getPointeeType(Type type) {
  if (auto tensorTy = type.dyn_cast<RankedTensorType>()) {
    // Tensor of pointers
    auto shape = tensorTy.getShape();
    auto ptrType = tensorTy.getElementType().dyn_cast<PointerType>();
    Type pointeeType = ptrType.getPointeeType();
    return RankedTensorType::get(shape, pointeeType, tensorTy.getEncoding());
  } else if (auto ptrType = type.dyn_cast<PointerType>()) {
    // scalar pointer
    Type pointeeType = ptrType.getPointeeType();
    return pointeeType;
  }
  return type;
}

Type getI32SameShape(Type type) {
  auto i32Type = IntegerType::get(type.getContext(), 32);
  if (auto tensorTy = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorTy.getShape(), i32Type,
                                 tensorTy.getEncoding());
  return i32Type;
}

Type getPointerTypeSameShape(Type type) {
  if (auto tensorTy = type.dyn_cast<RankedTensorType>()) {
    Type elementType = tensorTy.getElementType();
    auto shape = tensorTy.getShape();
    PointerType ptrType = PointerType::get(elementType, 1);
    return RankedTensorType::get(shape, ptrType, tensorTy.getEncoding());
  } else {
    return PointerType::get(type, 1);
  }
}

Type getPointerType(Type type) { return PointerType::get(type, 1); }

bool isTensorPointerType(Type type) {
  if (auto ptrType = type.dyn_cast<PointerType>())
    return ptrType.getPointeeType().isa<RankedTensorType>();
  return false;
}

Type getElementTypeOfTensorPointerType(Type type) {
  if (auto ptrType = type.dyn_cast<PointerType>())
    if (auto tensorTy = ptrType.getPointeeType().dyn_cast<RankedTensorType>())
      return tensorTy.getElementType();
  return {};
}

} // namespace triton

} // namespace mlir
