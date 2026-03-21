#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton;

#include "triton/Dialect/Triton/IR/TypeInterfaces.cpp.inc"

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

  int addressSpace = 1;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseInteger(addressSpace))
      return Type();
  }

  if (parser.parseGreater())
    return Type();

  return PointerType::get(pointeeType, addressSpace);
}

void PointerType::print(AsmPrinter &printer) const {
  if (getAddressSpace() == 1) {
    printer << "<" << getPointeeType() << ">";
  } else {
    printer << "<" << getPointeeType() << ", " << getAddressSpace() << ">";
  }
}

LogicalResult PointerType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type pointeeType, int addressSpace) {
  if (isa<RankedTensorType>(pointeeType)) {
    return emitError() << "pointer types cannot point to ranked tensor types";
  }
  return success();
}

namespace mlir {

namespace triton {

unsigned getPointeeBitWidth(Type type) {
  auto pointeeType = getPointeeType(type);
  if (auto tensorTy = dyn_cast<RankedTensorType>(pointeeType))
    return tensorTy.getElementType().getIntOrFloatBitWidth();
  return pointeeType.getIntOrFloatBitWidth();
}

Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return tensorTy.clone(i1Type);
  return i1Type;
}

Type getPointeeType(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    // Tensor of pointers
    auto ptrType = dyn_cast<PointerType>(tensorTy.getElementType());
    Type pointeeType = ptrType.getPointeeType();
    return tensorTy.clone(pointeeType);
  } else if (auto ptrType = dyn_cast<PointerType>(type)) {
    // scalar pointer
    Type pointeeType = ptrType.getPointeeType();
    return pointeeType;
  }
  return type;
}

Type getI32SameShape(Type type) {
  auto i32Type = IntegerType::get(type.getContext(), 32);
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return tensorTy.clone(i32Type);
  return i32Type;
}

Type getPointerTypeSameShape(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    Type elementType = tensorTy.getElementType();
    PointerType ptrType = PointerType::get(elementType, 1);
    return tensorTy.clone(ptrType);
  } else {
    return PointerType::get(type, 1);
  }
}

Type getPointerTypeToElement(Type type) {
  Type elementType = getElementTypeOrSelf(type);
  PointerType ptrType = PointerType::get(elementType, 1);
  return ptrType;
}

// upstream Triton only uses address space 1 for Pointer Type
Type getPointerType(Type type, int addressSpace) {
  return PointerType::get(type, addressSpace);
}

int getAddressSpace(Type type) {
  if (auto ptrType = dyn_cast<PointerType>(type))
    return ptrType.getAddressSpace();
  return 1;
}

} // namespace triton

} // namespace mlir
