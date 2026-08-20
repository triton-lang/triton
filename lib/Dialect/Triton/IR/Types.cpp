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

// Format: !tt.tensordesc<128x64xf16>
//         !tt.tensordesc<128x64xf16, #shared>
Type TensorDescType::parse(AsmParser &parser) {
  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (failed(parser.parseLess()))
    return Type();

  SmallVector<int64_t> shape;
  if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/false)))
    return Type();

  Type elementType;
  if (failed(parser.parseType(elementType)))
    return Type();

  Attribute sharedLayout;
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseAttribute(sharedLayout)))
      return Type();
  }

  if (failed(parser.parseGreater()))
    return Type();

  return TensorDescType::getChecked(loc, parser.getContext(), shape,
                                    elementType, sharedLayout);
}

void TensorDescType::print(AsmPrinter &printer) const {
  printer << "<";
  for (auto dim : getShape())
    printer << dim << "x";
  printer << getElementType();
  if (getSharedLayout())
    printer << ", " << getSharedLayout();
  printer << ">";
}

// Format: !tt.ptr<f32>            (defaults to the "global" address space)
//         !tt.ptr<f32, "flat">
Type PointerType::parse(AsmParser &parser) {
  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess())
    return Type();

  Type pointeeType;
  if (parser.parseType(pointeeType))
    return Type();

  PtrAddrSpace addressSpace = PtrAddrSpace::Global;
  if (succeeded(parser.parseOptionalComma())) {
    std::string name;
    if (parser.parseString(&name))
      return Type();
    std::optional<PtrAddrSpace> symbolized = symbolizePtrAddrSpace(name);
    if (!symbolized) {
      parser.emitError(parser.getCurrentLocation())
          << "invalid pointer address space '" << name << "'";
      return Type();
    }
    addressSpace = *symbolized;
  }

  if (parser.parseGreater())
    return Type();

  return PointerType::getChecked(loc, pointeeType, addressSpace);
}

void PointerType::print(AsmPrinter &printer) const {
  printer << "<" << getPointeeType();
  if (getAddressSpace() != PtrAddrSpace::Global)
    printer << ", \"" << stringifyPtrAddrSpace(getAddressSpace()) << "\"";
  printer << ">";
}

LogicalResult
TensorDescType::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<int64_t> shape, Type elementType,
                       Attribute sharedLayout) {
  if (isa<RankedTensorType>(elementType)) {
    return emitError()
           << "tensor descriptors must not wrap tensor types; use "
              "!tt.tensordesc<shape x element-type[, layout]> instead";
  }
  return success();
}

LogicalResult PointerType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type pointeeType, PtrAddrSpace addressSpace) {
  if (!pointeeType.isIntOrFloat())
    return emitError()
           << "pointer types must point to integer or floating-point types";
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
    PointerType ptrType = PointerType::get(elementType);
    return tensorTy.clone(ptrType);
  } else {
    return PointerType::get(type);
  }
}

bool elementTypeMatchesPointee(Type valueTy, Type ptrTy) {
  auto ptrType = dyn_cast<PointerType>(ptrTy);
  return ptrType && getElementTypeOrSelf(valueTy) == ptrType.getPointeeType();
}

Type getPointerType(Type type, PtrAddrSpace addressSpace) {
  return PointerType::get(type, addressSpace);
}

PtrAddrSpace getAddressSpace(Type type) {
  if (auto ptrType = dyn_cast<PointerType>(type))
    return ptrType.getAddressSpace();
  return PtrAddrSpace::Global;
}

} // namespace triton

} // namespace mlir
