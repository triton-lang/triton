#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton;

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/Triton/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// TensorDescType Verifier
//===----------------------------------------------------------------------===//
LogicalResult
TensorDescType::verify(function_ref<InFlightDiagnostic()> emitError,
                       RankedTensorType blockType, mlir::StringAttr mode,
                       mlir::DenseI64ArrayAttr elementStrides,
                       mlir::DenseI64ArrayAttr pixelBoxLowerCorner,
                       mlir::DenseI64ArrayAttr pixelBoxUpperCorner,
                       std::optional<int64_t> channelsPerPixel,
                       std::optional<int64_t> pixelsPerColumn) {
  // Validate mode is either "tiled" or "im2col"
  if (mode.getValue() != "tiled" && mode.getValue() != "im2col") {
    return emitError() << "TensorDescType mode must be 'tiled' or 'im2col', "
                       << "got '" << mode.getValue() << "'";
  }

  // In tiled mode, im2col-specific parameters should not be set
  if (mode.getValue() == "tiled") {
    if (elementStrides) {
      return emitError()
             << "TensorDescType in 'tiled' mode should not have elementStrides";
    }
    if (pixelBoxLowerCorner) {
      return emitError() << "TensorDescType in 'tiled' mode should not have "
                            "pixelBoxLowerCorner";
    }
    if (pixelBoxUpperCorner) {
      return emitError() << "TensorDescType in 'tiled' mode should not have "
                            "pixelBoxUpperCorner";
    }
    if (channelsPerPixel.has_value()) {
      return emitError() << "TensorDescType in 'tiled' mode should not have "
                            "channelsPerPixel";
    }
    if (pixelsPerColumn.has_value()) {
      return emitError() << "TensorDescType in 'tiled' mode should not have "
                            "pixelsPerColumn";
    }
  }

  // In im2col mode, validate blockType shape matches channelsPerPixel and
  // pixelsPerColumn
  if (mode.getValue() == "im2col") {
    // blockType must be rank 2 for im2col mode
    if (blockType.getRank() != 2) {
      return emitError()
             << "TensorDescType in 'im2col' mode requires rank-2 blockType, "
             << "got rank " << blockType.getRank();
    }

    auto shape = blockType.getShape();
    int64_t M = shape[0]; // pixelsPerColumn
    int64_t N = shape[1]; // channelsPerPixel

    // Validate pixelsPerColumn matches M dimension
    if (pixelsPerColumn.has_value() && pixelsPerColumn.value() != M) {
      return emitError() << "TensorDescType in 'im2col' mode: pixelsPerColumn ("
                         << pixelsPerColumn.value()
                         << ") must equal blockType's first dimension (" << M
                         << ")";
    }

    // Validate channelsPerPixel matches N dimension
    if (channelsPerPixel.has_value() && channelsPerPixel.value() != N) {
      return emitError()
             << "TensorDescType in 'im2col' mode: channelsPerPixel ("
             << channelsPerPixel.value()
             << ") must equal blockType's second dimension (" << N << ")";
    }
  }

  return success();
}

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

bool isTensorPointerType(Type type) {
  if (auto ptrType = dyn_cast<PointerType>(type))
    return isa<RankedTensorType>(ptrType.getPointeeType());
  return false;
}

bool isTensorOrTensorPointerType(Type type) {
  return isa<RankedTensorType>(type) || isTensorPointerType(type);
}

Type getElementTypeOfTensorPointerType(Type type) {
  if (auto ptrType = dyn_cast<PointerType>(type))
    if (auto tensorTy = dyn_cast<RankedTensorType>(ptrType.getPointeeType()))
      return tensorTy.getElementType();
  return {};
}

} // namespace triton

} // namespace mlir
