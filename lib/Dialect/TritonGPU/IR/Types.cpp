#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton::gpu;

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/Types.cpp.inc"

Type TokenType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  int type = 1;
  if (parser.parseInteger(type))
    return Type();

  if (parser.parseGreater())
    return Type();

  return TokenType::get(parser.getContext(), type);
}

void TokenType::print(AsmPrinter &printer) const {
  printer << "<" << getType() << ">";
}

static constexpr llvm::StringRef kMutableMemory = "mutable";

Type MemDescType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SmallVector<int64_t> dimensions; // required
  if (parser.parseDimensionList(dimensions, /*allowDynamic=*/false)) {
    return Type();
  }

  Type elementType; // required
  if (failed(parser.parseType(elementType))) {
    return Type();
  }

  Attribute encoding; // required
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseAttribute(encoding))) {
      return Type();
    }
  }

  Attribute memorySpace; // required
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseAttribute(memorySpace))) {
      return Type();
    }
  }

  bool mutableMemory = false;                   // optional
  SmallVector<int64_t> allocShape = dimensions; // optional
  if (succeeded(parser.parseOptionalComma())) {
    if (succeeded(parser.parseOptionalKeyword(kMutableMemory))) {
      mutableMemory = true;
    } else {
      if (failed(parser.parseDimensionList(allocShape, /*allowDynamic=*/false,
                                           /*withTrailingX=*/false))) {
        return Type();
      }
    }
  }

  if (mutableMemory && succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseDimensionList(allocShape, /*allowDynamic=*/false,
                                         /*withTrailingX=*/false))) {
      return Type();
    }
  }

  if (parser.parseGreater())
    return Type();
  return MemDescType::get(parser.getContext(), dimensions, elementType,
                          encoding, memorySpace, mutableMemory, allocShape);
}

void MemDescType::print(AsmPrinter &printer) const {
  printer << "<";
  for (auto dim : getShape())
    printer << dim << "x";
  printer << getElementType();
  printer << ", " << getEncoding();
  printer << ", " << getMemorySpace();
  if (getMutableMemory())
    printer << ", " << kMutableMemory;
  auto allocShape = getAllocShape();
  if (allocShape != getShape()) {
    printer << ", ";
    for (auto [i, dim] : llvm::enumerate(allocShape)) {
      printer << dim;
      if (i != allocShape.size() - 1)
        printer << "x";
    }
  }
  printer << ">";
}

//===----------------------------------------------------------------------===//
// Triton Dialect
//===----------------------------------------------------------------------===//
void ::mlir::triton::gpu::TritonGPUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/TritonGPU/IR/Types.cpp.inc"
      >();
}
