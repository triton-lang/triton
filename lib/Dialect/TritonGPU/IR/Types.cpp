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

  SmallVector<int64_t> dimensions;
  if (parser.parseDimensionList(dimensions, /*allowDynamic=*/false))
    return Type();

  // Parse the element type.
  Type elementType;
  if (parser.parseType(elementType))
    return Type();

  Attribute encoding;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseAttribute(encoding))
      return Type();
  }
  bool mutableMemory = false;
  Attribute memorySpace;
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseOptionalKeyword(kMutableMemory))) {
      if (parser.parseAttribute(memorySpace))
        return Type();
    } else {
      mutableMemory = true;
    }
  }
  if (mutableMemory == false && succeeded(parser.parseOptionalComma())) {
    if (parser.parseOptionalKeyword(kMutableMemory))
      return Type();
    mutableMemory = true;
  }
  if (parser.parseGreater())
    return Type();
  return MemDescType::get(parser.getContext(), dimensions, elementType,
                          encoding, memorySpace, mutableMemory);
}

void MemDescType::print(AsmPrinter &printer) const {
  printer << "<";
  for (auto dim : getShape())
    printer << dim << "x";
  printer << getElementType();
  if (getEncoding())
    printer << ", " << getEncoding();
  if (getMemorySpace())
    printer << ", " << getMemorySpace();
  if (getMutableMemory())
    printer << ", " << kMutableMemory;
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
