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

unsigned getPointeeBitWidth(RankedTensorType tensorTy) {
  auto ptrTy = tensorTy.getElementType().cast<triton::PointerType>();
  auto pointeeType = ptrTy.getPointeeType();
  return pointeeType.getIntOrFloatBitWidth();
}

} // namespace mlir
