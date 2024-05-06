#include "triton/Dialect/TritonCPU/IR/Types.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton::cpu;

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonCPU/IR/Types.cpp.inc"

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

//===----------------------------------------------------------------------===//
// Triton Dialect
//===----------------------------------------------------------------------===//
void ::mlir::triton::cpu::TritonCPUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/TritonCPU/IR/Types.cpp.inc"
      >();
}
