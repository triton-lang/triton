#include "triton/Dialect.h"
#include "triton/Types.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/DialectImplementation.h"


#include "triton/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton;

void TritonDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "triton/Ops.cpp.inc"
               >();

  // We can also add interface here.
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//
// pointer-type ::= `!triton.ptr<` element-type ` >`
static Type parsePointerType(TritonDialect const &dialect,
                             DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();


  Type pointeeType;
  if (parser.parseType(pointeeType))
    return Type();

  if (parser.parseGreater())
    return Type();

  return PointerType::get(pointeeType);
}

// trtion-type ::= pointer-type
Type TritonDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "ptr")
    return parsePointerType(*this, parser);

  parser.emitError(parser.getNameLoc(), "unknown Triton type: ") << keyword;
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//
static void print(PointerType type, DialectAsmPrinter &os) {
  os << "ptr<" << type.getPointeeType() << ">";
}

void TritonDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<PointerType>( [&](auto type) { print(type, os); })
      .Default([](Type) { llvm_unreachable("unhandled Triton type"); });
}
