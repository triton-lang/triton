#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/DialectImplementation.h"


#include "triton/Dialect/Triton/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton;

void TritonDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/Triton/Ops.cpp.inc"
               >();

  // We can also add interface here.
}
