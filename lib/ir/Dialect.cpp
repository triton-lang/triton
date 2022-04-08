#include "triton/ir/Dialect.h"
#include "triton/ir/Types.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/DialectImplementation.h"


#include "triton/ir/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton;

void TritonDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "triton/ir/Ops.cpp.inc"
               >();

  // We can also add interface here.
}
