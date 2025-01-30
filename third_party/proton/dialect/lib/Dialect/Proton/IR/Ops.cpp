#include "Dialect/Proton/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#define GET_OP_CLASSES
#include "Dialect/Proton/IR/Ops.cpp.inc"
#include "Dialect/Proton/IR/OpsEnums.cpp.inc"

namespace mlir {
namespace triton {
namespace proton {

// -- CircularRecordOp --
LogicalResult CircularRecordOp::verify() {
  // TODO(fywkevin): checks the following:
  // 1. circular buffer size power of 2.
  // 2. function's noinline is false.
  return success();
}

} // namespace proton
} // namespace triton
} // namespace mlir
