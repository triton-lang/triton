#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "triton/Dialect/Triton/IR/AttrInterfaces.h.inc"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/DialectImplementation.h"

#include "triton/Dialect/Triton/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton;

void TritonDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/Triton/IR/Ops.cpp.inc"
      >();

  // We can also add interface here.
}

Operation *TritonDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return builder.create<arith::ConstantOp>(loc, type, value);
}

LogicalResult TritonDialect::verifyLayout(Operation *op) {
  bool succeeded = true;
  for (size_t argNo = 0; argNo < op->getNumOperands(); argNo++) {
    Value operand = op->getOperand(argNo);
    auto opType = operand.getType().dyn_cast<RankedTensorType>();
    if (!opType)
      continue;
    Attribute opEncoding = opType.getEncoding();
    if (!opEncoding)
      continue;
    auto verifier = opEncoding.dyn_cast<LayoutVerificationAttrInterface>();
    if (verifier) {
      // llvm::outs() << opType << "\n";
      succeeded =
          succeeded && verifier.verifyLayoutForArg(op, argNo).succeeded();
    }
  }
  return succeeded ? success() : failure();
}