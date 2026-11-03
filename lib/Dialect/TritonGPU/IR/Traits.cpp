#include "triton/Dialect/TritonGPU/IR/Traits.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::triton::gpu;

LogicalResult OpTrait::impl::verifyEquivalentMemDescType(Type typeA,
                                                         Type typeB) {
  auto memdescA = dyn_cast<MemDescType>(typeA);
  auto memdescB = dyn_cast<MemDescType>(typeB);
  if (!memdescA || !memdescB)
    return success(memdescA == memdescB);
  if (memdescA.getShape() != memdescB.getShape())
    return failure();
  if (memdescA.getAllocShape() != memdescB.getAllocShape())
    return failure();
  if (memdescA.getElementType() != memdescB.getElementType())
    return failure();
  if (memdescA.getMemorySpace() != memdescB.getMemorySpace())
    return failure();
  if (memdescA.getMutableMemory() != memdescB.getMutableMemory())
    return failure();

  Attribute encodingA = memdescA.getEncoding();
  Attribute encodingB = memdescB.getEncoding();
  if (encodingA == encodingB)
    return success();
  if (static_cast<bool>(encodingA) != static_cast<bool>(encodingB))
    return failure();

  auto layoutInterface =
      cast<triton::DialectInferLayoutInterface>(&encodingA.getDialect());
  return layoutInterface->verifyLayoutsAreEqual(memdescA.getShape(), encodingA,
                                                encodingB, {});
}

// Check that the Triton layouts on op's operands and return types are valid.
// For example, we check that the number of warps per block in a Triton GPU
// blocked layout matches that of its module.
//
// It's a little weird to check these properties of a layout only when the
// layout is used in an op, since most of the properties don't actually depend
// on the op.  They do depend on the *module*, though, and a layout is attached
// to a module only by virtue of being used in one of the module's ops.
LogicalResult OpTrait::impl::verifyMemDescLayouts(Operation *op) {
  auto checkLayout = [&](Value val, auto makeErr) -> LogicalResult {
    auto memDescTy = dyn_cast<MemDescType>(val.getType());
    if (!memDescTy)
      return success();

    mlir::Attribute layout = memDescTy.getEncoding();
    if (!layout)
      return success();

    Dialect &dialect = layout.getDialect();
    auto verifyLayoutInterface =
        dyn_cast<mlir::triton::DialectVerifyTensorLayoutInterface>(&dialect);
    if (verifyLayoutInterface) {
      return verifyLayoutInterface->verifyMemDescLayout(layout, memDescTy, op,
                                                        makeErr);
    }

    return success();
  };

  for (size_t i = 0; i < op->getNumOperands(); i++) {
    auto operand = op->getOperand(i);
    auto err = checkLayout(operand, [&]() {
      // Stringify the operand using `printAsOperand`.  This prints e.g. "%42"
      // rather than the full definition.
      std::string operandStr;
      llvm::raw_string_ostream os(operandStr);
      // If we don't assume verified, dump() will recursively call this
      // function!
      operand.printAsOperand(os, OpPrintingFlags().assumeVerified());

      return op->emitError("Operand ")
             << i << " (" << operand << ") has an invalid layout: ";
    });
    if (!err.succeeded())
      return err;
  }

  for (size_t i = 0; i < op->getNumResults(); i++) {
    auto result = op->getResult(i);
    auto err = checkLayout(result, [&]() {
      if (op->getNumResults() == 1) {
        return op->emitError("Result has an invalid layout: ");
      } else {
        return op->emitError("Result ") << i << " has an invalid layout: ";
      }
    });
    if (!err.succeeded())
      return err;
  }

  return success();
}
