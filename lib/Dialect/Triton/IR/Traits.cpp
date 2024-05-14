#include "triton/Dialect/Triton/IR/Traits.h"

#include <numeric>

#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;

static LogicalResult verifySameEncoding(Type typeA, Type typeB,
                                        bool allowTensorPointerType) {
  // TODO(Keren): the allowTensorPointerType argument is a hack to allow.
  // The type checking code is kind of a mess with the current design.
  auto getEncoding = [=](Type type) -> Attribute {
    Attribute ret;
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      ret = tensorType.getEncoding();
    }
    if (!allowTensorPointerType) {
      assert(!triton::isTensorPointerType(type));
    }
    return ret;
  };
  auto encodingA = getEncoding(typeA);
  auto encodingB = getEncoding(typeB);
  if (!encodingA || !encodingB)
    return success();
  return encodingA == encodingB ? success() : failure();
}

LogicalResult
OpTrait::impl::verifySameOperandsEncoding(Operation *op,
                                          bool allowTensorPointerType) {
  if (failed(verifyAtLeastNOperands(op, 1)))
    return failure();

  auto type = op->getOperand(0).getType();
  for (auto opType : llvm::drop_begin(op->getOperandTypes(), 1))
    if (failed(verifySameEncoding(opType, type, allowTensorPointerType)))
      return op->emitOpError() << "requires the same encoding for all operands";

  return success();
}

LogicalResult OpTrait::impl::verifySameOperandsAndResultEncoding(
    Operation *op, bool allowTensorPointerType) {
  if (op->getNumOperands() == 0)
    return success();

  if (failed(verifyAtLeastNOperands(op, 1)) ||
      failed(verifyAtLeastNResults(op, 1)))
    return failure();

  auto type = op->getOperand(0).getType();
  for (auto resultType : op->getResultTypes())
    if (failed(verifySameEncoding(resultType, type, allowTensorPointerType)))
      return op->emitOpError()
             << "requires the same encoding for all operands and results";

  return verifySameOperandsEncoding(op, allowTensorPointerType);
}

LogicalResult OpTrait::impl::verifyTensorSize(Operation *op) {
  for (auto opType : op->getOperandTypes()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(opType)) {
      int64_t numElements = 1;
      for (int64_t s : tensorType.getShape())
        numElements *= s;
      if (numElements > maxTensorNumElements)
        return op->emitError("Maximum allowed number of elements is ")
               << maxTensorNumElements << ", but " << *op
               << " has more than that";
      if ((numElements & (numElements - 1)) != 0)
        return op->emitError("Number of elements must be power-of-two, but ")
               << *op << " doesn't follow the rule (" << numElements << ")"
               << " elements";
    }
  }
  for (auto opType : op->getResultTypes()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(opType)) {
      int64_t numElements = 1;
      for (int64_t s : tensorType.getShape())
        numElements *= s;
      if (numElements > maxTensorNumElements)
        return op->emitError("Maximum allowed number of elements is ")
               << maxTensorNumElements << ", but " << *op
               << " has more than that";
      if ((numElements & (numElements - 1)) != 0)
        return op->emitError("Number of elements must be power-of-two, but ")
               << *op << " doesn't follow the rule (" << numElements << ")"
               << " elements";
    }
  }
  return success();
}

// Check that the Triton layouts on op's operands and return types are valid.
// For example, we check that the number of warps per block in a Triton GPU
// blocked layout matches that of its module.
//
// It's a little weird to check these properties of a layout only when the
// layout is used in an op, since most of the properties don't actually depend
// on the op.  They do depend on the *module*, though, and a layout is attached
// to a module only by virtue of being used in one of the module's ops.
LogicalResult OpTrait::impl::verifyTensorLayouts(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  auto checkLayout = [&](Value val, auto makeErr) -> LogicalResult {
    // Only ranked tensors can have layouts.
    auto rankedTy = dyn_cast<RankedTensorType>(val.getType());
    if (!rankedTy)
      return success();

    mlir::Attribute layout = rankedTy.getEncoding();
    if (!layout)
      return success();

    if (isa<ttg::SharedEncodingAttr>(layout))
      return makeErr() << "Shared layout is not allowed on tensor type.";
    // TODO(jlebar): Currently this only checks blocked layouts, but other
    // layouts also have invariants!

    // TODO(jlebar): Handle the case when the encoding is nested within tt.ptr.
    if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(layout)) {
      // A different verifier should have checked that the layout itself is
      // valid, including that threads-per-warp has the same rank as
      // warps-per-block etc.
      auto layoutRank = blocked.getThreadsPerWarp().size();
      if (layoutRank != rankedTy.getRank()) {
        return makeErr() << layout << ".\nLayout has rank " << layoutRank
                         << ", but the tensor it's attached to has rank "
                         << rankedTy.getRank() << ".";
      }

      int moduleThreadsPerWarp =
          ttg::TritonGPUDialect::getThreadsPerWarp(module);
      int64_t layoutThreadsPerWarp = product(blocked.getThreadsPerWarp());
      if (layoutThreadsPerWarp != moduleThreadsPerWarp) {
        return makeErr() << layout << ".\nLayout has a total of "
                         << layoutThreadsPerWarp
                         << " threads per warp, but the module specifies "
                         << moduleThreadsPerWarp << " threads per warp.";
      }

      int moduleWarpsPerCTA = ttg::TritonGPUDialect::getNumWarps(module);
      int64_t layoutWarpsPerCTA = product(blocked.getWarpsPerCTA());
      if (layoutWarpsPerCTA != moduleWarpsPerCTA) {
        return makeErr() << layout << ".\nLayout has a total of "
                         << layoutWarpsPerCTA
                         << " warps per CTA, but the module specifies "
                         << moduleWarpsPerCTA << " warps per CTA.";
      }

      if (blocked.getCTALayout().getCTAsPerCGA().size() > 0) {
        int moduleCTAsPerCGA = ttg::TritonGPUDialect::getNumCTAs(module);
        int64_t layoutCTAsPerCGA =
            product(blocked.getCTALayout().getCTAsPerCGA());
        if (layoutCTAsPerCGA != moduleCTAsPerCGA) {
          return makeErr() << layout << ".\nLayout has a total of "
                           << layoutCTAsPerCGA
                           << " CTAs per CGA, but the module specifies "
                           << moduleCTAsPerCGA << " CTAs per CGA.";
        }
      }
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

static ArrayRef<int64_t> getTypeShape(Type type) {
  auto rankedType = dyn_cast<RankedTensorType>(type);
  if (auto ptrType = dyn_cast<triton::PointerType>(type))
    rankedType = dyn_cast<RankedTensorType>(ptrType.getPointeeType());
  return rankedType ? rankedType.getShape() : ArrayRef<int64_t>();
}

LogicalResult OpTrait::impl::verifySameLoadStoreOperandsShape(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)))
    return failure();

  auto firstOperandShape = getTypeShape(op->getOperand(0).getType());
  for (auto type : llvm::drop_begin(op->getOperandTypes(), 1))
    if (failed(verifyCompatibleShape(getTypeShape(type), firstOperandShape)))
      return op->emitOpError() << "requires the same shape for all operands";

  return success();
}

LogicalResult
OpTrait::impl::verifySameLoadStoreOperandsAndResultShape(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)) ||
      failed(verifyAtLeastNResults(op, 1)))
    return failure();

  auto firstOperandShape = getTypeShape(op->getOperand(0).getType());
  for (auto type : op->getResultTypes())
    if (failed(verifyCompatibleShape(getTypeShape(type), firstOperandShape)))
      return op->emitOpError()
             << "requires the same shape for all operands and results";

  return verifySameLoadStoreOperandsShape(op);
}
