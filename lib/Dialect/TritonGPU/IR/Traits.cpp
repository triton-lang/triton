#include "triton/Dialect/TritonGPU/IR/Traits.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

mlir::LogicalResult
mlir::OpTrait::impl::verifyResultsAreSharedEncoding(Operation *op) {
  if (failed(verifyAtLeastNResults(op, 1)))
    return failure();

  for (auto result : op->getResults())
    if (!triton::gpu::isSharedEncoding(result))
      return op->emitOpError() << "requires all results to be shared encoding";

  return success();
};

mlir::LogicalResult
mlir::OpTrait::impl::verifyOperandAndResultHaveSameEncoding(Operation *op) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
    return failure();
  }

  auto operandType = op->getOperand(0).getType().dyn_cast<RankedTensorType>();
  auto resultType = op->getResult(0).getType().dyn_cast<RankedTensorType>();

  if (!operandType || !resultType) {
    return failure();
  }
  auto operandLayout = operandType.getEncoding();
  auto resultLayout = resultType.getEncoding();

  if (auto blockedLayoutSrc =
          dyn_cast<triton::gpu::BlockedEncodingAttr>(operandLayout)) {
    auto blockedLayoutRes =
        dyn_cast<triton::gpu::BlockedEncodingAttr>(resultLayout);
    if (!blockedLayoutRes) {
      return op->emitOpError()
             << "requires operand and result to have same layout";
    }

    if (!triton::gpu::sameBlockedEncodings(blockedLayoutSrc,
                                           blockedLayoutRes)) {
      return op->emitOpError()
             << "requires operand and result to have same layout";
    }
  } else if (auto mfmaLayoutSrc =
                 dyn_cast<triton::gpu::MfmaEncodingAttr>(operandLayout)) {
    auto mfmaLayoutRes = dyn_cast<triton::gpu::MfmaEncodingAttr>(resultLayout);
    if (!mfmaLayoutRes) {
      return op->emitOpError()
             << "requires operand and result to have same layout";
    }
    if (!triton::gpu::sameMfmaEncodings(mfmaLayoutSrc, mfmaLayoutRes)) {
      return op->emitOpError()
             << "requires operand and result to have same layout";
    }
  } else {
    assert(false &&
           "Unexpected Layout in verifyOperandAndResultHaveSmeEncoding");
  }

  return success();
};
