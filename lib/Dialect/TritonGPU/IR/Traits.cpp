#include "triton/Dialect/TritonGPU/IR/Traits.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

mlir::LogicalResult
mlir::OpTrait::impl::verifyResultsAreSharedEncoding(Operation *op) {
  if (failed(verifyAtLeastNResults(op, 1)))
    return failure();

  for (auto result : op->getResults())
    if (!triton::gpu::hasSharedEncoding(result))
      return op->emitOpError() << "requires all results to be shared encoding";

  return success();
};

mlir::LogicalResult
mlir::OpTrait::impl::verifyOperandsAreSharedEncoding(Operation *op) {
  for (auto operand : op->getOperands())
    if (!triton::gpu::hasSharedEncoding(operand))
      return op->emitOpError() << "requires all operands to be shared encoding";

  return success();
};
