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
