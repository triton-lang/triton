#include "triton/Dialect/TritonGPU/IR/Traits.h"
#include "triton/Analysis/Utility.h"

mlir::LogicalResult
mlir::OpTrait::impl::verifyResultsAreSharedEncoding(Operation *op) {
  if (failed(verifyAtLeastNResults(op, 1)))
    return failure();

  for (auto result : op->getResults())
    if (!isSharedEncoding(result))
      return op->emitOpError() << "requires all results to be shared encoding";

  return success();
};
