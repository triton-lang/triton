#include "triton/Dialect/TritonGPUROCM/IR/Traits.h"
#include "triton/Dialect/TritonGPUROCM/IR/Dialect.h"

mlir::LogicalResult
mlir::OpTrait::impl::verifyResultsAreSharedEncodingROCM(Operation *op) {
  if (failed(verifyAtLeastNResults(op, 1)))
    return failure();

  for (auto result : op->getResults())
    if (!triton::gpu_rocm::isSharedEncoding(result))
      return op->emitOpError() << "requires all results to be shared encoding";

  return success();
};
