#ifndef NV_DIALECT_HOPPER_TRANSFORMS_WSDATAPARTITION_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_WSDATAPARTITION_H_

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {

// Retry and Unsupported precede structural rewriting. A failed rewrite must
// terminate the pass because removing task IDs cannot undo it.
enum class DataPartitionResult {
  Success,
  Retry,
  Unsupported,
  FailedAfterRewrite
};

DataPartitionResult doDataPartition(triton::FuncOp &funcOp,
                                    unsigned numConsumerGroups);

} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_WSDATAPARTITION_H_
