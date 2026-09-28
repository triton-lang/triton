#ifndef NV_DIALECT_HOPPER_TRANSFORMS_WSDATAPARTITION_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_WSDATAPARTITION_H_

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {

// Returned before data-partition rewriting so the pass can remove generated
// task IDs and loop markers before falling back.
enum class DataPartitionResult { Success, Retry, UnsupportedAtomicRMW };

DataPartitionResult doDataPartition(triton::FuncOp &funcOp,
                                    unsigned numConsumerGroups);

} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_WSDATAPARTITION_H_
