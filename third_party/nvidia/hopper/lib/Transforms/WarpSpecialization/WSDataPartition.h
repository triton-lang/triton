#ifndef NV_DIALECT_HOPPER_TRANSFORMS_WSDATAPARTITION_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_WSDATAPARTITION_H_

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {

enum class DataPartitionResult { Success, Retry, UnsupportedAtomicRMW };

DataPartitionResult doDataPartition(triton::FuncOp &funcOp,
                                    unsigned numConsumerGroups);

} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_WSDATAPARTITION_H_
