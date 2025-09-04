#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::nvidia_gpu {

LogicalResult verifyBarrierType(Operation *op,
                                mlir::triton::gpu::MemDescType barrierType);

}

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_
