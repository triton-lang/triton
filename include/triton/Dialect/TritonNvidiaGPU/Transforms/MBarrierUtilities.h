#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::nvidia_gpu {

bool isCrossCTAMBarrier(Value barrier, int numCTAs);

void getCrossCTAConsumerBarriers(Operation *op,
                                 SmallVectorImpl<Value> &barriers);

bool isCrossCTAConsumer(Operation *op, Value barrier);

bool requiresCrossCTAMBarrierInitSync(FunctionOpInterface funcOp,
                                      Value barrier, int numCTAs);

} // namespace mlir::triton::nvidia_gpu
