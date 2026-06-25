#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::triton::nvidia_gpu {

bool isCrossCTAMBarrier(Value barrier, int numCTAs);

bool isCrossCTAConsumer(Operation *op,
                        llvm::function_ref<bool(Value)> aliasesBarrier);

bool isCrossCTAConsumer(Operation *op, Value barrier);

bool requiresCrossCTAMBarrierInitSync(
    FunctionOpInterface funcOp, Value barrier, int numCTAs,
    llvm::function_ref<bool(Value)> aliasesBarrier);

bool requiresCrossCTAMBarrierInitSync(FunctionOpInterface funcOp,
                                      Value barrier, int numCTAs);

} // namespace mlir::triton::nvidia_gpu
