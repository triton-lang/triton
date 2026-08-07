#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::triton::nvidia_gpu {

bool hasTCGen5CommitCrossCTA(Operation *op);

bool requiresCrossCTAMBarrierInitSync(
    FunctionOpInterface funcOp, Value barrier, int numCTAs,
    llvm::function_ref<bool(Value)> aliasesBarrier);

} // namespace mlir::triton::nvidia_gpu
