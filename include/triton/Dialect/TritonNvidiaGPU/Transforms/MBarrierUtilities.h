#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::triton::nvidia_gpu {

bool isCrossCTALoadStore(::mlir::triton::gpu::MemDescType memDescTy,
                         ::mlir::RankedTensorType regTy);

bool isCrossCTAGatherScatter(::mlir::triton::gpu::MemDescType memDescTy,
                             ::mlir::RankedTensorType regTy, unsigned axis);

bool hasTCGen5CommitCrossCTA(Operation *op);

bool requiresCrossCTAMBarrierInitSync(
    FunctionOpInterface funcOp, Value barrier, int numCTAs,
    llvm::function_ref<bool(Value)> aliasesBarrier);

} // namespace mlir::triton::nvidia_gpu
