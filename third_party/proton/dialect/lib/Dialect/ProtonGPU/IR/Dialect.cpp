#include "Dialect/ProtonGPU/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "Dialect/ProtonGPU/IR/Dialect.cpp.inc"

const int mlir::triton::proton::gpu::getWarpNumPerGroup() { return 4; }

const int mlir::triton::proton::gpu::getBytesPerClockEntry() { return 8; }

void mlir::triton::proton::gpu::ProtonGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/ProtonGPU/IR/Ops.cpp.inc"
      >();
}
