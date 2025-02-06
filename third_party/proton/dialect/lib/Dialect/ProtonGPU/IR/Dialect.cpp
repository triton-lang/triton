#include "Dialect/ProtonGPU/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "Dialect/ProtonGPU/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton::proton;

void mlir::triton::proton::gpu::ProtonGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/ProtonGPU/IR/ProtonGPUAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/ProtonGPU/IR/Ops.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "Dialect/ProtonGPU/IR/ProtonGPUAttrDefs.cpp.inc"
