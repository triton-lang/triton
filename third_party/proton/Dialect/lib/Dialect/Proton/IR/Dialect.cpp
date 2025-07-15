#include "Dialect/Proton/IR/Dialect.h"
#include "Dialect/Proton/IR/Interfaces.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "Dialect/Proton/IR/Dialect.cpp.inc"

void mlir::triton::proton::ProtonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Proton/IR/Ops.cpp.inc"
      >();
  addInterfaces<ProtonInlinerInterface>();
}
