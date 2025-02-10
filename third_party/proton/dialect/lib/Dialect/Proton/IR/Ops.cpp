#include "Dialect/Proton/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define GET_OP_CLASSES
#include "Dialect/Proton/IR/Ops.cpp.inc"

#include "Dialect/Proton/IR/OpsEnums.cpp.inc"
