#ifndef PROTONGPU_IR_TYPES_H_
#define PROTONGPU_IR_TYPES_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include "proton/Dialect/include/Dialect/ProtonGPU/IR/OpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Types.h.inc"

#endif // PROTONGPU_IR_TYPES_H_
