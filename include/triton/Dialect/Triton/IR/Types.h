#ifndef TRITON_IR_TYPES_H_
#define TRITON_IR_TYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/Triton/IR/Types.h.inc"

namespace mlir {

unsigned getPointeeBitWidth(RankedTensorType tensorTy);

}

#endif // TRITON_IR_TYPES_H_
