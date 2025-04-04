#ifndef PROTONGPU_IR_TYPES_H_
#define PROTONGPU_IR_TYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "proton/dialect/include/Dialect/ProtonGPU/IR/Types.h.inc"

namespace mlir {

namespace triton::proton::gpu {} // namespace triton::proton::gpu

} // namespace mlir

#endif // PROTONGPU_IR_TYPES_H_
