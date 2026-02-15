#ifndef DIALECT_PROTONGPU_IR_DIALECT_H_
#define DIALECT_PROTONGPU_IR_DIALECT_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h.inc"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Types.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Ops.h.inc"

#define GET_ATTRDEF_CLASSES
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/AttrDefs.h.inc"

namespace mlir {
namespace triton {
namespace proton {
namespace gpu {

const int getBytesPerClockEntry();

const int getCircularHeaderSize();

const int getTotalNumWarps(ModuleOp mod);

} // namespace gpu
} // namespace proton
} // namespace triton
} // namespace mlir

#endif // DIALECT_PROTONGPU_IR_DIALECT_H_
