#ifndef DIALECT_PROTONGPU_IR_DIALECT_H_
#define DIALECT_PROTONGPU_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/dialect/include/Dialect/ProtonGPU/IR/Dialect.h.inc"
#include "proton/dialect/include/Dialect/ProtonGPU/IR/OpsEnums.h.inc"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "proton/dialect/include/Dialect/ProtonGPU/IR/Ops.h.inc"

#define GET_ATTRDEF_CLASSES
#include "proton/dialect/include/Dialect/ProtonGPU/IR/AttrDefs.h.inc"

namespace mlir {
namespace triton {
namespace proton {
namespace gpu {

const int getBytesPerClockEntry();

} // namespace gpu
} // namespace proton
} // namespace triton
} // namespace mlir

#endif // DIALECT_PROTONGPU_IR_DIALECT_H_
