#ifndef DIALECT_PROTON_IR_DIALECT_H_
#define DIALECT_PROTON_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "proton/dialect/include/Dialect/Proton/IR/Dialect.h.inc"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GET_ATTRDEF_CLASSES
#include "proton/dialect/include/Dialect/Proton/IR/ProtonAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "proton/dialect/include/Dialect/Proton/IR/Ops.h.inc"

#endif // DIALECT_PROTON_IR_DIALECT_H_
