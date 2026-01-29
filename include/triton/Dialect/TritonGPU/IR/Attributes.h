#ifndef TRITON_DIALECT_TRITONGPU_IR_ATTRIBUTES_H_
#define TRITON_DIALECT_TRITONGPU_IR_ATTRIBUTES_H_

#include "mlir/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/CGAEncodingAttr.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"

#include "triton/Dialect/TritonGPU/IR/OpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/AttrDefs.h.inc"

#endif // TRITON_DIALECT_TRITONGPU_IR_ATTRIBUTES_H_
