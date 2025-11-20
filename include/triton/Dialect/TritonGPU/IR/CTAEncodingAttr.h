#ifndef TRITON_DIALECT_TRITONGPU_IR_CTAENCODINGATTR_H_
#define TRITON_DIALECT_TRITONGPU_IR_CTAENCODINGATTR_H_

#include "mlir/IR/Attributes.h"
#include "triton/Tools/LinearLayout.h"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/CTAEncodingAttr.h.inc"
#undef GET_ATTRDEF_CLASSES

#endif // TRITON_DIALECT_TRITONGPU_IR_CTAENCODINGATTR_H_
