#ifndef TRITON_DIALECT_TRITONGPU_IR_ATTRIBUTES_H_
#define TRITON_DIALECT_TRITONGPU_IR_ATTRIBUTES_H_

#include "mlir/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/CGAEncodingAttr.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"

#include "triton/Dialect/TritonGPU/IR/OpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/AttrDefs.h.inc"

namespace mlir::triton::gpu {
// Availability belongs to the type, including block arguments and individual
// results. An ordinary encoding has all of its physical copies available.
ThreadAvailabilityAttr getThreadAvailability(Type type);
Attribute getPlacementEncoding(Attribute encoding);
bool availabilityCovers(Type source, Type destination);
LogicalResult verifyThreadAvailability(Operation *op, Type type,
                                       ThreadAvailabilityAttr domain);
} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_IR_ATTRIBUTES_H_
