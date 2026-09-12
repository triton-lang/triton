#ifndef TRITON_GPU_DIALECT_INTERFACES_H
#define TRITON_GPU_DIALECT_INTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "triton/Dialect/TritonGPU/IR/CGAEncodingAttr.h"

namespace mlir::triton::gpu {
LogicalResult verifyMBarrierOpInterface(Operation *op);
}

// clang-format off
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/AttrInterfaces.h.inc"
#include "triton/Dialect/TritonGPU/IR/OpInterfaces.h.inc"
// clang-format on

#endif // TRITON_GPU_DIALECT_INTERFACES_H
