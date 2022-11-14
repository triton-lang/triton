#ifndef TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// TritonGPU depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h.inc"
#include "triton/Dialect/TritonGPU/IR/Traits.h"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/Triton/IR/AttrInterfaces.h.inc"
#include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonGPU/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace gpu {

unsigned getElemsPerThread(Type type);

SmallVector<unsigned> getThreadsPerWarp(Attribute layout);

SmallVector<unsigned> getWarpsPerCTA(Attribute layout);

SmallVector<unsigned> getSizePerThread(Attribute layout);

SmallVector<unsigned> getThreadsPerCTA(const Attribute &layout);

SmallVector<unsigned> getShapePerCTA(const Attribute &layout);

SmallVector<unsigned> getOrder(const Attribute &layout);

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_
