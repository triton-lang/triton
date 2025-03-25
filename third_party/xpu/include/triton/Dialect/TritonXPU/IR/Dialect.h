//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_DIALECT_TRITONXPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONXPU_IR_DIALECT_H_

// TritonXPUDialect
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h" // cf
#include "triton/Dialect/Triton/IR/Dialect.h"           // arith/scf/math/triton
#include "triton/Dialect/TritonGPU/IR/Dialect.h"        // SliceEncodingAttr

#include "triton/Dialect/TritonXPU/IR/Dialect.h.inc" // TritonXPUDialect

// TritonXPUAttr
#include "mlir/IR/Attributes.h"
#include "triton/Dialect/TritonXPU/IR/TritonXPUAttrInterfaces.h.inc"
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonXPU/IR/TritonXPUAttrDefs.h.inc"

// TritonXPUOps
#define GET_OP_CLASSES
#include "triton/Dialect/TritonXPU/IR/Ops.h.inc"

// TritonXPUTypes
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonXPU/IR/Types.h.inc"

namespace mlir {
namespace triton {
namespace xpu {

unsigned getTotalElemsPerThread(Type eltTy);

unsigned getTotalElemsPerThread(Attribute layout, ArrayRef<int64_t> shape,
                                Type eltTy);

unsigned getGroupSize(Attribute layout);

// Return a blocked encoding where the shape is distributed contiguously amongst
// the threads, warps, CTAs with 1 element per threads.
triton::xpu::ClusterLayoutAttr
getDefaultClusterEncoding(MLIRContext *context, ArrayRef<int64_t> shape,
                          uint32_t buffer_size, uint32_t core_num);

SmallVector<unsigned>
getCoresPerClusterWithUniqueData(Attribute layout,
                                 ArrayRef<int64_t> tensorShape);

SmallVector<unsigned>
getCoresPerGroupWithUniqueData(Attribute layout, ArrayRef<int64_t> tensorShape);

SmallVector<unsigned> getUniqueContigPerCore(Attribute layout,
                                             ArrayRef<int64_t> shape);

} // namespace xpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONXPU_IR_DIALECT_H_
