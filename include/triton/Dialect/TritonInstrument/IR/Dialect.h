#ifndef TRITON_DIALECT_TRITONINSTRUMENT_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONINSTRUMENT_IR_DIALECT_H_

// TritonInstrument depends on Triton and TritonGPU
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonInstrument/IR/Dialect.h.inc"
#include "triton/Dialect/TritonInstrument/IR/Ops.h.inc"

namespace mlir::triton::instrument {

Value createPointerTensor(OpBuilder &b, Location loc, Value base,
                          RankedTensorType tensorType);
Operation *createStoreScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                    Value tensor, RankedTensorType tensorType);
Operation *createLoadScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                   RankedTensorType tensorType);
Value expandOuterSlicedDim(OpBuilder &b, Location loc, Value tensor);
Value expandAllSlicedDims(OpBuilder &b, Location loc, Value tensor);

} // namespace mlir::triton::instrument

#endif // TRITON_DIALECT_TRITONINSTRUMENT_IR_DIALECT_H_
