#ifndef TRITON_DIALECT_TRITONINSTRUMENT_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONINSTRUMENT_IR_DIALECT_H_

// TritonInstrument depends on Triton and TritonGPU
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "triton/Dialect/TritonInstrument/IR/OpsEnums.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonInstrument/IR/Dialect.h.inc"
#include "triton/Dialect/TritonInstrument/IR/Ops.h.inc"

#endif // TRITON_DIALECT_TRITONINSTRUMENT_IR_DIALECT_H_
