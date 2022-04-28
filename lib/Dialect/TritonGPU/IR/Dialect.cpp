#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.cpp.inc"

using namespace mlir::triton::gpu;

void TritonGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"
  >();
}
