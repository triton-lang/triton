#include "mlir/IR/BuiltinTypes.h"

// clang-format off
#include "Dialect/TritonMetalGPU/IR/Dialect.h"
#include "Dialect/TritonMetalGPU/IR/Dialect.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::triton::metalgpu;

void mlir::triton::metalgpu::TritonMetalGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/TritonMetalGPU/IR/Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Dialect/TritonMetalGPU/IR/Ops.cpp.inc"

void SimdgroupStoreOp::build(OpBuilder &builder, OperationState &state,
                           Value ptr, Value value) {
  return SimdgroupStoreOp::build(builder, state, ptr, value, /*mask=*/{});
}
