#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

#include "triton/Dialect/TritonInstrument/IR/Dialect.cpp.inc"
using namespace mlir::triton::instrument;

void TritonInstrumentDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonInstrument/IR/Ops.cpp.inc"
      >();
}
