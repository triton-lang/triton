#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include <llvm-6.0/llvm/Support/ErrorHandling.h>

#include "triton/Dialect/TritonGPU/IR/Dialect.cpp.inc"

using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.cpp.inc"

mlir::Attribute 
TritonGPUDistributedEncodingAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  llvm_unreachable("Not implemented");
}

void TritonGPUDistributedEncodingAttr::print(mlir::AsmPrinter &printer) const {
  llvm_unreachable("Not implemented");
}

mlir::Attribute 
TritonGPUMmaEncodingAttr::parse(mlir::AsmParser &parser, ::mlir::Type type) {
  llvm_unreachable("Not implemented");
}

void TritonGPUMmaEncodingAttr::print(mlir::AsmPrinter &printer) const {
  llvm_unreachable("Not implemented");
}

mlir::Attribute
TritonGPUSharedEncodingAttr::parse(mlir::AsmParser &parser, ::mlir::Type type) {
  llvm_unreachable("Not implemented");
}

void TritonGPUSharedEncodingAttr::print(mlir::AsmPrinter &printer) const {
  llvm_unreachable("Not implemented");
}

void TritonGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"
  >();
}


#define GET_OP_CLASSES
#include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"
