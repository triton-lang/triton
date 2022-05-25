#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

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
  printer << "<"
          << "threadTileSize = " << getThreadTileSize()
          << ", blockTileSize = " << getBlockTileSize()
          << ", order = " << getOrder()
          << ">";
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
  printer << "<"
          // << "threadTileSize = " << getThreadTileSize()
          // << ", blockTileSize = " << getBlockTileSize()
          // << ", order = " << getOrder()
          << ">";
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


// verify TritonGPU ops
mlir::LogicalResult
TritonGPUDialect::verifyOperationAttribute(mlir::Operation *op,
                                           mlir::NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
