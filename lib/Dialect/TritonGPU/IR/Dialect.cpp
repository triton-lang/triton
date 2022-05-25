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

namespace mlir {
namespace triton {

// Type inference
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i1Type, tensorType.getEncoding());
  return Type();
}

static Type getPointeeType(Type type) {
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    // Tensor of pointers
    auto shape = tensorType.getShape();
    auto ptrType = tensorType.getElementType().dyn_cast<PointerType>();
    Type pointeeType = ptrType.getPointeeType();
    return RankedTensorType::get(shape, pointeeType, tensorType.getEncoding());
  } else if (auto ptrType = type.dyn_cast<PointerType>()) {
    // scalar pointer
    Type pointeeType = ptrType.getPointeeType();
    return pointeeType;
  }
  return Type();
}

}
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
