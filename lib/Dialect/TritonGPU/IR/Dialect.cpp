#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton::gpu;

// parse an array of integers
static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       const NamedAttribute &attr,
                                       SmallVector<unsigned, 2> &res,
                                       StringRef desc)  {
  auto arrayAttr = attr.getValue().dyn_cast<ArrayAttr>();
  if (!arrayAttr) {
    parser.emitError(parser.getNameLoc(), "expected an array for ")
            << desc;
    return failure();
  }
  for (Attribute i : arrayAttr) {
    auto intAttr = i.dyn_cast<IntegerAttr>();
    if (!intAttr) {
      parser.emitError(parser.getNameLoc(), "expected an integer value in ")
              << desc;
      return failure();
    }
    res.push_back(intAttr.getUInt());
  }
  return success();
};

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.cpp.inc"

Attribute 
TritonGPUShardedEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};
  
  SmallVector<unsigned, 2> threadTileSize;
  SmallVector<unsigned, 2> warpTileSize;
  SmallVector<unsigned, 2> blockTileSize;
  SmallVector<unsigned, 2> order;

  // parse an array of integers
  // auto parseIntArrayAttr = [&parser](const NamedAttribute &attr,
  //                             SmallVector<unsigned, 2> &res,
  //                             StringRef desc) -> LogicalResult {
  //   auto arrayAttr = attr.getValue().dyn_cast<ArrayAttr>();
  //   if (!arrayAttr) {
  //     parser.emitError(parser.getNameLoc(), "expected an array for ")
  //            << desc;
  //     return failure();
  //   }
  //   for (Attribute i : arrayAttr) {
  //     auto intAttr = i.dyn_cast<IntegerAttr>();
  //     if (!intAttr) {
  //       parser.emitError(parser.getNameLoc(), "expected an integer value in ")
  //              << desc;
  //       return failure();
  //     }
  //     res.push_back(intAttr.getUInt());
  //   }
  //   return success();
  // };

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "threadTileSize") {
      if (parseIntArrayAttr(parser, attr, threadTileSize, "thread tile size").failed())
        return {};
    } else if (attr.getName() == "warpTileSize") {
      if (parseIntArrayAttr(parser, attr, warpTileSize, "warp tile size").failed())
        return {};
    } else if (attr.getName() == "blockTileSize") {
      if (parseIntArrayAttr(parser, attr, blockTileSize, "block tile size").failed())
        return {};
    } else if (attr.getName() == "order") {
      if (parseIntArrayAttr(parser, attr, order, "order").failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  return parser.getChecked<TritonGPUShardedEncodingAttr>(parser.getContext(),
                                                         threadTileSize,
                                                         warpTileSize,
                                                         blockTileSize,
                                                         order);
}

void TritonGPUShardedEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<"
          << "threadTileSize = [" << getThreadTileSize() << "]"
          << ", warpTileSize = [" << getWarpTileSize() << "]"
          << ", blockTileSize = [" << getBlockTileSize() << "]"
          << ", order = [" << getOrder() << "]"
          << ">";
}

Attribute 
TritonGPUMmaEncodingAttr::parse(AsmParser &parser, Type type) {
  llvm_unreachable("Not implemented");
}

void TritonGPUMmaEncodingAttr::print(AsmPrinter &printer) const {
  llvm_unreachable("Not implemented");
}

Attribute
TritonGPUSharedEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned vec = 0;
  unsigned perPhase = 0;
  unsigned maxPhase = 0;
  SmallVector<unsigned, 2> order;

  auto parseUInt = [&parser](const NamedAttribute &attr,
                             unsigned &value,
                             StringRef desc) -> LogicalResult {
    auto intAttr = attr.getValue().dyn_cast<IntegerAttr>();
    if (!intAttr) {
      parser.emitError(parser.getNameLoc(), "expected an integer ") << desc;
      return failure();
    }
    value = intAttr.getUInt();
    return success();
  };

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "vec") {
      if (parseUInt(attr, vec, "vec").failed())
        return {};
    } else if (attr.getName() == "perPhase") {
      if (parseUInt(attr, perPhase, "perPhase").failed())
        return {};
    } else if (attr.getName() == "maxPhase") {
      if (parseUInt(attr, maxPhase, "maxPhase").failed())
        return {};
    } else if (attr.getName() == "order") {
      if (parseIntArrayAttr(parser, attr, order, "order").failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
             << attr.getName().strref();
      return {};
    }
  }

  return parser.getChecked<TritonGPUSharedEncodingAttr>(parser.getContext(),
                                                        vec,
                                                        perPhase,
                                                        maxPhase,
                                                        order);
}

void TritonGPUSharedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<"
          << "vec = " << getVec()
          << ", perPhase = " << getPerPhase()
          << ", order = [" << getOrder() << "]"
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
LogicalResult
TritonGPUDialect::verifyOperationAttribute(Operation *op,
                                           NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
