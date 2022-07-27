#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton::gpu;

// parse an array of integers
static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       const NamedAttribute &attr,
                                       /*SmallVector<unsigned, 2>*/ auto &res,
                                       StringRef desc) {
  auto arrayAttr = attr.getValue().dyn_cast<ArrayAttr>();
  if (!arrayAttr) {
    parser.emitError(parser.getNameLoc(), "expected an array for ") << desc;
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

static Attribute parseBlocked(AsmParser &parser, Type type) {
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
  SmallVector<unsigned, 2> broadcastAxis;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "threadTileSize") {
      if (parseIntArrayAttr(parser, attr, threadTileSize, "thread tile size")
              .failed())
        return {};
    } else if (attr.getName() == "warpTileSize") {
      if (parseIntArrayAttr(parser, attr, warpTileSize, "warp tile size")
              .failed())
        return {};
    } else if (attr.getName() == "blockTileSize") {
      if (parseIntArrayAttr(parser, attr, blockTileSize, "block tile size")
              .failed())
        return {};
    } else if (attr.getName() == "order") {
      if (parseIntArrayAttr(parser, attr, order, "order").failed())
        return {};
    } else if (attr.getName() == "broadcastAxis") {
      if (parseIntArrayAttr(parser, attr, broadcastAxis, "broadcastAxis")
              .failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  return parser.getChecked<TritonGPUBlockedEncodingAttr>(
      parser.getContext(), threadTileSize, warpTileSize, blockTileSize, order,
      broadcastAxis);
}

static void printBlocked(AsmPrinter &printer, auto *attr) {
  printer << "<{"
          << "threadTileSize = [" << attr->getThreadTileSize() << "]"
          << ", warpTileSize = [" << attr->getWarpTileSize() << "]"
          << ", blockTileSize = [" << attr->getBlockTileSize() << "]"
          << ", order = [" << attr->getOrder() << "]"
          << ", broadcastAxis = [" << attr->getBroadcastAxis() << "]"
          << "}>";
}

Attribute TritonGPUBlockedEncodingAttr::parse(AsmParser &parser, Type type) {
  parseBlocked(parser, type);
}

void TritonGPUBlockedEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printBlocked(printer, this);
}

Attribute TritonGPUBlockedMulticastEncodingAttr::parse(AsmParser &parser,
                                                       Type type) {
  parseBlocked(parser, type);
}

void TritonGPUBlockedMulticastEncodingAttr::print(AsmPrinter &printer) const {
  printBlocked(printer, this);
}

static Attribute parseMma(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  SmallVector<unsigned, 2> fragmentPerWarp;
  SmallVector<unsigned, 2> shapePerWarp;
  SmallVector<unsigned, 2> warpPerTile;
  SmallVector<unsigned, 2> shapePerTile;
  SmallVector<unsigned, 2> repetitions;
  SmallVector<unsigned, 2> contigPerThread;
  SmallVector<unsigned, 2> broadcastAxis;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "fragmentPerWarp") {
      if (parseIntArrayAttr(parser, attr, fragmentPerWarp, "fragmentPerWarp")
              .failed())
        return {};
    } else if (attr.getName() == "shapePerWarp") {
      if (parseIntArrayAttr(parser, attr, shapePerWarp, "shapePerWarp")
              .failed())
        return {};
    } else if (attr.getName() == "warpPerTile") {
      if (parseIntArrayAttr(parser, attr, warpPerTile, "warpPerTile").failed())
        return {};
    } else if (attr.getName() == "shapePerTile") {
      if (parseIntArrayAttr(parser, attr, shapePerTile, "shapePerTile")
              .failed())
        return {};
    } else if (attr.getName() == "repetitions") {
      if (parseIntArrayAttr(parser, attr, repetitions, "repetitions").failed())
        return {};
    } else if (attr.getName() == "contigPerThread") {
      if (parseIntArrayAttr(parser, attr, contigPerThread, "contigPerThread")
              .failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  return parser.getChecked<TritonGPUMmaEncodingAttr>(
      parser.getContext(), fragmentPerWarp, shapePerWarp, warpPerTile,
      shapePerTile, repetitions, contigPerThread, broadcastAxis);
}

static void printMma(AsmPrinter &printer, auto *attr) {
  printer << "<{"
          << "fragmentPerWarp = [" << attr->getFragmentPerWarp() << "]"
          << ", shapePerWarp = [" << attr->getShapePerWarp() << "]"
          << ", warpPerTile = [" << attr->getWarpPerTile() << "]"
          << ", shapePerTile = [" << attr->getShapePerTile() << "]"
          << ", repetitions = [" << attr->getRepetitions() << "]"
          << ", contigPerThread = [" << attr->getContigPerThread() << "]"
          << "}>";
}

Attribute TritonGPUMmaEncodingAttr::parse(AsmParser &parser, Type type) {
  return parseMma(parser, type);
}

void TritonGPUMmaEncodingAttr::print(AsmPrinter &printer) const {
  printMma(printer, this);
}

Attribute TritonGPUMmaMulticastEncodingAttr::parse(AsmParser &parser,
                                                   Type type) {
  return parseMma(parser, type);
}

void TritonGPUMmaMulticastEncodingAttr::print(AsmPrinter &printer) const {
  printMma(printer, this);
}

Attribute TritonGPUSharedEncodingAttr::parse(AsmParser &parser, Type type) {
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

  auto parseUInt = [&parser](const NamedAttribute &attr, unsigned &value,
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

  return parser.getChecked<TritonGPUSharedEncodingAttr>(
      parser.getContext(), vec, perPhase, maxPhase, order);
}

void TritonGPUSharedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "vec = " << getVec() << ", perPhase = " << getPerPhase()
          << ", maxPhase = " << getMaxPhase() << ", order = [" << getOrder()
          << "]"
          << "}>";
}

class TritonGPUOpAsmInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto mmaAttr = attr.dyn_cast<TritonGPUMmaEncodingAttr>()) {
      os << "mma";
      TritonGPUOpAsmInterface::printMma(mmaAttr, os);
      return AliasResult::FinalAlias;
    } else if (auto mmaMulticastAttr =
                   attr.dyn_cast<TritonGPUMmaMulticastEncodingAttr>()) {
      os << "mma_multicast";
      TritonGPUOpAsmInterface::printMma(mmaAttr, os);
      return AliasResult::FinalAlias;
    } else if (auto sharedAttr = attr.dyn_cast<TritonGPUSharedEncodingAttr>()) {
      os << "shared";
      TritonGPUOpAsmInterface::printShared(sharedAttr, os);
      return AliasResult::FinalAlias;
    } else if (auto blockedAttr =
                   attr.dyn_cast<TritonGPUBlockedEncodingAttr>()) {
      os << "blocked";
      TritonGPUOpAsmInterface::printBlocked(blockedAttr, os);
      return AliasResult::FinalAlias;
    } else if (auto blockedMulticastAttr =
                   attr.dyn_cast<TritonGPUBlockedMulticastEncodingAttr>()) {
      os << "blocked_multicast";
      TritonGPUOpAsmInterface::printBlocked(blockedMulticastAttr, os);
    }
    OpAsmDialectInterface::getAlias(attr, os);
  }

private:
  static void printMma(const auto &attr, raw_ostream &os) {
    TritonGPUOpAsmInterface::printArray(attr.getFragmentPerWarp(), os);
    TritonGPUOpAsmInterface::printArray(attr.getShapePerWarp(), os);
    TritonGPUOpAsmInterface::printArray(attr.getWarpPerTile(), os);
    TritonGPUOpAsmInterface::printArray(attr.getShapePerTile(), os);
    TritonGPUOpAsmInterface::printArray(attr.getRepetitions(), os);
    TritonGPUOpAsmInterface::printArray(attr.getContigPerThread(), os);
  }

  static void printShared(const auto &attr, raw_ostream &os) {
    os << "_" << attr.getVec();
    os << "_" << attr.getPerPhase();
    os << "_" << attr.getMaxPhase();
    TritonGPUOpAsmInterface::printArray(attr.getOrder(), os);
  }

  static void printBlocked(const auto &attr, raw_ostream &os) {
    TritonGPUOpAsmInterface::printArray(attr.getThreadTileSize(), os);
    TritonGPUOpAsmInterface::printArray(attr.getWarpTileSize(), os);
    TritonGPUOpAsmInterface::printArray(attr.getBlockTileSize(), os);
    TritonGPUOpAsmInterface::printArray(attr.getOrder(), os);
    TritonGPUOpAsmInterface::printArray(attr.getBroadcastAxis(), os);
  }

  static void printArray(const auto &array, raw_ostream &os,
                         const std::string &delimiter = "x") {
    os << "_";
    if (array.empty()) {
      os << "none";
      return;
    }
    for (unsigned i = 0; i < array.size(); i++) {
      os << array[i];
      if (i != array.size() - 1) {
        os << delimiter;
      }
    }
  }
};

void TritonGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"
      >();
  addInterfaces<TritonGPUOpAsmInterface>();
}

namespace mlir {
namespace triton {

// Type inference
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i1Type,
                                 tensorType.getEncoding());
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

} // namespace triton
} // namespace mlir

static LogicalResult verify(CopyAsyncOp op) {
  Type resType = op.getResult().getType();
  if (auto tensorType = resType.dyn_cast<RankedTensorType>()) {
    Attribute encoding = tensorType.getEncoding();
    if (!encoding.isa<TritonGPUSharedEncodingAttr>())
      return op.emitOpError("copy_async should return a shared memory tensor");
  } else
    return op.emitOpError("copy_async should return a tensor");
  return success();
}

#define GET_OP_CLASSES
#include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"

// verify TritonGPU ops
LogicalResult TritonGPUDialect::verifyOperationAttribute(Operation *op,
                                                         NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
