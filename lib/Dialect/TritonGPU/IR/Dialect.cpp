#include <numeric>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton::gpu;

// Utility
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

static LogicalResult parseIntAttrValue(AsmParser &parser, const Attribute &attr,
                                       unsigned &value, StringRef desc) {
  auto intAttr = attr.dyn_cast<IntegerAttr>();
  if (!intAttr) {
    parser.emitError(parser.getNameLoc(), "expected an integer type in ")
        << desc;
    return failure();
  }
  if (intAttr.getType().isSignedInteger()) {
    int64_t attrVal = intAttr.getSInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else if (intAttr.getType().isSignlessInteger()) {
    int64_t attrVal = intAttr.getInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else {
    value = intAttr.getUInt();
  }
  return success();
}

// parse an array of integers
static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       const NamedAttribute &attr,
                                       SmallVector<unsigned, 2> &res,
                                       StringRef desc) {
  auto arrayAttr = attr.getValue().dyn_cast<ArrayAttr>();
  if (!arrayAttr) {
    parser.emitError(parser.getNameLoc(), "expected an array for ") << desc;
    return failure();
  }
  for (Attribute i : arrayAttr) {
    unsigned value;
    if (parseIntAttrValue(parser, i, value, desc).failed())
      return failure();
    res.push_back(value);
  }
  return success();
};

static LogicalResult parseUInt(AsmParser &parser, const NamedAttribute &attr,
                               unsigned &value, StringRef desc) {
  return parseIntAttrValue(parser, attr.getValue(), value, desc);
};

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.cpp.inc"

SliceEncodingAttr BlockedEncodingAttr::squeeze(int axis) {
  return SliceEncodingAttr::get(getContext(), axis, *this);
}

//===----------------------------------------------------------------------===//
// Blocked Encoding
//===----------------------------------------------------------------------===//

Attribute BlockedEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  SmallVector<unsigned, 2> sizePerThread;
  SmallVector<unsigned, 2> threadsPerWarp;
  SmallVector<unsigned, 2> warpsPerCTA;
  SmallVector<unsigned, 2> order;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "sizePerThread") {
      if (parseIntArrayAttr(parser, attr, sizePerThread,
                            "number of elements per thread")
              .failed())
        return {};
    } else if (attr.getName() == "threadsPerWarp") {
      if (parseIntArrayAttr(parser, attr, threadsPerWarp,
                            "number of threads per warp")
              .failed())
        return {};
    } else if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA,
                            "number of warps per CTA")
              .failed())
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

  return parser.getChecked<BlockedEncodingAttr>(
      parser.getContext(), sizePerThread, threadsPerWarp, warpsPerCTA, order);
}

void BlockedEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "sizePerThread = [" << getSizePerThread() << "]"
          << ", threadsPerWarp = [" << getThreadsPerWarp() << "]"
          << ", warpsPerCTA = [" << getWarpsPerCTA() << "]"
          << ", order = [" << getOrder() << "]"
          << "}>";
}

//===----------------------------------------------------------------------===//
// MMA encoding
//===----------------------------------------------------------------------===//

Attribute MmaEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned version = 0;
  SmallVector<unsigned, 2> warpsPerCTA;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "version") {
      if (parseUInt(parser, attr, version, "version").failed())
        return {};
    }
    if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA, "warpsPerCTA").failed())
        return {};
    }
  }

  return parser.getChecked<MmaEncodingAttr>(parser.getContext(), version,
                                            warpsPerCTA);
}

void MmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "version = " << getVersion() << ", "
          << "warpsPerCTA = [" << getWarpsPerCTA() << "]"
          << "}>";
}

//===----------------------------------------------------------------------===//
// Sliced Encoding
//===----------------------------------------------------------------------===//

Attribute SliceEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned dim = 0;
  Attribute parent;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "dim") {
      if (parseUInt(parser, attr, dim, "dim").failed())
        return {};
    }
    if (attr.getName() == "parent") {
      if (parser.parseAttribute(parent).failed())
        return {};
    }
  }

  return parser.getChecked<SliceEncodingAttr>(parser.getContext(), dim, parent);
}

void SliceEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "dim = " << getDim() << ", "
          << "parent = " << getParent() << "}>";
}

//===----------------------------------------------------------------------===//
// Shared encoding
//===----------------------------------------------------------------------===//

Attribute SharedEncodingAttr::parse(AsmParser &parser, Type type) {
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

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "vec") {
      if (parseUInt(parser, attr, vec, "vec").failed())
        return {};
    } else if (attr.getName() == "perPhase") {
      if (parseUInt(parser, attr, perPhase, "perPhase").failed())
        return {};
    } else if (attr.getName() == "maxPhase") {
      if (parseUInt(parser, attr, maxPhase, "maxPhase").failed())
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

  return parser.getChecked<SharedEncodingAttr>(parser.getContext(), vec,
                                               perPhase, maxPhase, order);
}

void SharedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "vec = " << getVec() << ", perPhase = " << getPerPhase()
          << ", maxPhase = " << getMaxPhase() << ", order = [" << getOrder()
          << "]"
          << "}>";
}

//===----------------------------------------------------------------------===//
// CopyAsyncOp
//===----------------------------------------------------------------------===//

ParseResult parseCopyAsyncOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> allOperands;
  Type resultTypes[1], ptrType;
  SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseCustomTypeWithFallback(ptrType) || parser.parseArrow() ||
      parser.parseCustomTypeWithFallback(resultTypes[0]))
    return failure();
  result.addTypes(resultTypes);

  SmallVector<Type> operandTypes;
  operandTypes.push_back(ptrType); // ptr
  if (allOperands.size() >= 2)
    operandTypes.push_back(triton::getI1SameShape(ptrType)); // mask
  if (allOperands.size() >= 3)
    operandTypes.push_back(triton::getPointeeType(ptrType)); // other

  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void printCopyAsyncOp(OpAsmPrinter &printer, CopyAsyncOp copyAsyncOp) {
  printer << " ";
  printer << copyAsyncOp.getOperation()->getOperands();
  printer.printOptionalAttrDict(copyAsyncOp->getAttrs(), /*elidedAttrs=*/{});
  printer << " : ";
  printer.printStrippedAttrOrType(copyAsyncOp.ptr().getType());
  printer << " -> ";
  printer.printStrippedAttrOrType(copyAsyncOp.result().getType());
}

//===----------------------------------------------------------------------===//
// InsertSliceAsyncOp
//===----------------------------------------------------------------------===//

ParseResult parseInsertSliceAsyncOp(OpAsmParser &parser,
                                    OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> allOperands;
  Type srcType, dstType;
  SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseCustomTypeWithFallback(srcType) || parser.parseArrow() ||
      parser.parseCustomTypeWithFallback(dstType))
    return failure();
  result.addTypes(dstType);

  SmallVector<Type> operandTypes;
  operandTypes.push_back(srcType); // src
  operandTypes.push_back(dstType); // dst
  operandTypes.push_back(
      IntegerType::get(parser.getBuilder().getContext(), 32)); // offset
  if (allOperands.size() >= 4)
    operandTypes.push_back(triton::getI1SameShape(srcType)); // mask
  if (allOperands.size() >= 5)
    operandTypes.push_back(triton::getPointeeType(srcType)); // other

  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void printInsertSliceAsyncOp(OpAsmPrinter &printer,
                             InsertSliceAsyncOp insertSliceAsyncOp) {
  printer << " ";
  printer << insertSliceAsyncOp.getOperation()->getOperands();
  printer.printOptionalAttrDict(insertSliceAsyncOp->getAttrs(),
                                /*elidedAttrs=*/{});
  printer << " : ";
  printer.printStrippedAttrOrType(insertSliceAsyncOp.src().getType());
  printer << " -> ";
  printer.printStrippedAttrOrType(insertSliceAsyncOp.result().getType());
}

//===----------------------------------------------------------------------===//
// ExtractSliceOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult ExtractSliceOp::inferReturnTypes(
    ::mlir::MLIRContext *context, llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    ::mlir::RegionRange regions,
    llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto srcType = operands[0].getType().cast<RankedTensorType>();
  auto encoding = srcType.getEncoding();
  auto srcShape = srcType.getShape();
  auto axis = attributes.get("axis").cast<IntegerAttr>().getInt();
  if (axis < 0 || axis > srcShape.size())
    return failure();
  auto dstShape = srcShape.drop_front(axis + 1);
  auto returnType =
      RankedTensorType::get(dstShape, srcType.getElementType(), encoding);
  inferredReturnTypes.assign({returnType});
  return success();
}

//===----------------------------------------------------------------------===//
// ASM Interface (i.e.: alias)
//===----------------------------------------------------------------------===//

class TritonGPUOpAsmInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto mmaAttr = attr.dyn_cast<MmaEncodingAttr>()) {
      os << "mma";
      return AliasResult::FinalAlias;
    } else if (auto sharedAttr = attr.dyn_cast<SharedEncodingAttr>()) {
      os << "shared";
      return AliasResult::FinalAlias;
    } else if (auto blockedAttr = attr.dyn_cast<BlockedEncodingAttr>()) {
      os << "blocked";
      return AliasResult::FinalAlias;
    } /* else if (auto sliceAttr = attr.dyn_cast<SliceEncodingAttr>()) {
      os << "slice";
      return AliasResult::FinalAlias;
    } */
    return OpAsmDialectInterface::getAlias(attr, os);
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

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

static LogicalResult verify(CopyAsyncOp op) {
  if (!isSharedEncoding(op.getResult())) {
    return op.emitOpError("copy_async should return a shared memory tensor");
  }
  return success();
}

static LogicalResult verify(InsertSliceAsyncOp op) {
  if (!isSharedEncoding(op.getResult())) {
    return op.emitOpError("copy_async should return a shared memory tensor");
  }
  return success();
}

static LogicalResult verify(ExtractSliceOp op) {
  if (!isSharedEncoding(op.getResult())) {
    return op.emitOpError("extract_slice should return a shared memory tensor");
  }
  return success();
}

static LogicalResult verify(AllocTensorOp op) {
  if (!isSharedEncoding(op.getResult())) {
    return op.emitOpError("alloc_tensor should return a shared memory tensor");
  }
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
