#include <numeric>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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

namespace gpu {

// TODO: Inheritance of layout attributes
// so that all distributed layouts implement
// these utilities

unsigned getElemsPerThread(Attribute layout, ArrayRef<int64_t> shape) {
  if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
    return blockedLayout.getElemsPerThread(shape);
  } else if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
    return sliceLayout.getElemsPerThread(shape);
  } else if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
    return mmaLayout.getElemsPerThread(shape);
  } else if (auto sharedLayout = layout.dyn_cast<SharedEncodingAttr>()) {
    return sharedLayout.getElemsPerThread(shape);
  } else if (auto dotLayout = layout.dyn_cast<DotOperandEncodingAttr>()) {
    return dotLayout.getElemsPerThread(shape);
  } else {
    assert(0 && "getElemsPerThread not implemented");
    return 0;
  }
}

unsigned getElemsPerThread(Type type) {
  if (type.isIntOrIndexOrFloat() || type.isa<triton::Float8Type>() ||
      type.isa<triton::PointerType>())
    return 1;
  auto tensorType = type.cast<RankedTensorType>();
  return getElemsPerThread(tensorType.getEncoding(), tensorType.getShape());
}

SmallVector<unsigned> getThreadsPerWarp(const Attribute &layout) {
  if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
    return SmallVector<unsigned>(blockedLayout.getThreadsPerWarp().begin(),
                                 blockedLayout.getThreadsPerWarp().end());
  }
  if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
    if (mmaLayout.isVolta())
      return {4, 8};
    if (mmaLayout.isAmpere())
      return {8, 4};
  }
  assert(0 && "getThreadsPerWarp not implemented");
  return {};
}

SmallVector<unsigned> getWarpsPerCTA(const Attribute &layout) {
  if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
    return SmallVector<unsigned>(blockedLayout.getWarpsPerCTA().begin(),
                                 blockedLayout.getWarpsPerCTA().end());
  }
  if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
    return SmallVector<unsigned>(mmaLayout.getWarpsPerCTA().begin(),
                                 mmaLayout.getWarpsPerCTA().end());
  }
  assert(0 && "getWarpsPerCTA not implemented");
  return {};
}

SmallVector<unsigned> getSizePerThread(const Attribute &layout) {
  if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
    return SmallVector<unsigned>(blockedLayout.getSizePerThread().begin(),
                                 blockedLayout.getSizePerThread().end());
  } else if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
    return getSizePerThread(sliceLayout.getParent());
  } else if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
    if (mmaLayout.isAmpere()) {
      return {2, 2};
    } else if (mmaLayout.isVolta()) {
      // Note: here the definition of sizePerThread is obscure, which doesn't
      // mean vecSize=4 can be supported in the last dimension.
      return {2, 4};
    } else {
      llvm_unreachable("Unexpected mma version");
    }
  } else if (auto dotLayout = layout.dyn_cast<DotOperandEncodingAttr>()) {
    auto parentLayout = dotLayout.getParent();
    assert(parentLayout && "DotOperandEncodingAttr must have a parent");
    if (auto parentMmaLayout = parentLayout.dyn_cast<MmaEncodingAttr>()) {
      assert(parentMmaLayout.isAmpere() &&
             "mmaLayout version = 1 is not implemented yet");
      auto parentShapePerCTA = getShapePerCTA(parentLayout);
      auto opIdx = dotLayout.getOpIdx();
      if (opIdx == 0) {
        return {2, 4};
      } else if (opIdx == 1) {
        return {4, 1};
      } else {
        assert(0 && "DotOperandEncodingAttr opIdx must be 0 or 1");
        return {};
      }
    } else {
      assert(0 && "DotOperandEncodingAttr non-MmaEncodingAttr parent not "
                  "supported yet");
      return {};
    }
  } else {
    assert(0 && "getSizePerThread not implemented");
    return {};
  }
}

SmallVector<unsigned> getContigPerThread(Attribute layout) {
  if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
    assert(mmaLayout.isVolta() || mmaLayout.isAmpere());
    return {1, 2};
  } else {
    return getSizePerThread(layout);
  }
}

SmallVector<unsigned> getThreadsPerCTA(const Attribute &layout) {
  SmallVector<unsigned> threads;
  if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
    for (int d = 0, n = blockedLayout.getOrder().size(); d < n; ++d)
      threads.push_back(blockedLayout.getThreadsPerWarp()[d] *
                        blockedLayout.getWarpsPerCTA()[d]);
  } else if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
    assert(0 && "Unimplemented usage of MmaEncodingAttr");
  } else {
    assert(0 && "Unimplemented usage of getShapePerCTA");
  }

  return threads;
}

SmallVector<unsigned> getShapePerCTA(const Attribute &layout) {
  SmallVector<unsigned> shape;
  if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
    for (unsigned d = 0, n = blockedLayout.getOrder().size(); d < n; ++d)
      shape.push_back(blockedLayout.getSizePerThread()[d] *
                      blockedLayout.getThreadsPerWarp()[d] *
                      blockedLayout.getWarpsPerCTA()[d]);
  } else if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
    unsigned dim = sliceLayout.getDim();
    auto parent = sliceLayout.getParent();
    for (unsigned d = 0, n = getOrder(parent).size(); d < n; ++d) {
      if (d == dim)
        continue;
      shape.push_back(getShapePerCTA(parent)[d]);
    }
  } else if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
    if (mmaLayout.isAmpere())
      return {16 * mmaLayout.getWarpsPerCTA()[0],
              8 * mmaLayout.getWarpsPerCTA()[1]};
    if (mmaLayout.isVolta())
      return {16 * mmaLayout.getWarpsPerCTA()[0],
              16 * mmaLayout.getWarpsPerCTA()[1]};
    assert(0 && "Unexpected MMA layout version found");
  } else if (auto dotLayout = layout.dyn_cast<DotOperandEncodingAttr>()) {
    auto parentLayout = dotLayout.getParent();
    assert(parentLayout && "DotOperandEncodingAttr must have a parent");
    if (auto parentMmaLayout = parentLayout.dyn_cast<MmaEncodingAttr>()) {
      assert(parentMmaLayout.isAmpere() &&
             "mmaLayout version = 1 is not implemented yet");
      auto parentShapePerCTA = getShapePerCTA(parentLayout);
      auto opIdx = dotLayout.getOpIdx();
      if (opIdx == 0) {
        return {parentShapePerCTA[0], 16};
      } else if (opIdx == 1) {
        return {16, parentShapePerCTA[1]};
      } else {
        assert(0 && "DotOperandEncodingAttr opIdx must be 0 or 1");
      }
    } else {
      assert(0 && "DotOperandEncodingAttr non-MmaEncodingAttr parent not "
                  "supported yet");
    }
  } else if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
    if (mmaLayout.isAmpere()) {
      return {16 * mmaLayout.getWarpsPerCTA()[0],
              8 * mmaLayout.getWarpsPerCTA()[1]};
    } else if (mmaLayout.isVolta()) {
      return {16 * mmaLayout.getWarpsPerCTA()[0],
              16 * mmaLayout.getWarpsPerCTA()[1]};
    } else {
      llvm_unreachable("Unexpected mma version");
    }
  } else {
    assert(0 && "Unimplemented usage of getShapePerCTA");
  }
  return shape;
}

SmallVector<unsigned> getOrder(const Attribute &layout) {
  if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
    return SmallVector<unsigned>(blockedLayout.getOrder().begin(),
                                 blockedLayout.getOrder().end());
  } else if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
    return {1, 0};
  } else if (auto dotLayout = layout.dyn_cast<DotOperandEncodingAttr>()) {
    return {1, 0};
  } else if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
    SmallVector<unsigned> parentOrder = getOrder(sliceLayout.getParent());
    unsigned dim = sliceLayout.getDim();
    SmallVector<unsigned> order;
    for (unsigned d : parentOrder) {
      if (d == dim)
        continue;
      else if (d > dim)
        order.push_back(d - 1);
      else
        order.push_back(d);
    }
    return order;
  } else if (auto sharedLayout = layout.dyn_cast<SharedEncodingAttr>()) {
    return SmallVector<unsigned>(sharedLayout.getOrder().begin(),
                                 sharedLayout.getOrder().end());
  } else {
    assert(0 && "Unimplemented usage of getOrder");
    return {};
  }
};

} // namespace gpu
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

unsigned BlockedEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape) const {
  size_t rank = shape.size();
  auto sizePerThread = getSizePerThread();
  auto warpsPerCTA = getWarpsPerCTA();
  auto threadsPerWarp = getThreadsPerWarp();
  assert(rank == sizePerThread.size() &&
         "unexpected rank in BlockedEncodingAttr::getElemsPerThread");
  SmallVector<unsigned> elemsPerThread(rank);
  for (size_t i = 0; i < rank; ++i) {
    unsigned t = sizePerThread[i] * threadsPerWarp[i] * warpsPerCTA[i];
    elemsPerThread[i] = ceil<unsigned>(shape[i], t) * sizePerThread[i];
  }
  return product<unsigned>(elemsPerThread);
}

template <class T>
SmallVector<T> SliceEncodingAttr::paddedShape(ArrayRef<T> shape) const {
  size_t rank = shape.size();
  unsigned dim = getDim();
  SmallVector<T> retShape(rank + 1);
  for (unsigned d = 0; d < rank + 1; ++d) {
    if (d < dim)
      retShape[d] = shape[d];
    else if (d == dim)
      retShape[d] = 1;
    else
      retShape[d] = shape[d - 1];
  }
  return retShape;
}
template SmallVector<unsigned>
SliceEncodingAttr::paddedShape<unsigned>(ArrayRef<unsigned> shape) const;
template SmallVector<int64_t>
SliceEncodingAttr::paddedShape<int64_t>(ArrayRef<int64_t> shape) const;

unsigned SliceEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape) const {
  size_t rank = shape.size();
  auto parent = getParent();
  return ::getElemsPerThread(parent, paddedShape(shape));
}

unsigned MmaEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape) const {
  size_t rank = shape.size();
  assert(rank == 2 && "Unexpected rank of mma layout");
  assert((isVolta() || isAmpere()) && "Only version 1 and 2 is supported");

  int res = 0;
  if (isVolta()) {
    unsigned mmasRow = ceil<unsigned>(shape[0], 16 * getWarpsPerCTA()[0]);
    unsigned mmasCol = ceil<unsigned>(shape[1], 16 * getWarpsPerCTA()[1]);
    // Each warp-level mma884 will perform a m16xn16xk4 mma, thus get a m16xn16
    // matrix as result.
    res = mmasRow * mmasCol * (16 * 16 / 32);
  } else if (isAmpere()) {
    unsigned elemsCol = ceil<unsigned>(shape[0], 16 * getWarpsPerCTA()[0]) * 2;
    unsigned elemsRow = ceil<unsigned>(shape[1], 8 * getWarpsPerCTA()[1]) * 2;
    res = elemsCol * elemsRow;
  } else {
    llvm_unreachable("Unexpected mma version");
  }

  return res;
}

unsigned SharedEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape) const {
  // TODO:
  assert(0 && "SharedEncodingAttr::getElemsPerThread not implemented");
  return 0;
}

unsigned
DotOperandEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape) const {
  if (auto blockedLayout = getParent().dyn_cast<BlockedEncodingAttr>()) {
    return blockedLayout.getElemsPerThread(shape);
  }
  assert(0 && "DotOperandEncodingAttr::getElemsPerThread not implemented");
  return 0;
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

  auto ret = parser.getChecked<BlockedEncodingAttr>(
      parser.getContext(), sizePerThread, threadsPerWarp, warpsPerCTA, order);
  return ret;
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

  unsigned versionMajor = 0;
  unsigned versionMinor = 0;
  SmallVector<unsigned, 2> warpsPerCTA;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "versionMajor") {
      if (parseUInt(parser, attr, versionMajor, "versionMajor").failed())
        return {};
    }
    if (attr.getName() == "versionMinor") {
      if (parseUInt(parser, attr, versionMinor, "versionMinor").failed())
        return {};
    }
    if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA, "warpsPerCTA").failed())
        return {};
    }
  }

  return parser.getChecked<MmaEncodingAttr>(parser.getContext(), versionMajor,
                                            versionMinor, warpsPerCTA);
}

void MmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "versionMajor = " << getVersionMajor() << ", "
          << "versionMinor = " << getVersionMinor() << ", "
          << "warpsPerCTA = [" << getWarpsPerCTA() << "]"
          << "}>";
}

//===----------------------------------------------------------------------===//
// Sliced Encoding
//===----------------------------------------------------------------------===//

Attribute SliceEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};
  unsigned dim = attrs.get("dim").cast<IntegerAttr>().getInt();
  Attribute parent = attrs.get("parent");
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
// Mma encoding
//===----------------------------------------------------------------------===//

bool MmaEncodingAttr::isVolta() const { return getVersionMajor() == 1; }

bool MmaEncodingAttr::isAmpere() const { return getVersionMajor() == 2; }

// Get [isARow, isBRow, isAVec4, isBVec4] from versionMinor
std::tuple<bool, bool, bool, bool>
MmaEncodingAttr::decodeVoltaLayoutStates() const {
  unsigned versionMinor = getVersionMinor();
  bool isARow = versionMinor & (1 << 0);
  bool isBRow = versionMinor & (1 << 1);
  bool isAVec4 = versionMinor & (1 << 2);
  bool isBVec4 = versionMinor & (1 << 3);
  return std::make_tuple(isARow, isBRow, isAVec4, isBVec4);
}

//===----------------------------------------------------------------------===//
// DotOperand Encoding
//===----------------------------------------------------------------------===//
Attribute DotOperandEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};
  unsigned opIdx = attrs.get("opIdx").cast<IntegerAttr>().getInt();
  Attribute parent = attrs.get("parent");
  Attribute isMMAv1Row;
  if (parent.isa<MmaEncodingAttr>() &&
      parent.cast<MmaEncodingAttr>().isVolta()) {
    isMMAv1Row = attrs.get("isMMAv1Row");
    if (!isMMAv1Row)
      llvm::report_fatal_error("isMMAv1Row attribute is missing");
  }
  return parser.getChecked<DotOperandEncodingAttr>(parser.getContext(), opIdx,
                                                   parent, isMMAv1Row);
}

void DotOperandEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "opIdx = " << getOpIdx() << ", "
          << "parent = " << getParent();
  if (getIsMMAv1Row())
    printer << ", isMMAv1Row = " << getIsMMAv1Row();
  printer << "}>";
}

//===----------------------------------------------------------------------===//
// InsertSliceAsyncOp
//===----------------------------------------------------------------------===//

ParseResult parseInsertSliceAsyncOp(OpAsmParser &parser,
                                    OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 8> allOperands;
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
      IntegerType::get(parser.getBuilder().getContext(), 32)); // index

  int hasMask = 0, hasOther = 0;
  if (allOperands.size() >= 4) {
    operandTypes.push_back(triton::getI1SameShape(srcType)); // mask
    hasMask = 1;
  }
  if (allOperands.size() >= 5) {
    operandTypes.push_back(triton::getPointeeType(srcType)); // other
    hasOther = 1;
  }

  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();

  // Deduce operand_segment_sizes from the number of the operands.
  auto operand_segment_sizesAttrName =
      InsertSliceAsyncOp::operand_segment_sizesAttrName(result.name);
  result.addAttribute(
      operand_segment_sizesAttrName,
      parser.getBuilder().getI32VectorAttr({1, 1, 1, hasMask, hasOther}));
  return success();
}

void printInsertSliceAsyncOp(OpAsmPrinter &printer,
                             InsertSliceAsyncOp insertSliceAsyncOp) {
  printer << " ";
  printer << insertSliceAsyncOp.getOperation()->getOperands();
  // "operand_segment_sizes" can be deduced, so we don't print it.
  printer.printOptionalAttrDict(
      insertSliceAsyncOp->getAttrs(),
      {insertSliceAsyncOp.operand_segment_sizesAttrName()});
  printer << " : ";
  printer.printStrippedAttrOrType(insertSliceAsyncOp.src().getType());
  printer << " -> ";
  printer.printStrippedAttrOrType(insertSliceAsyncOp.result().getType());
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

struct TritonGPUInferLayoutInterface
    : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding) const override {
    resultEncoding = SliceEncodingAttr::get(getDialect()->getContext(), axis,
                                            operandEncoding);
    return success();
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            Optional<Location> location) const override {
    auto sliceEncoding = operandEncoding.dyn_cast<SliceEncodingAttr>();
    if (!sliceEncoding)
      return emitOptionalError(
          location, "ExpandDimsOp operand encoding must be SliceEncodingAttr");
    if (sliceEncoding.getDim() != axis)
      return emitOptionalError(
          location, "Incompatible slice dimension for ExpandDimsOp operand");
    resultEncoding = sliceEncoding.getParent();
    return success();
  }

  LogicalResult inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                                   Attribute retEncoding,
                                   Optional<Location> location) const override {
    if (auto dotOpEnc = operandEncoding.dyn_cast<DotOperandEncodingAttr>()) {
      if (opIdx != dotOpEnc.getOpIdx())
        return emitOptionalError(location, "Wrong opIdx");
      if (retEncoding != dotOpEnc.getParent())
        return emitOptionalError(location, "Incompatible parent encoding");
    } else
      return emitOptionalError(
          location, "Dot's a/b's encoding should be of DotOperandEncodingAttr");
    return success();
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
  addInterfaces<TritonGPUInferLayoutInterface>();
}

#define GET_OP_CLASSES
#include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"

// verify TritonGPU ops
LogicalResult TritonGPUDialect::verifyOperationAttribute(Operation *op,
                                                         NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
