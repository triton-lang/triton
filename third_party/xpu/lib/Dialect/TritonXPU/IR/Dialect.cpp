//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "triton/Dialect/Triton/IR/Dialect.h" // triton::PointType [Ops.cpp.inc]
#include "mlir/IR/DialectImplementation.h"    // DialectAsmParser

// clang-format off
#include "triton/Dialect/TritonXPU/IR/Dialect.h" // before cpp.inc
#include "triton/Dialect/TritonXPU/IR/Dialect.cpp.inc"
// clang-format on

#include "triton/Dialect/Triton/IR/Utility.h" // ceil
#include "llvm/ADT/TypeSwitch.h"              // TypeSwitch

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace xpu {

SmallVector<unsigned> getSizePerCore(Attribute layout) {
  if (auto tritonXPUAttr = mlir::dyn_cast<TritonXPU_AttrTrait>(layout)) {
    return tritonXPUAttr.getSizePerCoreInterface();
  }

  llvm::report_fatal_error("getSizePerCoreInterface not implemented");
  return SmallVector<unsigned>();
}

SmallVector<unsigned> getCoresPerGroup(Attribute layout) {
  if (auto tritonXPUAttr = mlir::dyn_cast<TritonXPU_AttrTrait>(layout)) {
    return tritonXPUAttr.getCoresPerGroupInterface();
  }

  llvm::report_fatal_error("getCoresPerGroupInterface not implemented");
  return SmallVector<unsigned>();
}

SmallVector<unsigned> getGroupsPerCluster(Attribute layout) {
  if (auto tritonXPUAttr = mlir::dyn_cast<TritonXPU_AttrTrait>(layout)) {
    return tritonXPUAttr.getGroupsPerClusterInterface();
  }

  llvm::report_fatal_error("getGroupsPerClusterInterface not implemented");
  return SmallVector<unsigned>();
}

SmallVector<unsigned> getCoresPerCluster(Attribute layout) {
  if (auto tritonXPUAttr = mlir::dyn_cast<TritonXPU_AttrTrait>(layout)) {
    return tritonXPUAttr.getCoresPerClusterInterface();
  }

  llvm::report_fatal_error("getCoresPerClusterInterface not implemented");
  return SmallVector<unsigned>();
}

unsigned getTotalElemsPerThread(Type type) {
  if (type.isIntOrIndexOrFloat() || isa<triton::PointerType>(type))
    return 1;
  auto tensorType = cast<RankedTensorType>(type);
  return getTotalElemsPerThread(tensorType.getEncoding(), tensorType.getShape(),
                                tensorType.getElementType());
}

unsigned getTotalElemsPerThread(Attribute layout, ArrayRef<int64_t> shape,
                                Type eltTy) {
  if (auto tritonXPUAttr = mlir::dyn_cast<TritonXPU_AttrTrait>(layout)) {
    return tritonXPUAttr.getTotalElemsPerThread(shape, eltTy);
  } else if (auto tritonGPUAttr =
                 mlir::dyn_cast<triton::gpu::TritonGPU_AttrTrait>(layout)) {
    return tritonGPUAttr.getTotalElemsPerThread(shape, eltTy);
  } else {
    llvm::report_fatal_error("getTotalElemsPerThread not implemented");
    return 0;
  }
}

unsigned getGroupSize(Attribute layout) {
  unsigned size = 1;
  auto coresPerGroup = getCoresPerGroup(layout);
  for (auto e : coresPerGroup) {
    size *= e;
  }
  return size;
}

// 1 element per thread
// order = reverse(arange(rank))
triton::xpu::ClusterLayoutAttr
getDefaultClusterEncoding(MLIRContext *context, ArrayRef<int64_t> shape,
                          uint32_t buffer_size, uint32_t core_num) {
  int rank = shape.size();
  llvm::SmallVector<unsigned> order(rank);
  std::iota(order.begin(), order.end(), 0);
  // TODO[dyq]: why blockEncoding reverse order in triton 3.0
  triton::xpu::ClusterLayoutAttr encoding = triton::xpu::ClusterLayoutAttr::get(
      context, shape, order, buffer_size, core_num);
  return encoding;
}

SmallVector<unsigned>
getCoresPerClusterWithUniqueData(Attribute layout,
                                 ArrayRef<int64_t> tensorShape) {
  if (auto sliceLayout =
          mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(layout)) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(tensorShape);
    auto parentCoresPerCluster =
        getCoresPerClusterWithUniqueData(parentLayout, parentShape);
    SmallVector<unsigned> coresPerCluster = parentCoresPerCluster;
    coresPerCluster.erase(coresPerCluster.begin() + sliceLayout.getDim());
    return coresPerCluster;
  }
  auto coresPerCluster = getCoresPerCluster(layout);
  assert(coresPerCluster.size() == tensorShape.size() &&
         "layout and tensor shape must have the same rank");
  for (unsigned i = 0; i < coresPerCluster.size(); i++) {
    auto sizePerCore = getSizePerCore(layout)[i];
    auto maxCoresPerDim = ceil<unsigned>(tensorShape[i], sizePerCore);
    coresPerCluster[i] = std::min<unsigned>(coresPerCluster[i], maxCoresPerDim);
  }

  return coresPerCluster;
}

SmallVector<unsigned>
getCoresPerGroupWithUniqueData(Attribute layout,
                               ArrayRef<int64_t> tensorShape) {
  if (auto sliceLayout =
          mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(layout)) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(tensorShape);
    auto parentCoresPerGroup =
        getCoresPerGroupWithUniqueData(parentLayout, parentShape);
    SmallVector<unsigned> coresPerGroup = parentCoresPerGroup;
    coresPerGroup.erase(coresPerGroup.begin() + sliceLayout.getDim());
    return coresPerGroup;
  }
  auto coresPerGroup = getCoresPerGroup(layout);
  assert(coresPerGroup.size() == tensorShape.size() &&
         "layout and tensor shape must have the same rank");
  for (unsigned i = 0; i < coresPerGroup.size(); i++) {
    coresPerGroup[i] = std::min<unsigned>(coresPerGroup[i], tensorShape[i]);
  }

  return coresPerGroup;
}

SmallVector<unsigned> getUniqueContigPerCore(Attribute layout,
                                             ArrayRef<int64_t> shape) {
  // If slice layout, call recursively on parent layout, and drop
  // sliced dim
  if (auto sliceLayout =
          mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(layout)) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(shape);
    auto parentUniqueContigPerCore =
        triton::xpu::getUniqueContigPerCore(parentLayout, parentShape);
    parentUniqueContigPerCore.erase(parentUniqueContigPerCore.begin() +
                                    sliceLayout.getDim());
    return parentUniqueContigPerCore;
  }
  // Base case
  auto rank = shape.size();
  SmallVector<unsigned> ret(rank);
  auto contigPerCore = getSizePerCore(layout);
  assert(contigPerCore.size() == rank && "Unexpected contigPerCore size");
  for (int d = 0; d < rank; ++d) {
    ret[d] = std::min<unsigned>(shape[d], contigPerCore[d]);
  }
  return ret;
}

LogicalResult
ClusterLayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<unsigned> sizePerCore,
                          ArrayRef<unsigned> coresPerGroup,
                          ArrayRef<unsigned> groupsPerCluster,
                          ArrayRef<unsigned> order, bool isReduceOpt) {
  if (sizePerCore.size() != coresPerGroup.size() ||
      coresPerGroup.size() != groupsPerCluster.size() ||
      groupsPerCluster.size() != order.size()) {
    return emitError() << "sizePerCore, coresPerGroup, groupsPerCluster, and "
                          "order must all have the same rank.";
  }

  if (!isPermutationOfIota(order)) {
    return emitError()
           << "order must be a permutation of 0..(rank-1), but was [" << order
           << "]";
  }
  return success();
}

} // namespace xpu
} // namespace triton
} // namespace mlir

//===----------------------------------------------------------------------===//
// Parse Utility
//===----------------------------------------------------------------------===//

static LogicalResult parseIntAttrValue(AsmParser &parser, Attribute attr,
                                       unsigned &value, StringRef desc) {
  auto intAttr = mlir::dyn_cast<IntegerAttr>(attr);
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

static LogicalResult parseBoolAttrValue(AsmParser &parser, Attribute attr,
                                        bool &value, StringRef desc) {
  auto boolAttr = mlir::dyn_cast<BoolAttr>(attr);
  if (!boolAttr) {
    parser.emitError(parser.getNameLoc(), "expected an bool type in ") << desc;
    return failure();
  }
  value = boolAttr.getValue();
  return success();
}

// parse an array of integers
static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       const NamedAttribute &attr,
                                       SmallVector<unsigned> &res,
                                       StringRef desc) {
  auto arrayAttr = mlir::dyn_cast<ArrayAttr>(attr.getValue());
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

static LogicalResult parseBool(AsmParser &parser, const NamedAttribute &attr,
                               bool &value, StringRef desc) {
  return parseBoolAttrValue(parser, attr.getValue(), value, desc);
};

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#include "triton/Dialect/TritonXPU/IR/TritonXPUAttrInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonXPU/IR/TritonXPUAttrDefs.cpp.inc"

SmallVector<unsigned>
triton::xpu::ClusterLayoutAttr::getSizePerCoreInterface() const {
  return SmallVector<unsigned>(getSizePerCore());
}
SmallVector<unsigned>
triton::xpu::ClusterLayoutAttr::getCoresPerGroupInterface() const {
  return SmallVector<unsigned>(getCoresPerGroup());
}
SmallVector<unsigned>
triton::xpu::ClusterLayoutAttr::getGroupsPerClusterInterface() const {
  return SmallVector<unsigned>(getGroupsPerCluster());
}
SmallVector<unsigned>
triton::xpu::ClusterLayoutAttr::getCoresPerClusterInterface() const {
  SmallVector<unsigned> coresPerCluster;
  for (unsigned d = 0, n = getOrder().size(); d < n; ++d)
    coresPerCluster.push_back(getCoresPerGroup()[d] * getGroupsPerCluster()[d]);
  return coresPerCluster;
}

SmallVector<unsigned>
triton::xpu::ClusterLayoutAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                                  Type eltTy) const {
  size_t rank = shape.size();
  auto sizePerCore = getSizePerCore();
  auto coresPerGroup = getCoresPerGroup();
  auto groupsPerCluster = getGroupsPerCluster();
  assert(rank == sizePerCore.size() &&
         "unexpected rank in BlockedEncodingAttr::getElemsPerThread");
  SmallVector<unsigned> elemsPerThread(rank);
  for (size_t i = 0; i < rank; ++i) {
    unsigned t = groupsPerCluster[i] * coresPerGroup[i];
    elemsPerThread[i] = ceil<unsigned>(shape[i], t);
  }
  return elemsPerThread;
}

unsigned
triton::xpu::ClusterLayoutAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                       Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}

Attribute triton::xpu::ClusterLayoutAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  SmallVector<unsigned> sizePerCore;
  SmallVector<unsigned> coresPerGroup;
  SmallVector<unsigned> groupsPerCluster;
  SmallVector<unsigned> order;
  bool isReduceOpt;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "sizePerCore") {
      if (parseIntArrayAttr(parser, attr, sizePerCore,
                            "number of elements per core")
              .failed())
        return {};
    } else if (attr.getName() == "coresPerGroup") {
      if (parseIntArrayAttr(parser, attr, coresPerGroup,
                            "number of cores per group")
              .failed())
        return {};
    } else if (attr.getName() == "groupsPerCluster") {
      if (parseIntArrayAttr(parser, attr, groupsPerCluster,
                            "number of groups per cluster")
              .failed())
        return {};
    } else if (attr.getName() == "order") {
      if (parseIntArrayAttr(parser, attr, order, "order").failed())
        return {};
    } else if (attr.getName() == "isReduceOpt") {
      if (parseBool(parser, attr, isReduceOpt, "isReduceOpt").failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  return parser.getChecked<ClusterLayoutAttr>(parser.getContext(), sizePerCore,
                                              coresPerGroup, groupsPerCluster,
                                              order, isReduceOpt);
}

void triton::xpu::ClusterLayoutAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "sizePerCore = [" << ArrayRef(getSizePerCore()) << "]"
          << ", coresPerGroup = [" << ArrayRef(getCoresPerGroup()) << "]"
          << ", groupsPerCluster = [" << ArrayRef(getGroupsPerCluster()) << "]"
          << ", order = [" << getOrder() << "]"
          << ", isReduceOpt = " << getIsReduceOpt() << "}>";

  printer << "}>";
}

//===----------------------------------------------------------------------===//
// ASM Interface (i.e.: alias)
//===----------------------------------------------------------------------===//

class TritonXPUOpAsmInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto clusterAttr =
            mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(attr)) {
      os << "cluster";
      return AliasResult::FinalAlias;
    }
    return OpAsmDialectInterface::getAlias(attr, os);
  }
};

struct TritonXPUInferLayoutInterface
    : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding) const override {
    resultEncoding = triton::gpu::SliceEncodingAttr::get(
        getDialect()->getContext(), axis, operandEncoding);
    return success();
  }

  // Infer the encoding of a tt.trans(x) given the encoding of x.
  //
  // Our goal is to choose an encoding so that the trans is a "nop".  For
  // example, in a blocked encoding, the same GPU threads hold the same
  // elements, they're just "renamed" -- what was element [i,j] of the tensor is
  // now element [j,i], but that element is held by the same GPU thread.
  //
  // For most properties of the encoding, we let
  //   outputEnc.prop = inputEnc.prop * trans.order,
  // where `x * y` means we apply permutation y to x.
  //
  // This works because prop[i] tells you something about the i'th dimension of
  // the tensor. (For example, sizePerThread[2] == 4 means that one GPU thread
  // contains 4 elements along dim 2 of the tensor.) The transpose reorders the
  // dimensions according to the perm trans.order, so we achieve our goal of
  // having a "nop" transpose by reordering the values in the prop the same way.
  //
  // The big exception to this is the encoding's `order`.
  //
  // An encoding's order is a list of dimensions, from fastest moving (most
  // minor) to slowest moving.  Thus enc.order[i] does not tell you something
  // about the i'th dimension of the tensor, and it would be disasterously
  // incorrect to do enc.order * trans.order.
  //
  // But!  If we invert enc.order, it *does* meet this criterion.  For example,
  // if enc.order = [2,0,1], inverse(enc.order) = [1,2,0].  If you stare at it,
  // you'll see that inverse(enc.order)[i] == j means that dimension i is the
  // j'th most minor.  Therefore we can safely permute *this* by trans.order.
  //
  // Thus we have
  //
  //   outputEnc.order = inverse(inverse(inputEnc.order) * trans.order)
  //                   = inverse(trans.order) * inputEnc.order.
  //
  LogicalResult inferTransOpEncoding(Attribute operandEncoding,
                                     ArrayRef<int32_t> order, // trans order
                                     Attribute &resultEncoding) const override {
    llvm_unreachable(
        "TODO[dyq]: Add triton::xpu::GlobalEncodingAttr Calculation Logic");
    return failure(); // unhandled encoding
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> location) const override {
    auto sliceEncoding =
        mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(operandEncoding);
    if (!sliceEncoding)
      return emitOptionalError(
          location, "ExpandDimsOp operand encoding must be SliceEncodingAttr");
    if (sliceEncoding.getDim() != axis)
      return emitOptionalError(
          location, "Incompatible slice dimension for ExpandDimsOp operand");
    resultEncoding = sliceEncoding.getParent();
    return success();
  }

  LogicalResult
  inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                     Attribute retEncoding,
                     std::optional<Location> location) const override {
    llvm_unreachable("TODO[dyq]: XPUSDNN-CHECK Add "
                     "triton::xpu::GlobalEncodingAttr Calculation Logic");
    return failure();
  }

  LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const override {
    llvm_unreachable("TODO[dyq]: XPUSDNN-CHECK Add "
                     "triton::xpu::GlobalEncodingAttr Calculation Logic");
    return failure();
  }

  // Given a src shape + encoding and a dst shape, our goal is to compute a dst
  // encoding that makes the reshape a "nop".  That is, if GPU thread [x,y,z]
  // contains elements [a,b,c,d] before the reshape, it contains those same
  // elements after the reshape, they're just "renamed".
  //
  // A dst encoding that satisfies this property does not exist for all inputs.
  // Here are some positive and negative examples.
  //
  //   - NOT OK: 4x4 order=[0,1] -> 16.  Reshape merges elements so
  //     dim 1 is the fastest-changing in the dst, but the src has the opposite
  //     order.
  //   - OK: 2x2x32 order=[1,0,2] -> 4x32.  We choose dst order [0,1].
  //     What's important is that the 2x2 dimensions appear in major-to-minor
  //     order.
  //   - NOT OK: 32x32 sizePerThread=[2,2] -> 1024.  Thread 0 in the src
  //     contains elements [(0,0), (0,1), (1,0), and (1,1)].  We cannot express
  //     this with an encoding based on the dst shape.
  //   - OK: 32x4 sizePerThread=[4,4] -> 128.  dst with sizePerThread=[16] will
  //     contain the same elements as before.
  //
  // Users of this function require that it is symmetrical: if
  // (srcShape,srcEnc,dstShape) => dstEnc, then (dstShape,dstEnc,srcShape) =>
  // srcEnc.
  LogicalResult
  inferReshapeOpNoReorderEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                                  ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                                  std::optional<Location> loc) const override {
    llvm_unreachable(
        "TODO[dyq]: Add triton::xpu::GlobalEncodingAttr Calculation Logic");
    return failure();
  }

  LogicalResult
  inferJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                      std::optional<Location> loc) const override {
    llvm_unreachable(
        "TODO[dyq]: Add triton::xpu::GlobalEncodingAttr Calculation Logic");
    return failure();
  }

  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       std::optional<Location> loc) const override {
    llvm_unreachable(
        "TODO[dyq]: Add triton::xpu::GlobalEncodingAttr Calculation Logic");
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Dialect Initialization
//===----------------------------------------------------------------------===//

void mlir::triton::xpu::TritonXPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonXPU/IR/TritonXPUAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST // declare
#include "triton/Dialect/TritonXPU/IR/Ops.cpp.inc"
      >();

  addInterfaces<TritonXPUOpAsmInterface>();
  addInterfaces<TritonXPUInferLayoutInterface>();
}

#define GET_OP_CLASSES // define
#include "triton/Dialect/TritonXPU/IR/Ops.cpp.inc"

// verify TritonXPU ops
LogicalResult
triton::xpu::TritonXPUDialect::verifyOperationAttribute(Operation *op,
                                                        NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
