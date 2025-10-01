#include "triton/Dialect/Triton/IR/Dialect.h"

#include <cstdint>
#include <numeric>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LayoutUtility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

// Include TableGen'erated code
#include "triton/Dialect/TritonGPU/IR/Dialect.cpp.inc"
#include "triton/Dialect/TritonGPU/IR/OpInterfaces.cpp.inc"
#include "triton/Dialect/TritonGPU/IR/TypeInterfaces.cpp.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// Utility
namespace mlir {
namespace triton {
namespace gpu {

LinearEncodingAttr TritonGPUDialect::toLinearEncoding(ArrayRef<int64_t> shape,
                                                      Attribute layout) {
  // LinearEncoding is a DistributedLayout
  std::vector<int64_t> allocationShape;
  CacheKey key{std::vector<int64_t>(shape.begin(), shape.end()), layout};
  if (auto result = leCache.get(key)) {
    return *result;
  }
  auto linearLayout = toLinearLayout(shape, layout);
  auto linearEncoding =
      LinearEncodingAttr::get(layout.getContext(), std::move(linearLayout));
  leCache.set(key, linearEncoding);
  return linearEncoding;
}

LinearEncodingAttr toLinearEncoding(DistributedEncodingTrait layout,
                                    ArrayRef<int64_t> shape) {
  auto *ctx = layout.getContext();
  return ctx->getLoadedDialect<TritonGPUDialect>()->toLinearEncoding(shape,
                                                                     layout);
}

LinearEncodingAttr toLinearEncoding(RankedTensorType type) {
  auto *ctx = type.getContext();
  return ctx->getLoadedDialect<TritonGPUDialect>()->toLinearEncoding(
      type.getShape(), type.getEncoding());
}

unsigned getTotalElemsPerThread(Attribute layout, ArrayRef<int64_t> shape) {
  return toLinearEncoding(cast<DistributedEncodingTrait>(layout), shape)
      .getTotalElemsPerThread(shape);
}

SmallVector<unsigned> getElemsPerThread(Attribute layout,
                                        ArrayRef<int64_t> shape) {
  return toLinearEncoding(cast<DistributedEncodingTrait>(layout), shape)
      .getElemsPerThread(shape);
}

SmallVector<unsigned> getElemsPerThread(Type type) {
  if (type.isIntOrIndexOrFloat() || isa<triton::PointerType>(type))
    return SmallVector<unsigned>(1, 1);
  auto tensorType = cast<RankedTensorType>(type);
  return getElemsPerThread(tensorType.getEncoding(), tensorType.getShape());
}

unsigned getTotalElemsPerThread(Type type) {
  if (type.isIntOrIndexOrFloat() || isa<triton::PointerType>(type))
    return 1;
  auto tensorType = cast<RankedTensorType>(type);
  return getTotalElemsPerThread(tensorType.getEncoding(),
                                tensorType.getShape());
}

SmallVector<unsigned> getThreadsPerWarp(Attribute layout,
                                        ArrayRef<int64_t> shape) {
  return toLinearEncoding(cast<DistributedEncodingTrait>(layout), shape)
      .getThreadsPerWarp();
}

SmallVector<unsigned> getWarpsPerCTA(Attribute layout,
                                     ArrayRef<int64_t> shape) {
  return toLinearEncoding(cast<DistributedEncodingTrait>(layout), shape)
      .getWarpsPerCTA();
}

SmallVector<unsigned> getContigPerThread(RankedTensorType type) {
  return toLinearEncoding(type).getContigPerThread();
}

bool isExpensiveView(Type srcType, Type dstType) {
  auto tensorSrcType = cast<RankedTensorType>(srcType);
  auto tensorDstType = cast<RankedTensorType>(dstType);
  auto llSrc = toLinearLayout(tensorSrcType);
  auto llDst = toLinearLayout(tensorDstType);
  // In case there are replicated value we need to make sure the new and old
  // layout have matching masks.
  for (auto [srcMask, dstMask] :
       llvm::zip(llSrc.getFreeVariableMasks(), llDst.getFreeVariableMasks())) {
    assert(srcMask.first == dstMask.first);
    if (srcMask.second != dstMask.second)
      return true;
  }
  return getTotalElemsPerThread(srcType) != getTotalElemsPerThread(dstType);
}

/* Utility function used by get.*Order methods of SliceEncodingAttr.
 * Erase dim and decrease all values larger than dim by 1.
 * Example:    order = [0, 2, 4, 3, 1], dim = 2
 *          resOrder = [0,    3, 2, 1]
 */
static SmallVector<unsigned> eraseOrder(ArrayRef<unsigned> order,
                                        unsigned dim) {
  unsigned rank = order.size();
  assert(dim < rank && "Invalid dim to erase");
  SmallVector<unsigned> resOrder;
  for (unsigned i : order)
    if (i < dim)
      resOrder.push_back(i);
    else if (i > dim)
      resOrder.push_back(i - 1);
  return resOrder;
}

SmallVector<unsigned> getMatrixOrder(unsigned rank, bool rowMajor) {
  // Return the order that represents that the batch is in row-major or
  // column-major order for a batch of matrices of shape [*, m, n] with
  // len(shape) == rank.
  SmallVector<unsigned> order(rank);
  if (rank < 2) {
    return order;
  }
  std::iota(order.rbegin(), order.rend(), 0);
  if (!rowMajor) {
    std::swap(order[0], order[1]);
  }
  return order;
}

SmallVector<unsigned> getOrderForDotOperand(unsigned opIdx, unsigned rank,
                                            bool kContig) {
  // kContig: if true, the matrix is fastest-running on k,
  //         otherwise it is on m (resp. n)
  // opIdx=0: [*batch, m, k]
  // opIdx=1: [*batch, k, n]
  assert(opIdx == 0 || opIdx == 1);
  auto rowMajor = bool(opIdx) != kContig;
  return getMatrixOrder(rank, rowMajor);
}

SmallVector<unsigned> getRepOrder(RankedTensorType type) {
  auto layout = type.getEncoding();
  if (auto distributedLayout = mlir::dyn_cast<DistributedEncodingTrait>(layout))
    return distributedLayout.getRepOrder();
  else
    llvm::report_fatal_error("Unimplemented usage of getRepOrder");
  return {};
}

// Legacy impl for now
// This one's not terribly bad as we don't broadcast ShareEncodings
SmallVector<unsigned> getOrder(SharedEncodingTrait layout,
                               ArrayRef<int64_t> shape) {
  if (auto swizzledLayout = dyn_cast<SwizzledSharedEncodingAttr>(layout)) {
    return llvm::to_vector(swizzledLayout.getOrder());
  }
  if (auto paddedEnc = dyn_cast<PaddedSharedEncodingAttr>(layout)) {
    return paddedEnc.getOrder();
  }
  if (auto linearEnc = dyn_cast<SharedLinearEncodingAttr>(layout)) {
    return linearEnc.getOrder();
  }
  if (auto sharedLayout = dyn_cast<NVMMASharedEncodingAttr>(layout)) {
    if (shape.size() == 1) {
      return {0};
    }
    return getMatrixOrder(shape.size(), !sharedLayout.getTransposed());
  }
  if (auto sharedLayout = dyn_cast<AMDRotatingSharedEncodingAttr>(layout)) {
    return llvm::to_vector(sharedLayout.getOrder());
  }
  llvm::report_fatal_error("Unimplemented usage of getOrder for MemDescType");
  return {};
}

SmallVector<unsigned> getOrder(DistributedEncodingTrait layout,
                               ArrayRef<int64_t> shape) {
  return toLinearEncoding(layout, shape).getOrder();
}

SmallVector<unsigned> getOrderForMemory(DistributedEncodingTrait layout,
                                        ArrayRef<int64_t> shape) {
  auto linear = toLinearEncoding(layout, shape);
  auto order = linear.getOrder();
  auto threadOrder = linear.getThreadOrder();
  if (order == threadOrder) {
    return order;
  }
  // Heuristic:
  // If the element contiguity does not align with the thread order
  // because the thread order dimension has contiguity of 1---meaning that
  // the order position of this dimension is irrelevant---we prefer
  // to use the thread order for the memory layout
  auto contig = linear.getElemsPerThread(shape);
  if (contig[threadOrder[0]] == 1) {
    return threadOrder;
  }
  return order;
}

SmallVector<unsigned> getThreadOrder(DistributedEncodingTrait layout,
                                     ArrayRef<int64_t> shape) {
  return toLinearEncoding(layout, shape).getThreadOrder();
}

SmallVector<unsigned> getWarpOrder(DistributedEncodingTrait layout,
                                   ArrayRef<int64_t> shape) {
  return toLinearEncoding(layout, shape).getWarpOrder();
}

CTALayoutAttr getCTALayout(Attribute layout) {
  if (auto ttgLayout = mlir::dyn_cast<LayoutEncodingTrait>(layout)) {
    return CTALayoutAttr::get(layout.getContext(), getCTAsPerCGA(ttgLayout),
                              getCTASplitNum(ttgLayout),
                              getCTAOrder(ttgLayout));
  }
  llvm::report_fatal_error("Unimplemented usage of getCTALayout");
  return {};
}

SmallVector<unsigned> getCTAsPerCGA(Attribute layout) {
  if (auto ttgLayout = mlir::dyn_cast<LayoutEncodingTrait>(layout))
    return ttgLayout.getCTAsPerCGA();
  llvm::report_fatal_error("Unimplemented usage of getCTAsPerCGA");
}

SmallVector<unsigned> getCTASplitNum(Attribute layout) {
  SmallVector<unsigned> res;
  if (auto ttgLayout = mlir::dyn_cast<LayoutEncodingTrait>(layout)) {
    return ttgLayout.getCTASplitNum();
  } else if (auto tmemLayout =
                 mlir::dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
                     layout)) {
    res.resize(2);
    res[0] = tmemLayout.getCTASplitM();
    res[1] = tmemLayout.getCTASplitN();
  } else if (auto tmemScaleLayout = mlir::dyn_cast<
                 triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(layout)) {
    res.resize(2);
    res[0] = tmemScaleLayout.getCTASplitM();
    res[1] = tmemScaleLayout.getCTASplitN();
  } else {
    assert(false && "Unimplemented usage of getCTASplitNum");
  }
  return res;
}

SmallVector<unsigned> getCTAOrder(Attribute layout) {
  SmallVector<unsigned> res;
  if (auto ttgLayout = mlir::dyn_cast<LayoutEncodingTrait>(layout)) {
    res = ttgLayout.getCTAOrder();
  } else {
    llvm::report_fatal_error("Unimplemented usage of getCTAOrder");
  }
  return res;
}

SmallVector<int64_t> getShapePerCTA(ArrayRef<unsigned> CTASplitNum,
                                    ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  auto splitNum = llvm::to_vector(CTASplitNum);
  if (splitNum.size() <= rank) { // pipelining
    splitNum.insert(splitNum.begin(), rank - splitNum.size(), 1);
  } else { // memory slicing
    splitNum =
        llvm::to_vector(llvm::drop_begin(splitNum, splitNum.size() - rank));
  }
  SmallVector<int64_t> shapePerCTA(rank);
  for (unsigned i = 0; i < rank; ++i) {
    shapePerCTA[i] = shape[i] / std::min<unsigned>(shape[i], splitNum[i]);
  }
  return shapePerCTA;
}

SmallVector<int64_t> getShapePerCTA(Attribute layout, ArrayRef<int64_t> shape) {
  return getShapePerCTA(getCTASplitNum(layout), shape);
}

SmallVector<int64_t> getAllocationShapePerCTA(Attribute layout,
                                              ArrayRef<int64_t> shapeLogical) {
  SmallVector<int64_t> shape(shapeLogical);
  if (auto sharedMMALayout = dyn_cast<NVMMASharedEncodingAttr>(layout)) {
    if (sharedMMALayout.getFp4Padded()) {
      auto packedAxis = getOrder(sharedMMALayout, shapeLogical)[0];
      shape[packedAxis] *= 2;
    }
  }
  return getShapePerCTA(layout, shape);
}

SmallVector<int64_t> getShapePerCTA(Type type) {
  auto tensorType = cast<TensorOrMemDesc>(type);
  return getShapePerCTA(tensorType.getEncoding(), tensorType.getShape());
}

SmallVector<int64_t> getAllocationShapePerCTA(Type type) {
  auto tensorType = cast<TensorOrMemDesc>(type);
  return getAllocationShapePerCTA(tensorType.getEncoding(),
                                  tensorType.getShape());
}

unsigned getNumCTAs(Attribute layout) {
  return product<unsigned>(getCTAsPerCGA(layout));
}

SmallVector<unsigned> orderPerDimImpl(const LinearLayout &ll,
                                      StringAttr dimName,
                                      ArrayRef<unsigned> defaultOrder) {
  assert(ll.getBases().contains(dimName));
  const auto &bases = ll.getBases().find(dimName)->second;
  llvm::SetVector<unsigned> order;
  auto nonZero = [](auto val) { return val != 0; };
  for (const auto &basis : bases) {
    // Bases can have one or zero non-zero elements
    // Skip a basis if it's broadcasting (all zeros)
    // e.g. warps for DotOperandEncodingAttr (see ampereDotToLinearLayout)
    auto it = std::find_if(basis.begin(), basis.end(), nonZero);
    if (it != basis.end()) {
      auto i = it - basis.begin();
      order.insert(i);
    }
  }
  // If any dim is missing, we add them in the defaultOrder
  for (auto i : defaultOrder) {
    order.insert(i);
  }
  return order.takeVector();
}

bool isExpensiveCat(CatOp cat, Attribute targetEncoding) {
  // If the new elements per thread is less than the old one, we will need to
  // do convert encoding that goes through shared memory anyway. So we
  // consider it as expensive.
  RankedTensorType tensorTy = cat.getType();
  auto totalElemsPerThread = gpu::getTotalElemsPerThread(tensorTy);
  auto shape = tensorTy.getShape();
  auto newTotalElemsPerThread =
      gpu::getTotalElemsPerThread(targetEncoding, shape);
  return newTotalElemsPerThread < totalElemsPerThread;
}

static LogicalResult
verifyLayoutOrder(function_ref<InFlightDiagnostic()> emitError,
                  ArrayRef<unsigned> order) {
  if (!isPermutationOfIota(order)) {
    return emitError()
           << "order must be a permutation of 0..(rank-1), but was [" << order
           << "]";
  }
  return success();
}

LogicalResult CTALayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<unsigned> CTAsPerCGA,
    ArrayRef<unsigned> CTASplitNum, ArrayRef<unsigned> CTAOrder) {
  if (!llvm::all_equal(
          {CTAsPerCGA.size(), CTASplitNum.size(), CTAOrder.size()})) {
    return emitError() << "CTAsPerCGA, CTASplitNum, and CTAOrder must all have "
                          "the same rank.";
  }

  if (failed(verifyLayoutOrder(emitError, CTAOrder)))
    return failure();

  if (llvm::any_of(CTAsPerCGA, [](unsigned x) { return x == 0; })) {
    return emitError() << "Every element in CTAsPerCGA must be greater than 0.";
  }

  if (llvm::any_of(CTASplitNum, [](unsigned x) { return x == 0; })) {
    return emitError()
           << "Every element in CTASplitNum must be greater than 0.";
  }

  return success();
}

LogicalResult
BlockedEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            ArrayRef<unsigned> sizePerThread,
                            ArrayRef<unsigned> threadsPerWarp,
                            ArrayRef<unsigned> warpsPerCTA,
                            ArrayRef<unsigned> order, CTALayoutAttr CTALayout) {
  if (!llvm::all_equal({sizePerThread.size(), threadsPerWarp.size(),
                        warpsPerCTA.size(), order.size()})) {
    return emitError() << "sizePerThread, threadsPerWarp, warpsPerCTA, and "
                          "order must all have the same rank.";
  }
  if (llvm::any_of(sizePerThread,
                   [](unsigned x) { return !llvm::isPowerOf2_64(x); })) {
    return emitError()
           << "Every element in sizePerThread must be a power of two.";
  }
  if (llvm::any_of(threadsPerWarp,
                   [](unsigned x) { return !llvm::isPowerOf2_64(x); })) {
    return emitError()
           << "Every element in threadsPerWarp must be a power of two.";
  }
  if (llvm::any_of(warpsPerCTA,
                   [](unsigned x) { return !llvm::isPowerOf2_64(x); })) {
    return emitError()
           << "Every element in warpsPerCTA must be a power of two.";
  }

  // Empty CTALayout is allowed, but if it's present its rank must match the
  // BlockedEncodingAttr's rank.
  if (order.size() != CTALayout.getRank()) {
    return emitError() << "BlockedEncodingAttr and CTALayout's fields must "
                          "have the same rank.";
  }
  return verifyLayoutOrder(emitError, order);
}

// 1 element per thread
// order = reverse(arange(rank))
triton::gpu::BlockedEncodingAttr
getDefaultBlockedEncoding(MLIRContext *context, ArrayRef<int64_t> shape,
                          int numWarps, int threadsPerWarp, int numCTAs) {
  int rank = shape.size();
  llvm::SmallVector<unsigned> order(rank);
  std::iota(order.begin(), order.end(), 0);
  std::reverse(order.begin(), order.end());
  llvm::SmallVector<unsigned> sizePerThread(rank, 1);
  triton::gpu::BlockedEncodingAttr encoding =
      triton::gpu::BlockedEncodingAttr::get(context, shape, sizePerThread,
                                            order, numWarps, threadsPerWarp,
                                            numCTAs);
  return encoding;
}

LogicalResult tryJoinOnAxis(MLIRContext *ctx, const LinearLayout &inLl,
                            LinearLayout &outLl, bool fwdInference, int axis,
                            std::optional<Location> loc) {
  auto kRegister = StringAttr::get(ctx, "register");
  auto outDims = llvm::to_vector(inLl.getOutDimNames());
  if (fwdInference) {
    auto split = LinearLayout::identity1D(2, kRegister, outDims[axis]);
    outLl = split * inLl;
  } else {
    // Assert that there is a dimension with size 2 in the axis
    // that has contiguous elements
    // Note that this is more general than the fwdInference case in that
    // - It allows the dimension not to be the fastest running
    // - It allows broadcasting
    // In general, this allows us to split along any axis as long as
    // the basis (0, 0, ..., 0, 1, 0, ..., 0) is in the registers.
    bool found = false;
    LinearLayout::BasesT newBases;
    for (const auto &basesDim : inLl.getBases()) {
      std::vector<std::vector<int32_t>> newBasesDim;
      for (auto base : basesDim.second) {
        if (base[axis] == 1 && basesDim.first == kRegister) {
          found = true;
          continue;
        }
        base[axis] /= 2;
        newBasesDim.push_back(std::move(base));
      }
      newBases.insert({basesDim.first, std::move(newBasesDim)});
    }
    if (!found)
      return emitOptionalError(loc,
                               "Fp4ToFpOp/SplitOp requires at least 2 elements "
                               "per thread in the axis/last dimension");
    outLl = LinearLayout(std::move(newBases), std::move(outDims));
  }
  return success();
}

} // namespace gpu
} // namespace triton
} // namespace mlir

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
    parser.emitError(parser.getNameLoc(), "expected a bool type in ") << desc;
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

static LogicalResult parseType(AsmParser &parser, const NamedAttribute &attr,
                               Type &value, StringRef desc) {
  auto typeAttr = mlir::dyn_cast<TypeAttr>(attr.getValue());
  if (!typeAttr) {
    parser.emitError(parser.getNameLoc(), "expected a Type in ") << desc;
    return failure();
  }
  value = typeAttr.getValue();
  return success();
}

std::optional<LinearLayout>
parseLinearLayout(const DictionaryAttr &dict, AsmParser &parser,
                  ArrayRef<std::string> inDimNames) {
  LinearLayout::BasesT bases;

  // Parse the basis names in order (the order is relevant)
  for (const auto &inDimNameStr : inDimNames) {
    auto inDimName = StringAttr::get(parser.getContext(), inDimNameStr);
    Attribute value = dict.get(inDimName);
    if (!value) {
      parser.emitError(parser.getCurrentLocation(), "Expected basis of '")
          << inDimName.getValue() << "' not found";
      return {};
    }
    // Expecting an array of arrays
    auto arrayOfArraysAttr = mlir::dyn_cast<ArrayAttr>(value);
    if (!arrayOfArraysAttr) {
      parser.emitError(parser.getCurrentLocation(),
                       "Expected array of arrays for basis of '")
          << inDimName.getValue() << "'";
      return {};
    }

    std::vector<std::vector<int32_t>> inDimBases;
    for (Attribute arrayAttr : arrayOfArraysAttr) {
      auto intArrayAttr = mlir::dyn_cast<ArrayAttr>(arrayAttr);
      if (!intArrayAttr) {
        parser.emitError(parser.getCurrentLocation(),
                         "Expected array of integers in basis for '")
            << inDimName.getValue() << "'";
        return {};
      }
      std::vector<int32_t> basis;
      for (Attribute intAttr : intArrayAttr) {
        auto intValueAttr = mlir::dyn_cast<IntegerAttr>(intAttr);
        if (!intValueAttr) {
          parser.emitError(parser.getCurrentLocation(),
                           "Expected integer in basis for '")
              << inDimName.getValue() << "'";
          return {};
        }
        basis.push_back(intValueAttr.getInt());
      }
      inDimBases.push_back(std::move(basis));
    }
    bases[inDimName] = std::move(inDimBases);
  }
  size_t rank = 0;
  for (const auto &basesDim : llvm::make_second_range(bases)) {
    if (!basesDim.empty()) {
      rank = basesDim[0].size();
      break;
    }
  }

  // To implement this we'd need to serialise the rank as well.
  // We can do this if we ever need it
  if (rank == 0) {
    parser.emitError(parser.getCurrentLocation(), "Empty Layout not supported");
    return {};
  }

  // Generate standared outDimNames (dim0, dim1, ...)
  SmallVector<StringAttr> outDimNames;
  for (int i = 0; i < rank; ++i) {
    outDimNames.push_back(
        StringAttr::get(parser.getContext(), "dim" + llvm::Twine(i)));
  }

  // Create LinearLayout
  return LinearLayout(std::move(bases), std::move(outDimNames));
}

// We don't use the default implementation as it's a bit too verbose
// This prints in the following format that is shape agnostic, in the sense
// that we don't print explicitly the outShape of the LL
// We always assume LLs to be surjective
// <{register = [[0, 1], [8, 0], [0, 8], [64, 0]],
//   lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
//   warp = [[16, 0], [32, 0]],
//   block = []}>
static void printLinearLayout(AsmPrinter &printer, const LinearLayout &ll) {
  printer << join(ll.getBases(), ", ", [](const auto &base) {
    return base.first.str() + " = " + "[" +
           join(base.second, ", ",
                [](const std::vector<int32_t> &vec) {
                  return "[" + join(vec, ", ") + "]";
                }) +
           "]";
  });
}

// Print the CTALayout if it's not equal to the default.
static void maybePrintCTALayout(mlir::MLIRContext *context,
                                mlir::AsmPrinter &printer, CTALayoutAttr layout,
                                unsigned rank) {
  if (layout != CTALayoutAttr::getDefault(context, rank)) {
    printer << ", CTAsPerCGA = [" << ArrayRef(layout.getCTAsPerCGA()) << "]"
            << ", CTASplitNum = [" << ArrayRef(layout.getCTASplitNum()) << "]"
            << ", CTAOrder = [" << ArrayRef(layout.getCTAOrder()) << "]";
  }
}

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGPU/IR/AttrInterfaces.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/AttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// Blocked Encoding
//===----------------------------------------------------------------------===//

static std::optional<CTALayoutAttr> getCTALayoutOrError(
    AsmParser &parser, std::optional<SmallVector<unsigned>> CTAsPerCGA,
    std::optional<SmallVector<unsigned>> CTASplitNum,
    std::optional<SmallVector<unsigned>> CTAOrder, unsigned rank) {
  if (CTAsPerCGA && CTASplitNum && CTAOrder) {
    return CTALayoutAttr::get(parser.getContext(), *CTAsPerCGA, *CTASplitNum,
                              *CTAOrder);
  }
  if (!CTAsPerCGA && !CTASplitNum && !CTAOrder) {
    return CTALayoutAttr::getDefault(parser.getContext(), rank);
  }
  parser.emitError(parser.getNameLoc(), "CTAsPerCGA, CTASplitNum, and CTAOrder "
                                        "must all be present or all be absent");
  return std::nullopt;
}

Attribute BlockedEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  SmallVector<unsigned> sizePerThread;
  SmallVector<unsigned> threadsPerWarp;
  SmallVector<unsigned> warpsPerCTA;
  SmallVector<unsigned> order;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;

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
    } else if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    } else if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    } else if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/sizePerThread.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<BlockedEncodingAttr>(parser.getContext(),
                                                sizePerThread, threadsPerWarp,
                                                warpsPerCTA, order, *CTALayout);
}

void BlockedEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "sizePerThread = [" << ArrayRef(getSizePerThread()) << "]"
          << ", threadsPerWarp = [" << ArrayRef(getThreadsPerWarp()) << "]"
          << ", warpsPerCTA = [" << ArrayRef(getWarpsPerCTA()) << "]"
          << ", order = [" << getOrder() << "]";

  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getSizePerThread().size());

  printer << "}>";
}

// FIXME Can we take the LinearLayout by const&?
LogicalResult
LinearEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           LinearLayout linearLayout) {
  // Example of LinearEncodingAttr
  // <{register = [[0, 1], [8, 0], [0, 8], [64, 0]],
  //   lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
  //   warp = [[16, 0], [32, 0]],
  //   block = []}>
  // The input dims must be {register, lane, warp, block}
  // The output dims of the linear layout should be dim0..dim[rank-1]

  static const auto expectedInDims =
      SmallVector<std::string>({"register", "lane", "warp", "block"});
  for (const auto &[i, dims] : llvm::enumerate(
           llvm::zip(linearLayout.getInDimNames(), expectedInDims))) {
    const auto &[dim, expectedDimStr] = dims;
    if (dim.str() != expectedDimStr) {
      return emitError() << "Expected input dimension " << i << " to be '"
                         << expectedDimStr << "'. Got " << dim;
    }
  }

  // outDims are ['dim0', 'dim1', ...]
  for (auto [i, dim] : llvm::enumerate(linearLayout.getOutDimNames())) {
    if (dim.str() != ("dim" + llvm::Twine(i)).str()) {
      return emitError()
             << "Expected output dimensions to be ['dim0', 'dim1', ...]. Got "
             << dim << " at position " << i;
    }
  }

  const auto &bases = linearLayout.getBases();
  auto nonZero = [](auto val) { return val != 0; };
  for (const auto &dimBases : llvm::make_second_range(bases)) {
    if (!llvm::all_of(dimBases, [&](const auto &basis) {
          return std::count_if(basis.begin(), basis.end(), nonZero) <= 1;
        })) {
      return emitError()
             << "In a distributed layout, each base must move in at most one "
                "dimension.";
    }
  }

  return success();
}

// If we only had BlockedEncodingAttr, we could simply return ArrayRefs here.
// But we need to have a consistent interface with e.g. SliceEncodingAttr, which
// computes some of these fields.
SmallVector<unsigned> BlockedEncodingAttr::getRepOrder() const {
  return SmallVector<unsigned>(getOrder());
}

//===----------------------------------------------------------------------===//
// Linear Encoding
//===----------------------------------------------------------------------===//

void LinearEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{";
  printLinearLayout(printer, getLinearLayout());
  printer << "}>";
}

Attribute LinearEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};

  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};

  if (parser.parseGreater().failed())
    return {};

  std::vector<std::string> inDimNames = {"register", "lane", "warp", "block"};
  auto maybeLL = parseLinearLayout(dict, parser, inDimNames);
  if (!maybeLL.has_value())
    return {};

  // Create and return the LinearEncodingAttr
  return parser.getChecked<LinearEncodingAttr>(parser.getContext(),
                                               std::move(*maybeLL));
}

static SmallVector<unsigned>
basesPerDimImpl(const LinearLayout::BasesT &namedBases, StringAttr dimName,
                size_t rank, bool skipBroadcast = true) {
  const auto &bases = namedBases.find(dimName)->second;

  if (bases.empty()) {
    return SmallVector<unsigned>(rank, 1);
  }

  SmallVector<unsigned> ret(rank, 1);
  auto nonZero = [](auto val) { return val != 0; };
  int nonZeroIdx = 0;
  for (const auto &basis : bases) {
    auto it = std::find_if(basis.begin(), basis.end(), nonZero);
    // Bases can have one or zero non-zero elements
    // Skip a basis if it's broadcasting (all zeros)
    // e.g. warps for DotOperandEncodingAttr (see ampereDotToLinearLayout)
    if (it != basis.end()) {
      nonZeroIdx = it - basis.begin();
      ret[nonZeroIdx] *= 2;
    } else if (!skipBroadcast) {
      // If we've seen a non-zero basis, we double the size of the previous dim
      // This is just needed to count the CTAsPerCGA
      ret[nonZeroIdx] *= 2;
    }
  }
  return ret;
}

SmallVector<unsigned>
LinearEncodingAttr::basesPerDim(StringAttr dimName, bool skipBroadcast) const {
  auto ll = getLinearLayout();
  auto rank = ll.getNumOutDims();
  return basesPerDimImpl(ll.getBases(), dimName, rank, skipBroadcast);
}

SmallVector<unsigned>
LinearEncodingAttr::orderPerDim(StringAttr dimName,
                                ArrayRef<unsigned> defaultOrder) const {
  return orderPerDimImpl(getLinearLayout(), dimName, defaultOrder);
}

// [Note. Divergence of methods wrt. legacy layouts]
// For smaller shapes where the CTATile is larger than the output
// tensor, some methods return different values than the legacy layouts. I think
// this is benign tho. An example: what is the vector of `warpsPerCTA` if
// all the warps hold the same data? I think it should be [1, 1], even if we
// have 4 warps. But perhaps for this we have to add some masking in some
// places... We'll see
SmallVector<unsigned> LinearEncodingAttr::getRepOrder() const {
  // This is not correct, but:
  // - It happens to agree in most places with the legacy layout
  // - getRepOrder does not make sense for LinearEncodingAttr as it already has
  //   the same shape as the tensor that uses it
  return getOrder();
}

SmallVector<unsigned> LinearEncodingAttr::getCTAsPerCGA() const {
  // CTAs are split into an identity part (SplitNum) and a broadcast part
  return basesPerDim(StringAttr::get(getContext(), "block"),
                     /*skipBroadcast=*/false);
}
SmallVector<unsigned> LinearEncodingAttr::getCTAOrder() const {
  return orderPerDim(StringAttr::get(getContext(), "block"), getOrder());
}
SmallVector<unsigned> LinearEncodingAttr::getCTASplitNum() const {
  return basesPerDim(StringAttr::get(getContext(), "block"));
}
SmallVector<unsigned> LinearEncodingAttr::getWarpsPerCTA() const {
  return basesPerDim(StringAttr::get(getContext(), "warp"));
}
SmallVector<unsigned> LinearEncodingAttr::getWarpOrder() const {
  return orderPerDim(StringAttr::get(getContext(), "warp"), getOrder());
}
SmallVector<unsigned> LinearEncodingAttr::getThreadsPerWarp() const {
  return basesPerDim(StringAttr::get(getContext(), "lane"));
}
SmallVector<unsigned> LinearEncodingAttr::getThreadOrder() const {
  return orderPerDim(StringAttr::get(getContext(), "lane"), getOrder());
}

SmallVector<unsigned> LinearEncodingAttr::getSizePerThread() const {
  auto rank = getOrder().size();
  auto ll = getLinearLayout();
  auto ctx = getContext();
  auto kRegister = StringAttr::get(ctx, "register");

  // We canonicalize on the spot, as if we use CGAs the regs are not in
  // canonical form The order is [reg, lane, warp, rep, block], so we first
  // remove the blocks
  llvm::SmallVector<unsigned> ctaShape;
  for (auto [shape, cgaNum] :
       llvm::zip(ll.getOutDimSizes(), getCTASplitNum())) {
    ctaShape.push_back(shape / cgaNum);
  }
  LinearLayout::BasesT bases = ll.getBases();

  llvm::SetVector<unsigned> reverseRepOrder;
  auto nonZero = [](auto val) { return val != 0; };
  auto &registers = bases[kRegister];
  while (!registers.empty()) {
    auto &basis = registers.back();
    auto it = std::find_if(basis.begin(), basis.end(), nonZero);
    // If there's broadcasting (base == zeros) there are no more reps
    if (it == basis.end()) {
      break;
    }
    auto dim = it - basis.begin();
    reverseRepOrder.insert(dim);
    // As soon as we stop finding reps, we stop
    if (dim != reverseRepOrder.back() || 2 * basis[dim] != ctaShape[dim]) {
      break;
    }
    ctaShape[dim] /= 2;
    registers.pop_back();
  }
  return basesPerDimImpl(bases, kRegister, rank);
}

SmallVector<unsigned> LinearEncodingAttr::getOrder() const {
  auto rank = getLinearLayout().getNumOutDims();
  SmallVector<unsigned> order(rank);
  // Choose [rank-1, rank-2, ... 0] as the default order in case
  // there are dims that do not move in the register
  // This order is as good as any really
  std::iota(order.rbegin(), order.rend(), 0);

  return orderPerDim(StringAttr::get(getContext(), "register"), order);
}

LinearLayout LinearEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  auto ll = getLinearLayout();
  auto canonicalDims = llvm::to_vector(ll.getOutDimNames());
  llvm::SmallDenseMap<StringAttr, int64_t> namedShape;
  llvm::SmallVector<StringAttr> permutedDims;
  for (auto dim : getRepOrder()) {
    permutedDims.push_back(canonicalDims[dim]);
    namedShape[canonicalDims[dim]] = shape[dim];
  }
  ll = ll.transposeOuts(permutedDims);
  ll = ensureLayoutNotSmallerThan(ll, namedShape);
  ll = ensureLayoutNotLargerThan(ll, namedShape, /*broadcastRegisters=*/false);
  ll = ll.transposeOuts(canonicalDims);
  return ll;
}

SmallVector<unsigned>
LinearEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape) const {
  // When broadcasting the layout the shape changes, otherwise the shape is
  // the same as the shape of the tensor
  // We can either have BroadcastOp with SameOperandsAndResultEncoding, or keep
  // the invariant that the shape of the LL is that of the tensor
  // We choose the former for BC
  auto scaledLayout = get(getContext(), toLinearLayout(shape));
  auto kRegister = StringAttr::get(getContext(), "register");
  return scaledLayout.basesPerDim(kRegister, /*skipBroadcast=*/false);
}

SmallVector<unsigned>
LinearEncodingAttr::getContig(const char *inDim,
                              SmallVector<unsigned int> lowerContig) const {
  auto ll = getLinearLayout();
  const auto &bases =
      ll.getBases().find(StringAttr::get(getContext(), inDim))->second;
  auto order = getOrder();
  auto rank = order.size();

  SmallVector<unsigned> contig(lowerContig);
  auto basisIt = bases.begin();
  for (unsigned dim : order) {
    std::vector<int32_t> basis(rank, 0);
    basis[dim] = contig[dim];

    while (basisIt != bases.end() && *basisIt == basis) {
      contig[dim] *= 2;
      basis[dim] *= 2;
      ++basisIt;
    }
  }
  return contig;
}

SmallVector<unsigned> LinearEncodingAttr::getContigPerThread() const {
  SmallVector<unsigned> contig(getOrder().size(), 1);
  return getContig("register", contig);
}

SmallVector<unsigned> LinearEncodingAttr::getContigPerWarp() const {
  return getContig("lane", getContigPerThread());
}

unsigned
LinearEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape) const {
  return product(getElemsPerThread(shape));
}

//===----------------------------------------------------------------------===//
// MMA encoding
//===----------------------------------------------------------------------===//

Attribute NvidiaMmaEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned versionMajor = 0;
  unsigned versionMinor = 0;
  SmallVector<unsigned> warpsPerCTA;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;
  SmallVector<unsigned> instrShape;

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
    if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    }
    if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    }
    if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    }
    if (attr.getName() == "instrShape") {
      if (parseIntArrayAttr(parser, attr, instrShape, "instrShape").failed()) {
        return {};
      }
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/warpsPerCTA.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<NvidiaMmaEncodingAttr>(
      parser.getContext(), versionMajor, versionMinor, warpsPerCTA, *CTALayout,
      instrShape);
}

void NvidiaMmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "versionMajor = " << getVersionMajor()
          << ", versionMinor = " << getVersionMinor() //
          << ", warpsPerCTA = [" << ArrayRef(getWarpsPerCTA()) << "]";

  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getRank());

  printer << ", instrShape = [" << getInstrShape() << "]}>";
}

//===----------------------------------------------------------------------===//
// MFMA encoding
//===----------------------------------------------------------------------===//

Attribute AMDMfmaEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned version = 0;
  SmallVector<unsigned> warpsPerCTA;
  SmallVector<unsigned> instrShape;
  bool isTransposed;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;
  SmallVector<unsigned> tilesPerWarp = {};
  unsigned elementBitWidth = 32;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "version") {
      if (parseUInt(parser, attr, version, "version").failed())
        return {};
    }
    if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA, "warpsPerCTA").failed())
        return {};
    }
    if (attr.getName() == "instrShape") {
      if (parseIntArrayAttr(parser, attr, instrShape, "instrShape").failed())
        return {};
    }
    if (attr.getName() == "isTransposed") {
      if (parseBool(parser, attr, isTransposed, "isTransposed").failed())
        return {};
    }
    if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    }
    if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    }
    if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    }
    if (attr.getName() == "tilesPerWarp") {
      if (parseIntArrayAttr(parser, attr, tilesPerWarp, "tilesPerWarp")
              .failed())
        return {};
    }
    if (attr.getName() == "elementBitWidth") {
      if (parseUInt(parser, attr, elementBitWidth, "elementBitWidth").failed())
        return {};
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/warpsPerCTA.size());
  if (!CTALayout.has_value())
    return {};

  if (tilesPerWarp.empty())
    tilesPerWarp = SmallVector<unsigned>(instrShape.size(), 1);

  return parser.getChecked<AMDMfmaEncodingAttr>(
      parser.getContext(), version, warpsPerCTA, instrShape, isTransposed,
      *CTALayout, tilesPerWarp, elementBitWidth);
}

void AMDMfmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "version = " << getVersion()                   //
          << ", warpsPerCTA = [" << getWarpsPerCTA() << "]" //
          << ", instrShape = [" << getInstrShape() << "]";

  printer << ", isTransposed = " << getIsTransposed();

  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getRank());

  auto tilesPerWarp = getTilesPerWarp();
  if (!hasUnitTilesPerWarp())
    printer << ", tilesPerWarp = [" << getTilesPerWarp() << "]";

  auto elementBitWidth = getElementBitWidth();
  if (elementBitWidth != 32)
    printer << ", elementBitWidth = " << elementBitWidth;

  printer << "}>";
}

LogicalResult AMDMfmaEncodingAttr::verify(
    function_ref<mlir::InFlightDiagnostic()> emitError, unsigned version,
    llvm::ArrayRef<unsigned int> warpsPerCTA,
    llvm::ArrayRef<unsigned int> instrShape, bool isTransposed,
    mlir::triton::gpu::CTALayoutAttr, llvm::ArrayRef<unsigned int> tilesPerWarp,
    unsigned elementBitWidth) {
  if (!(version >= 0 && version <= 4)) {
    return emitError() << "version must be in the [0, 4] range";
  }

  auto mDim = instrShape[0];
  auto nDim = instrShape[1];
  const std::array<std::pair<unsigned, unsigned>, 4> validDims = {
      {{32, 32}, {16, 16}, {64, 4}, {4, 64}}};
  if (!llvm::is_contained(validDims, std::make_pair(mDim, nDim))) {
    return emitError() << "invalid (mDim, nDim) combination: (" << mDim << ", "
                       << nDim << ")";
  }

  if (!(elementBitWidth == 32 || elementBitWidth == 64))
    return emitError() << "elementBitWidth must be 32 or 64";

  return success();
}

//===----------------------------------------------------------------------===//
// WMMA encoding
//===----------------------------------------------------------------------===//

Attribute AMDWmmaEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned version = 0;
  bool isTransposed = false;
  SmallVector<unsigned> warpsPerCTA;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;
  SmallVector<unsigned> instrShape = getDefaultInstrShape();

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "version") {
      if (parseUInt(parser, attr, version, "version").failed())
        return {};
    }
    if (attr.getName() == "isTranspose") {
      if (parseBool(parser, attr, isTransposed, "isTranspose").failed())
        return {};
    }
    if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA, "warpsPerCTA").failed())
        return {};
    }
    if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    }
    if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    }
    if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    }
    if (attr.getName() == "instrShape") {
      instrShape.clear();
      if (parseIntArrayAttr(parser, attr, instrShape, "instrShape").failed()) {
        return {};
      }
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/warpsPerCTA.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<AMDWmmaEncodingAttr>(parser.getContext(), version,
                                                isTransposed, warpsPerCTA,
                                                *CTALayout, instrShape);
}

void AMDWmmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "version = " << getVersion()
          << ", isTranspose = " << getIsTransposed() //
          << ", warpsPerCTA = [" << ArrayRef(getWarpsPerCTA()) << "]";

  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getWarpsPerCTA().size());

  if (getInstrShape() != ArrayRef(getDefaultInstrShape())) {
    printer << ", instrShape = [" << getInstrShape() << "]";
  }
  printer << "}>";
}

LogicalResult AMDWmmaEncodingAttr::verify(
    function_ref<mlir::InFlightDiagnostic()> emitError, unsigned version,
    bool isTransposed, llvm::ArrayRef<unsigned int> warpsPerCTA,
    CTALayoutAttr ctaLayout, llvm::ArrayRef<unsigned> instrShape) {
  if (!(version >= 1 && version <= 3))
    return emitError() << "WMMA version must be in the [1, 3] range";

  auto shape = SmallVector<unsigned>(instrShape);
  auto validShapesV1 = std::vector<llvm::SmallVector<unsigned>>{{16, 16, 16}};
  if (version == 1 && !llvm::is_contained(validShapesV1, shape))
    return emitError() << "invalid WMMA v1 instruction shape";

  auto validShapesV2 =
      std::vector<llvm::SmallVector<unsigned>>{{16, 16, 16}, {16, 16, 32}};
  if (version == 2 && !llvm::is_contained(validShapesV2, shape))
    return emitError() << "invalid WMMA v2 instruction shape";

  auto validShapesV3 = std::vector<llvm::SmallVector<unsigned>>{
      {16, 16, 4}, {16, 16, 32}, {16, 16, 64}, {16, 16, 128}};
  if (version == 3 && !llvm::is_contained(validShapesV3, shape))
    return emitError() << "invalid WMMA v3 instruction shape";

  return success();
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
  unsigned dim = mlir::cast<IntegerAttr>(attrs.get("dim")).getInt();
  auto parent = mlir::dyn_cast<DistributedEncodingTrait>(attrs.get("parent"));
  if (!parent) {
    parser.emitError(parser.getNameLoc(),
                     "expected a distributed encoding trait");
    return {};
  }
  return parser.getChecked<SliceEncodingAttr>(parser.getContext(), dim, parent);
}

void SliceEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "dim = " << getDim() << ", "
          << "parent = " << getParent() << "}>";
}

LogicalResult
SliceEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          unsigned dim, DistributedEncodingTrait parent) {
  unsigned rank = cast<LayoutEncodingTrait>(parent).getRank();
  if (rank <= 1)
    return emitError() << "parent layout must have at least rank >= 2";
  if (dim >= rank) {
    return emitError() << "slice dim=" << dim
                       << " must be less than the parent rank=" << rank;
  }
  return success();
}

SmallVector<unsigned> SliceEncodingAttr::getRepOrder() const {
  auto parentRepOrder = getParent().getRepOrder();
  return eraseOrder(parentRepOrder, getDim());
}

SmallVector<unsigned> SliceEncodingAttr::getCTASplitNum() const {
  SmallVector<unsigned> res = ::getCTASplitNum(getParent());
  res.erase(res.begin() + getDim());
  return res;
}

SmallVector<unsigned> SliceEncodingAttr::getCTAOrder() const {
  auto parentCTAOrder = ::getCTAOrder(getParent());
  return eraseOrder(parentCTAOrder, getDim());
}

SmallVector<unsigned> SliceEncodingAttr::getCTAsPerCGA() const {
  auto parentCTAsPerCGA = ::getCTAsPerCGA(getParent());
  if (parentCTAsPerCGA[getDim()] == 1) {
    parentCTAsPerCGA.erase(parentCTAsPerCGA.begin() + getDim());
    return parentCTAsPerCGA;
  }
  /* For getCTAsPerCGA of a slice layout, we have two choices:
   * (1) Return CTAsPerCGA of its parent. This is not a perfect solution
   * because the rank of the returned CTAsPerCGA does not match the rank of
   * tensorShape.
   * (2) Get CTAsPerCGA of its parent and erase the sliced dim. This is not a
   * perfect solution because the product of the returned CTAsPerCGA might not
   * match numCTAs.
   * To avoid introducing inconsistencies to the shape and
   * layout system, the usage of directly getting CTAsPerCGA of a slice layout
   * in which the sliced dim is not 1 is banned. You should always consider
   * slice layout as a special case and use getCTAsPerCGA(layout.getParent())
   * in the branch where layout is an instance of SliceEncodingAttr. This is
   * inconvenient but safe.
   */
  llvm::report_fatal_error(
      "getCTAsPerCGA for SliceEncodingAttr is not well-defined");
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

//===----------------------------------------------------------------------===//
// Helper shared encoding functions
//===----------------------------------------------------------------------===//

std::optional<CTALayoutAttr>
parseCTAAttrs(AsmParser &parser, NamedAttrList attrList, unsigned rank) {
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;

  for (const NamedAttribute &attr : attrList) {
    if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    } else if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    } else if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  return getCTALayoutOrError(parser, CTAsPerCGA, CTASplitNum, CTAOrder, rank);
}

template <typename SpecificEncoding>
Attribute parseSwizzledEncoding(AsmParser &parser, Type type) {
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
  SmallVector<unsigned> order;
  NamedAttrList remainingAttrs;
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
      remainingAttrs.push_back(attr);
    }
  }

  if (auto CTALayout = parseCTAAttrs(parser, remainingAttrs, order.size()))
    return parser.getChecked<SpecificEncoding>(
        parser.getContext(), vec, perPhase, maxPhase, order, *CTALayout);
  return {};
}

//===----------------------------------------------------------------------===//
// SwizzledShared encoding
//===----------------------------------------------------------------------===//

LogicalResult
SwizzledSharedEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   unsigned vec, unsigned perPhase,
                                   unsigned maxPhase, ArrayRef<unsigned> order,
                                   CTALayoutAttr ctaLayout) {
  if (order.size() != ctaLayout.getRank()) {
    return emitError() << "order size (" << order.size()
                       << ") must match CTALayout rank (" << ctaLayout.getRank()
                       << ")";
  }
  return verifyLayoutOrder(emitError, order);
}

Attribute SwizzledSharedEncodingAttr::parse(AsmParser &parser, Type type) {
  return parseSwizzledEncoding<SwizzledSharedEncodingAttr>(parser, type);
}

void SwizzledSharedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "vec = " << getVec() //
          << ", perPhase = " << getPerPhase()
          << ", maxPhase = " << getMaxPhase() //
          << ", order = [" << getOrder() << "]";
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getOrder().size());
  printer << "}>";
}

//===----------------------------------------------------------------------===//
// SharedLinear encoding
//===----------------------------------------------------------------------===//

LogicalResult
SharedLinearEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 LinearLayout linearLayout,
                                 unsigned layoutAlignment) {
  if (layoutAlignment == 0 || !llvm::isPowerOf2_32(layoutAlignment)) {
    return emitError() << "alignment must be a positive power of two";
  }
  static const auto expectedInDims =
      SmallVector<std::string>({"offset", "block"});
  for (const auto &[index, dims] : llvm::enumerate(
           llvm::zip(linearLayout.getInDimNames(), expectedInDims))) {
    const auto &[dim, expected] = dims;
    if (dim.str() != expected) {
      return emitError() << "Expected input dimension " << index << " to be '"
                         << expected << "'. Got " << dim;
    }
  }

  for (auto [i, dim] : llvm::enumerate(linearLayout.getOutDimNames())) {
    if (dim.str() != ("dim" + llvm::Twine(i)).str()) {
      return emitError()
             << "Expected output dimensions to be ['dim0', 'dim1', ...]. Got "
             << dim << " at position " << i;
    }
  }

  SmallVector<StringAttr> outDimNames =
      llvm::to_vector(linearLayout.getOutDimNames());
  if (outDimNames.empty()) {
    return emitError()
           << "SharedLinearEncodingAttr requires at least one output"
              " dimension.";
  }

  auto *ctx = outDimNames.front().getContext();
  auto kOffset = StringAttr::get(ctx, "offset");
  auto kBlock = StringAttr::get(ctx, "block");

  if (!linearLayout.isSurjective()) {
    return emitError() << "The layout must be surjective";
  }

  LinearLayout withoutBroadcast =
      linearLayout.removeZeroBasesAlongDim(kOffset).removeZeroBasesAlongDim(
          kBlock);
  if (!withoutBroadcast.isInvertible()) {
    return emitError()
           << "After removing the zero bases the layout must be bijective";
  }

  return success();
}

void SharedLinearEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{";
  auto layout = getLinearLayout();
  auto kBlock = StringAttr::get(getContext(), "block");
  auto kOffset = StringAttr::get(getContext(), "offset");
  if (layout.getBases().lookup(kBlock).empty()) {
    layout =
        layout.sublayout({kOffset}, llvm::to_vector(layout.getOutDimNames()));
  }
  printLinearLayout(printer, layout);
  printer << "}, alignment = " << getAlignment() << ">";
}

Attribute SharedLinearEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};

  DictionaryAttr layoutDictRaw;
  if (parser.parseAttribute(layoutDictRaw).failed())
    return {};

  if (layoutDictRaw.get("alignment")) {
    parser.emitError(parser.getCurrentLocation())
        << "alignment must be specified outside of the linear layout braces";
    return {};
  }

  NamedAttrList layoutAttrList(layoutDictRaw.getValue());
  auto *ctx = parser.getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  if (!layoutAttrList.get(kBlock)) {
    layoutAttrList.push_back({kBlock, ArrayAttr::get(ctx, {})});
  }

  DictionaryAttr layoutDict = layoutAttrList.getDictionary(ctx);

  // Parse alignment
  unsigned layoutAlignment;
  if (parser.parseComma().failed())
    return {};
  if (parser.parseKeyword("alignment").failed() || parser.parseEqual().failed())
    return {};
  if (parser.parseInteger(layoutAlignment).failed())
    return {};

  if (parser.parseGreater().failed())
    return {};

  std::vector<std::string> inDimNames = {"offset", "block"};
  auto maybeLL = parseLinearLayout(layoutDict, parser, inDimNames);
  if (!maybeLL.has_value())
    return {};

  // Special case for cleaner errors
  if (layoutDict.get("alignment")) {
    parser.emitError(parser.getCurrentLocation())
        << "alignment must be specified outside of the linear layout braces";
    return {};
  }

  if (layoutDict.size() != 2) {
    parser.emitError(parser.getCurrentLocation())
        << "SharedLinearEncodingAttr must have exactly two attributes: offset "
           "and block";
    return {};
  }

  return parser.getChecked<SharedLinearEncodingAttr>(
      parser.getContext(), std::move(*maybeLL), layoutAlignment);
}

SmallVector<unsigned>
SharedLinearEncodingAttr::basesPerDim(StringAttr dimName,
                                      bool skipBroadcast) const {
  auto ll = getLinearLayout();
  auto rank = ll.getNumOutDims();
  return basesPerDimImpl(ll.getBases(), dimName, rank, skipBroadcast);
}

SmallVector<unsigned>
SharedLinearEncodingAttr::orderPerDim(StringAttr dimName,
                                      ArrayRef<unsigned> defaultOrder) const {
  return orderPerDimImpl(getLinearLayout(), dimName, defaultOrder);
}

SmallVector<unsigned> SharedLinearEncodingAttr::getOrder() const {
  auto ll = getLinearLayout();
  auto rank = ll.getNumOutDims();
  SmallVector<unsigned> defaultOrder(rank);
  std::iota(defaultOrder.rbegin(), defaultOrder.rend(), 0);
  return orderPerDim(StringAttr::get(getContext(), "offset"), defaultOrder);
}

SmallVector<unsigned> SharedLinearEncodingAttr::getCTAsPerCGA() const {
  return basesPerDim(StringAttr::get(getContext(), "block"),
                     /*skipBroadcast=*/false);
}

SmallVector<unsigned> SharedLinearEncodingAttr::getCTAOrder() const {
  return orderPerDim(StringAttr::get(getContext(), "block"), getOrder());
}

SmallVector<unsigned> SharedLinearEncodingAttr::getCTASplitNum() const {
  return basesPerDim(StringAttr::get(getContext(), "block"));
}

LinearLayout
SharedLinearEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  auto ll = getLinearLayout();
  auto outDimNames = llvm::to_vector(ll.getOutDimNames());
  assert(shape.size() == outDimNames.size());
  // We don't support automatic broadcasting for shared linear layouts
  for (auto [size, llSize] : llvm::zip(shape, ll.getOutDimSizes())) {
    assert(size == llSize);
  }
  return ll;
}

//===----------------------------------------------------------------------===//
// PaddedShared encoding
//===----------------------------------------------------------------------===//

Attribute PaddedSharedEncodingAttr::parse(AsmParser &parser, Type type) {
  // <[
  if (failed(parser.parseLess()) || failed(parser.parseLSquare()))
    return {};

  // <interval_i>:+<padding_i>
  SmallVector<unsigned, 4> intervals, paddings;
  auto parseIntervalPaddingPair = [&]() {
    unsigned interval = 0, padding = 0;
    if (failed(parser.parseInteger(interval)) || failed(parser.parseColon()) ||
        failed(parser.parsePlus()) || failed(parser.parseInteger(padding)))
      return failure();
    intervals.push_back(interval);
    paddings.push_back(padding);
    return success();
  };
  // ]
  if (failed(parser.parseCommaSeparatedList(parseIntervalPaddingPair)) ||
      failed(parser.parseRSquare()))
    return {};

  // {<attr-dict>}
  auto attrList = DictionaryAttr::get(parser.getContext());
  if (failed(parser.parseAttribute(attrList)))
    return {};

  // We have 2 possible formats for the attr-dict:
  //  1) offset=[..], block=[..] handled by parseLinearLayout
  //  2) order=[..], shape=[..] which creates an identity mapping

  std::optional<LinearLayout> maybeLL;
  // Assume it's the first variant if offset or block is defined
  if (attrList.contains("offset") || attrList.contains("block")) {
    std::vector<std::string> inDimNames = {"offset", "block"};
    // Error out on additional attribute names
    for (const NamedAttribute &attr : attrList) {
      if (!llvm::is_contained(inDimNames, attr.getName())) {
        parser.emitError(parser.getCurrentLocation(), "Unexpected attribute ")
            << attr.getName() << " found";
      }
    }
    maybeLL = parseLinearLayout(attrList, parser, inDimNames);
  } else {
    // Parse the second form
    SmallVector<unsigned> order;
    SmallVector<unsigned> shape;
    for (const NamedAttribute &attr : attrList) {
      if (attr.getName() == "order") {
        if (parseIntArrayAttr(parser, attr, order, "order").failed())
          return {};
      } else if (attr.getName() == "shape") {
        if (parseIntArrayAttr(parser, attr, shape, "shape").failed())
          return {};
      } else {
        parser.emitError(parser.getCurrentLocation(), "Unexpected attribute ")
            << attr.getName() << " found";
        return {};
      }
    }

    if (order.size() != shape.size()) {
      parser.emitError(parser.getCurrentLocation(),
                       "Mismatch of shape and order ranks in padded layout");
      return {};
    }

    // Create identity mapping based on shape and order
    auto kOffset = StringAttr::get(parser.getContext(), "offset");
    maybeLL = identityStandardND(kOffset, shape, order);
    maybeLL = combineCtaCgaWithShape(
        *maybeLL, CTALayoutAttr::getDefault(parser.getContext(), shape.size()),
        SmallVector<int64_t>(ArrayRef(shape)));
  }

  if (!maybeLL.has_value())
    return {};

  // >
  if (parser.parseGreater().failed())
    return {};

  return parser.getChecked<PaddedSharedEncodingAttr>(
      parser.getContext(), intervals, paddings, *maybeLL);
}

void PaddedSharedEncodingAttr::print(AsmPrinter &printer) const {

  auto *ctx = getContext();
  const auto &ll = getLinearComponent();

  printer << "<[";
  llvm::interleaveComma(llvm::zip(getIntervals(), getPaddings()), printer,
                        [&](std::tuple<unsigned, unsigned> intervalPad) {
                          printer << std::get<0>(intervalPad) << ":+"
                                  << std::get<1>(intervalPad);
                        });
  printer << "] {";

  // We have a short hand form if linearComponent:
  //  1) does have an empty CTA layout (empty block dim)
  //  2) offsets are an identity mapping
  auto kOffset = StringAttr::get(ctx, "offset");
  auto kBlock = StringAttr::get(ctx, "block");
  auto shape = SmallVector<unsigned>(ll.getOutDimSizes());

  bool hasEmptyBlock = ll.getInDimSizeLog2(kBlock) == 0;

  LinearLayout identity = identityStandardND(kOffset, shape, getOrder())
                              .transposeOuts(to_vector(ll.getOutDimNames()));
  auto offsetLayout = ll.sublayout({kOffset}, to_vector(ll.getOutDimNames()));

  if (hasEmptyBlock && offsetLayout == identity) {
    printer << "order = [" << ArrayRef(getOrder()) << "], shape = ["
            << ArrayRef(shape) << "]";
  } else {
    printLinearLayout(printer, getLinearComponent());
  }

  printer << "}>";
}

LogicalResult PaddedSharedEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<unsigned> intervals,
    ArrayRef<unsigned> paddings, LinearLayout linearComponent) {
  if (intervals.size() != paddings.size())
    return emitError() << "intervals size (" << intervals.size()
                       << ") must match paddings size (" << paddings.size()
                       << ")";

  if (intervals.empty())
    return emitError() << "must have at least one interval-padding pair";

  if (!llvm::all_of(intervals, llvm::isPowerOf2_32))
    return emitError() << "interval values must all be power of two";
  if (!llvm::all_of(paddings, llvm::isPowerOf2_32))
    return emitError() << "padding values must all be power of two";

  llvm::SmallSet<unsigned, 4> intervalValues(intervals.begin(),
                                             intervals.end());
  if (intervalValues.size() != intervals.size())
    return emitError() << "interval values cannot have duplicates";

  const auto &ll = linearComponent;
  // The linear layout should map from [offset, block] to [dim0..dimN). All
  // bases should be 0 or power of twos and move in a single direction without
  // broadcasting

  if (ll == LinearLayout::empty())
    return emitError() << "linearComponent cannot be empty";

  assert(!ll.getInDimNames().empty());
  auto *ctx = ll.getInDimNames().begin()->getContext();

  if (!llvm::equal(ll.getInDimNames(),
                   std::array{StringAttr::get(ctx, "offset"),
                              StringAttr::get(ctx, "block")})) {
    return emitError()
           << "linearComponent must have [offset, block] as input dims";
  }

  if (!llvm::equal(ll.getOutDimNames(),
                   standardOutDimNames(ctx, ll.getNumOutDims()))) {
    return emitError()
           << "Expected output dimensions to be ['dim0', 'dim1', ...].";
  }

  const auto &bases = ll.getBases();

  // Check that we are not broadcasting or having repeated bases
  if (!ll.isInvertible()) {
    return emitError() << "Broadcasting is not supported.";
  }

  auto nonZero = [](auto val) { return val != 0; };
  for (const auto &dimBases : llvm::make_second_range(bases)) {
    if (!llvm::all_of(dimBases, [&](const auto &basis) {
          return llvm::count_if(basis, nonZero) <= 1;
        })) {
      return emitError()
             << "Each offset basis must move in at most one dimension.";
    }
    // Ensure all non zero elements are a power of 2. Combined with the
    // broadcast check above this prevents per element swizzling. The intent of
    // the linear component is to rearrange whole rows or cache-line sized
    // chunks of rows.
    if (!llvm::all_of(dimBases, [&](const auto &basis) {
          return llvm::all_of(
              basis, [](auto v) { return v == 0 || llvm::isPowerOf2_32(v); });
        })) {
      return emitError() << "Each offset basis must be 0 or a power of two.";
    }
  }

  return success();
}

PaddedSharedEncodingAttr PaddedSharedEncodingAttr::get(
    MLIRContext *context, ArrayRef<std::pair<unsigned, unsigned>> intervalPads,
    ArrayRef<unsigned> order, ArrayRef<int64_t> shape,
    CTALayoutAttr ctaLayout) {
  auto outDimNames = standardOutDimNames(context, shape.size());
  StringAttr kOffset = StringAttr::get(context, "offset");

  // Create identity mapping based on shape and order
  LinearLayout linearComponent =
      identityStandardND(kOffset, SmallVector<unsigned>(shape), order);
  linearComponent = combineCtaCgaWithShape(linearComponent, ctaLayout, shape);

  return get(context, intervalPads, linearComponent);
}

PaddedSharedEncodingAttr PaddedSharedEncodingAttr::get(
    MLIRContext *context, ArrayRef<std::pair<unsigned, unsigned>> intervalPads,
    LinearLayout linearComponent) {
  SmallVector<unsigned> intervals, paddings;
  intervals.reserve(intervalPads.size());
  paddings.reserve(intervalPads.size());
  for (auto [interval, padding] : intervalPads) {
    intervals.push_back(interval);
    paddings.push_back(padding);
  }
  return get(context, intervals, paddings, linearComponent);
}

SmallVector<unsigned>
PaddedSharedEncodingAttr::basesPerDim(StringAttr dimName,
                                      bool skipBroadcast) const {
  const auto &ll = getLinearComponent();
  auto rank = ll.getNumOutDims();
  return basesPerDimImpl(ll.getBases(), dimName, rank, skipBroadcast);
}

int64_t PaddedSharedEncodingAttr::getPaddedSize(ArrayRef<int64_t> shape) const {
  int64_t unpaddedSize = product(shape);
  int64_t paddingSize = 0;
  for (auto [interval, padding] :
       llvm::zip_equal(getIntervals(), getPaddings())) {
    paddingSize += (unpaddedSize >> llvm::Log2_32(interval))
                   << llvm::Log2_32(padding);
    // There is no need for padding after the last element
    if (unpaddedSize % interval == 0)
      paddingSize -= padding;
  }
  return unpaddedSize + paddingSize;
}

SmallVector<unsigned>
PaddedSharedEncodingAttr::orderPerDim(StringAttr dimName,
                                      ArrayRef<unsigned> defaultOrder) const {
  return orderPerDimImpl(getLinearComponent(), dimName, defaultOrder);
}

SmallVector<unsigned> PaddedSharedEncodingAttr::getOrder() const {
  auto rank = getLinearComponent().getNumOutDims();
  SmallVector<unsigned> order(rank);
  // Choose [rank-1, rank-2, ... 0] as the default order in case
  // there are dims that do not move in the offsets
  std::iota(order.rbegin(), order.rend(), 0);

  return orderPerDim(StringAttr::get(getContext(), "offset"), order);
}

// LayoutEncodingTrait, ["getCTAsPerCGA", "getCTAOrder", "getCTASplitNum"]>;
SmallVector<unsigned> PaddedSharedEncodingAttr::getCTAsPerCGA() const {
  // CTAs are split into an identity part (SplitNum) and a broadcast part
  return basesPerDim(StringAttr::get(getContext(), "block"),
                     /*skipBroadcast=*/false);
}
SmallVector<unsigned> PaddedSharedEncodingAttr::getCTAOrder() const {
  return orderPerDim(StringAttr::get(getContext(), "block"), getOrder());
}
SmallVector<unsigned> PaddedSharedEncodingAttr::getCTASplitNum() const {
  return basesPerDim(StringAttr::get(getContext(), "block"));
}

//===----------------------------------------------------------------------===//
// NVMMAShared encoding
//===----------------------------------------------------------------------===//

Attribute NVMMASharedEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned swizzlingByteWidth;
  bool transposed = false;
  bool fp4Padded = false;
  unsigned elementBitWidth;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;
  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "swizzlingByteWidth") {
      if (parseUInt(parser, attr, swizzlingByteWidth, "swizzlingByteWidth")
              .failed())
        return {};
    } else if (attr.getName() == "transposed") {
      if (parseBool(parser, attr, transposed, "transposed").failed())
        return {};
    } else if (attr.getName() == "elementBitWidth") {
      if (parseUInt(parser, attr, elementBitWidth, "elementBitWidth").failed())
        return {};
    } else if (attr.getName() == "fp4Padded") {
      if (parseBool(parser, attr, fp4Padded, "fp4Padded").failed())
        return {};
    } else if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    } else if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    } else if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/2);
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<NVMMASharedEncodingAttr>(
      parser.getContext(), swizzlingByteWidth, transposed, elementBitWidth,
      fp4Padded, *CTALayout);
}

void NVMMASharedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "swizzlingByteWidth = " << getSwizzlingByteWidth() //
          << ", transposed = " << getTransposed()               //
          << ", elementBitWidth = " << getElementBitWidth();
  if (getFp4Padded()) {
    // Print only in this case to reduce the noise for the more common case.
    printer << ", fp4Padded = true";
  }
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/2);
  printer << "}>";
}

int NVMMASharedEncodingAttr::getVec() const {
  if (getSwizzlingByteWidth() == 0)
    return 1;
  return 128 / getElementBitWidth();
}

int NVMMASharedEncodingAttr::getPerPhase() const {
  if (getSwizzlingByteWidth() == 0)
    return 1;
  return 128 / getSwizzlingByteWidth();
}

int NVMMASharedEncodingAttr::getMaxPhase() const {
  if (getSwizzlingByteWidth() == 0)
    return 1;
  return getSwizzlingByteWidth() / 16;
}

int32_t NVMMASharedEncodingAttr::getAlignment() const {
  return 128 * getMaxPhase();
}

//===----------------------------------------------------------------------===//
// AMDRotatingShared encoding
//===----------------------------------------------------------------------===//

Attribute AMDRotatingSharedEncodingAttr::parse(AsmParser &parser, Type type) {
  return parseSwizzledEncoding<AMDRotatingSharedEncodingAttr>(parser, type);
}

void AMDRotatingSharedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "vec = " << getVec() //
          << ", perPhase = " << getPerPhase()
          << ", maxPhase = " << getMaxPhase() //
          << ", order = [" << getOrder() << "]";
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getOrder().size());
  printer << "}>";
}

//===----------------------------------------------------------------------===//
// Mfma encoding
//===----------------------------------------------------------------------===//
// TODO: there is a lot of common code with MmaEncoding here

bool AMDMfmaEncodingAttr::hasUnitTilesPerWarp() const {
  return !llvm::any_of(getTilesPerWarp(), [](int x) { return x != 1; });
}

SmallVector<int64_t>
AMDMfmaEncodingAttr::getInstrShapeForOperand(int kWidth, int opIdx) const {
  auto mnkDim = getInstrShape();
  unsigned mDim = mnkDim[0];
  unsigned nDim = mnkDim[1];
  assert((mDim == nDim) && (mDim == 32 || mDim == 16 || mDim == 4) ||
         (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));

  constexpr int warpSize = 64; // MFMA is always based on the 64-wide warps.
  int kGroups = warpSize / std::min(mDim, nDim); // for 64x4 and 4x64,
                                                 // kGroups = 16
  int64_t kDim = kWidth * kGroups;

  if (opIdx == 0)
    return {mDim, kDim};
  else
    assert(opIdx == 1);
  return {kDim, nDim};
}

SmallVector<unsigned> AMDMfmaEncodingAttr::getRepOrder() const {
  return getMatrixOrder(getRank(), /*rowMajor*/ true);
}

SmallVector<unsigned>
AMDMfmaEncodingAttr::getRepOrderForOperand(int opIdx) const {
  return getOrderForDotOperand(opIdx, getRank(), /*kContig*/ true);
}

SmallVector<int64_t>
AMDMfmaEncodingAttr::getRepForOperand(ArrayRef<int64_t> operandShape,
                                      int kWidth, int opIdx) const {
  auto operandTileShape = getInstrShapeForOperand(kWidth, opIdx);
  auto rank = operandShape.size();
  auto warpsPerCTA = getWarpsPerCTA();
  auto tilesPerWarp = getTilesPerWarp();

  int numRepBatch =
      rank == 3 ? std::max<int64_t>(1, operandShape[0] / warpsPerCTA[0]) : 1;
  if (opIdx == 0)
    return {
        numRepBatch,
        std::max<int64_t>(1, operandShape[rank - 2] /
                                 (operandTileShape[0] * tilesPerWarp[rank - 2] *
                                  warpsPerCTA[rank - 2])) *
            tilesPerWarp[rank - 2],
        std::max<int64_t>(1, operandShape[rank - 1] / operandTileShape[1])};
  else {
    assert(opIdx == 1);
    return {
        numRepBatch,
        std::max<int64_t>(1, operandShape[rank - 2] / operandTileShape[0]),
        std::max<int64_t>(1, operandShape[rank - 1] /
                                 (operandTileShape[1] * tilesPerWarp[rank - 1] *
                                  warpsPerCTA[rank - 1])) *
            tilesPerWarp[rank - 1]};
  }
}

SwizzledSharedEncodingAttr AMDMfmaEncodingAttr::composeSharedLayoutForOperand(
    CTALayoutAttr ctaLayout, int operandIdx, ArrayRef<int64_t> operandShape,
    ArrayRef<unsigned> sharedOrder, unsigned vectorSize, unsigned elemBitWidth,
    bool needTrans) const {
  int kDimIndex = operandIdx == 0 ? 1 : 0;

  // Disable swizzling for scales
  if (operandIdx >= 2) {
    return SwizzledSharedEncodingAttr::get(getContext(), 1, 1, 1, sharedOrder,
                                           ctaLayout);
  }

  if (needTrans)
    kDimIndex = 1 - kDimIndex;

  bool isKContig = sharedOrder[0] == kDimIndex;
  // GFX950 supports LDS transpose load instructions, so we need swizzling even
  // when K dimension is not the contiguous dimension.
  bool isGFX950 = getVersion() == 4;
  bool swizzleNonKContig =
      isGFX950 && (elemBitWidth == 8 || elemBitWidth == 16);

  if (!isKContig && !swizzleNonKContig) {
    // Do not swizzle. In this case accesses will go in different banks even
    // without swizzling.
    return SwizzledSharedEncodingAttr::get(getContext(), 1, 1, 1, sharedOrder,
                                           ctaLayout);
  }

  const unsigned numBanks = isGFX950 ? 64 : 32;
  const unsigned bankBitWidth = 32;
  const unsigned simdWidth = 16;

  // Number of inner dimension rows per one pattern repeat
  int innerDimLength = operandShape[sharedOrder[0]];
  int elemsPerOneBanksRow = (numBanks * bankBitWidth) / elemBitWidth;

  int perPhase = std::max(1, elemsPerOneBanksRow / innerDimLength);
  int maxPhase =
      std::max(std::min(simdWidth / perPhase, innerDimLength / vectorSize), 1u);

  // TODO (zhanglx): figure out better parameters for mfma4
  if (getInstrShape()[0] == 4)
    maxPhase = 4;

  return SwizzledSharedEncodingAttr::get(getContext(), vectorSize, perPhase,
                                         maxPhase, sharedOrder, ctaLayout);
}

//===----------------------------------------------------------------------===//
// Wmma encoding
//===----------------------------------------------------------------------===//

SmallVector<unsigned> AMDWmmaEncodingAttr::getRepOrder() const {
  return getMatrixOrder(getRank(), /*rowMajor*/ true);
}

SmallVector<unsigned>
AMDWmmaEncodingAttr::getRepOrderForOperand(int opIdx) const {
  return getOrderForDotOperand(opIdx, getRank(), /*kContig*/ true);
}

SmallVector<int64_t>
AMDWmmaEncodingAttr::getRepForOperand(ArrayRef<int64_t> operandShape, int kDim,
                                      int opIdx) const {
  auto mnkDim = getInstrShape();
  SmallVector<int64_t, 2> operandTileShape{opIdx == 0 ? mnkDim[0] : kDim,
                                           opIdx == 0 ? kDim : mnkDim[1]};

  assert(operandTileShape.size() == 2);
  auto warpsPerCTA = getWarpsPerCTA();
  auto rank = operandShape.size();
  assert(rank == 2 || rank == 3);
  int numRepBatch =
      rank == 3 ? std::max<int64_t>(1, operandShape[0] / warpsPerCTA[0]) : 1;
  if (opIdx == 0)
    return {
        numRepBatch,
        std::max<int64_t>(1, operandShape[rank - 2] /
                                 (operandTileShape[0] * warpsPerCTA[rank - 2])),
        std::max<int64_t>(1, operandShape[rank - 1] / operandTileShape[1])};
  else {
    assert(opIdx == 1);
    return {
        numRepBatch,
        std::max<int64_t>(1, operandShape[rank - 2] / operandTileShape[0]),
        std::max<int64_t>(1, operandShape[rank - 1] / (operandTileShape[1] *
                                                       warpsPerCTA[rank - 1]))};
  }
}

SwizzledSharedEncodingAttr AMDWmmaEncodingAttr::composeSharedLayoutForOperand(
    CTALayoutAttr ctaLayout, int operandIdx, ArrayRef<int64_t> operandShape,
    ArrayRef<unsigned> sharedOrder, unsigned kWidth, unsigned elemBitWidth,
    bool needTrans) const {
  int kDimIndex = operandIdx == 0 ? 1 : 0;
  bool isKContig = sharedOrder[0] == kDimIndex;

  if (!isKContig) {
    // Do not swizzle. In this case accesses will go in different banks even
    // without swizzling.
    return SwizzledSharedEncodingAttr::get(getContext(), 1, 1, 1, sharedOrder,
                                           ctaLayout);
  }

  // max vectorization size for ds_load is 128 bits
  int vectorSize = std::min(kWidth * elemBitWidth, 128u) / elemBitWidth;

  const int numBanks = 32;
  const int bankBitWidth = 32;

  // Number of inner dimension rows per one pattern repeat
  int innerDimLength = operandShape[sharedOrder[0]];
  int elemsPerOneBanksRow = (numBanks * bankBitWidth) / elemBitWidth;

  int perPhase = std::max(1, elemsPerOneBanksRow / innerDimLength);
  // for both RDNA3 and RDNA4, the M/N dimension of wmma is 16
  // This represents the max number of rows that can be accessed
  // at the same time
  int mDim = getInstrShape()[0];
  int maxPhase =
      std::max(std::min(mDim / perPhase, innerDimLength / vectorSize), 1);

  return SwizzledSharedEncodingAttr::get(getContext(), vectorSize, perPhase,
                                         maxPhase, sharedOrder, ctaLayout);
}

//===----------------------------------------------------------------------===//
// Mma encoding
//===----------------------------------------------------------------------===//

bool NvidiaMmaEncodingAttr::isVolta() const { return getVersionMajor() == 1; }

bool NvidiaMmaEncodingAttr::isTuring() const {
  return getVersionMajor() == 2 && getVersionMinor() == 1;
}

bool NvidiaMmaEncodingAttr::isAmpere() const { return getVersionMajor() == 2; }

bool NvidiaMmaEncodingAttr::isHopper() const { return getVersionMajor() == 3; }

SmallVector<unsigned> NvidiaMmaEncodingAttr::getRepOrder() const {
  return getMatrixOrder(getRank(), /*rowMajor*/ true);
}

SmallVector<unsigned>
NvidiaMmaEncodingAttr::getRepOrderForOperand(int opIdx) const {
  return getOrderForDotOperand(opIdx, getRank(), /*kContig*/ true);
}

SmallVector<int64_t>
NvidiaMmaEncodingAttr::getRepForOperand(ArrayRef<int64_t> shape, int bitwidth,
                                        int kWidth, int opIdx) const {
  assert(kWidth >= std::max(32 / bitwidth, 1) &&
         "kWidth must be >= max(32 / bitwidth, 1) for this function to be "
         "well-defined");
  auto rank = shape.size();
  // Broadcast long K
  auto warpsPerCTA = to_vector(getWarpsPerCTA());
  auto kDim = opIdx == 0 ? rank - 1 : rank - 2;
  warpsPerCTA[kDim] = 1;

  SmallVector<int> tileSize;
  if (rank == 3) {
    tileSize.push_back(1);
  }
  // warpSizeK * (warpRepK * VecBitWidth)
  auto tileBitWidthK = (isAmpere() && bitwidth == 64) ? (4 * 256) : (4 * 64);
  if (opIdx == 0) {
    // m x k
    tileSize.push_back(16);
    tileSize.push_back(tileBitWidthK / bitwidth);
  } else {
    // k x n
    // Hopper path never uses the n value, since this method is only invoked
    // for in-RF (dotOpEnc) operands, but WGMMA only supports in A to be in RF
    // so it's fine if the n is incorrect here
    tileSize.push_back(tileBitWidthK / bitwidth);
    tileSize.push_back(8);
  }

  SmallVector<int64_t> numRep;
  // Lezcano: This is odd. Why do we always return a vector of size 3?
  if (rank != 3) {
    numRep.push_back(1);
  }
  for (auto [s, size, warp] : llvm::zip(shape, tileSize, warpsPerCTA)) {
    numRep.push_back(std::max<int64_t>(1, s / (size * warp)));
  }
  return numRep;
}

//===----------------------------------------------------------------------===//
// DotOperand Encoding
//===----------------------------------------------------------------------===//

SmallVector<unsigned> DotOperandEncodingAttr::getRepOrder() const {
  if (auto mma = mlir::dyn_cast<MmaEncodingTrait>(getParent())) {
    return mma.getRepOrderForOperand(getOpIdx());
  } else if (auto blocked = mlir::dyn_cast<BlockedEncodingAttr>(getParent())) {
    return to_vector(blocked.getOrder());
  }
  llvm::report_fatal_error(
      "getRepOrder not implemented for DotOperandEncodingAttr");
  return {};
}

SmallVector<unsigned> DotOperandEncodingAttr::getCTAsPerCGA() const {
  return ::getCTAsPerCGA(getParent());
}

SmallVector<unsigned> DotOperandEncodingAttr::getCTAOrder() const {
  return ::getCTAOrder(getParent());
}

SmallVector<unsigned> DotOperandEncodingAttr::getCTASplitNum() const {
  SmallVector<unsigned> res = ::getCTASplitNum(getParent());
  auto rank = res.size();
  assert(rank == 2 || rank == 3 && "Invalid dotLayout");

  // Do not split CTA in K dimension
  auto kDim = getOpIdx() == 0 ? rank - 1 : rank - 2;
  res[kDim] = 1;
  return res;
}

LogicalResult DotOperandEncodingAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    unsigned opIdx, Attribute parent, unsigned kWidth) {
  if (opIdx != 0 && opIdx != 1) {
    return emitError() << "ttg.dot_op opIdx parameter can be 0 or 1, got: "
                       << opIdx;
  }
  if (!parent) {
    return emitError() << "ttg.dot_op parent parameter cannot be null";
  }
  if (auto parentAttr = mlir::dyn_cast<NvidiaMmaEncodingAttr>(parent)) {
    if (kWidth != 0 && !(parentAttr.isAmpere() || parentAttr.isHopper()))
      return emitError() << "ttg.dot_op kWidth parameter can only be "
                            "non-zero for Ampere or Hopper MMA parent";
    if (kWidth == 0 && (parentAttr.isAmpere() || parentAttr.isHopper()))
      return emitError() << "ttg.dot_op kWidth parameter is mandatory for "
                            "Ampere or Hopper MMA parent";
    if (opIdx != 0 && parentAttr.isHopper())
      return emitError()
             << "ttg.dot_op opIdx parameter must be 0 for "
                "Hopper MMA parent, since Hopper WGMMA only allows first "
                "operand to be in registers";
    return success();
  }

  if (auto parentAttr = mlir::dyn_cast<AMDWmmaEncodingAttr>(parent)) {
    if (parentAttr.getVersion() == 1 && (kWidth != 8 && kWidth != 16))
      return emitError()
             << "ttg.dot_op kWidth parameter must be 8/16 for WMMA v1 "
                "(including packed cases for `scaled_dot`)";
    if (parentAttr.getVersion() == 2 && !llvm::is_contained({4, 8, 16}, kWidth))
      return emitError()
             << "ttg.dot_op kWidth parameter must be 4/8/16 for WMMA v2 "
                "(including packed cases for `scaled_dot`)";
    if (parentAttr.getVersion() == 3 && !llvm::is_contained({2, 8, 16}, kWidth))
      return emitError()
             << "ttg.dot_op kWidth parameter must be 2/8/16 for WMMA v3";
    return success();
  }

  if (auto parentAttr = mlir::dyn_cast<AMDMfmaEncodingAttr>(parent)) {
    if (kWidth == 0)
      return emitError() << "ttg.dot_op kWidth parameter is mandatory for "
                            "MFMA parent";
    return success();
  }

  if (auto parentAttr = mlir::dyn_cast<BlockedEncodingAttr>(parent)) {
    if (kWidth != 0)
      return emitError() << "ttg.dot_op kWidth parameter is not supported "
                            "when the parent is a blocked layout";
    return success();
  }

  return emitError() << "ttg.dot_op unexpected parent layout: " << parent;
}

//===----------------------------------------------------------------------===//
// ASM Interface (i.e.: alias)
//===----------------------------------------------------------------------===//

class TritonGPUOpAsmInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    // Encoding attributes
    if (auto mmaAttr = mlir::dyn_cast<MmaEncodingTrait>(attr)) {
      os << "mma";
      return AliasResult::FinalAlias;
    } else if (auto sharedAttr = mlir::dyn_cast<SharedEncodingTrait>(attr)) {
      os << "shared";
      return AliasResult::FinalAlias;
    } else if (auto blockedAttr = mlir::dyn_cast<BlockedEncodingAttr>(attr)) {
      os << "blocked";
      return AliasResult::FinalAlias;
    } else if (auto linearAttr = mlir::dyn_cast<LinearEncodingAttr>(attr)) {
      os << "linear";
      return AliasResult::FinalAlias;
    } /* else if (auto sliceAttr = dyn_cast<SliceEncodingAttr>(attr)) {
      os << "slice";
      return AliasResult::FinalAlias;
    } */
    // Memory space attributes
    if (auto smem = mlir::dyn_cast<SharedMemorySpaceAttr>(attr)) {
      os << "smem";
      return AliasResult::FinalAlias;
    }
    return OpAsmDialectInterface::getAlias(attr, os);
  }
};

struct TritonGPUInferLayoutInterface
    : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding,
                        std::optional<Location> loc) const override {
    resultEncoding =
        SliceEncodingAttr::get(getDialect()->getContext(), axis,
                               cast<DistributedEncodingTrait>(operandEncoding));
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
  LogicalResult
  inferTransOpEncoding(Attribute operandEncoding, ArrayRef<int64_t> shape,
                       ArrayRef<int32_t> order, Attribute &resultEncoding,
                       std::optional<Location> loc) const override {
    // Note: inferFooOpEncoding should not crash if given invalid inputs, which
    // happens when someone creates invalid IR.  If we return failure() on
    // error, then MLIR will generate a helpful error message.
    if (isIota(order)) {
      resultEncoding = operandEncoding;
      return success();
    }
    if (shape.size() != order.size()) {
      return emitOptionalError(loc, "shape and order rank do not match: ",
                               shape.size(), " vs ", order.size());
    }
    auto checkRank = [&](unsigned rank) {
      if (rank != order.size()) {
        return emitOptionalError(loc, "rank of encoding does not match order: ",
                                 rank, " vs ", order.size());
      }
      return success();
    };

    auto *ctx = getDialect()->getContext();
    auto invOrder = inversePermutation(order);
    SmallVector<unsigned> invOrderUnsigned(invOrder.begin(), invOrder.end());

    if (auto enc = dyn_cast<SwizzledSharedEncodingAttr>(operandEncoding)) {
      if (failed(checkRank(enc.getRank())))
        return failure();

      CTALayoutAttr ctaLayout =
          permuteCTALayout(ctx, enc.getCTALayout(), order);
      resultEncoding = SwizzledSharedEncodingAttr::get(
          ctx, enc.getVec(), enc.getPerPhase(), enc.getMaxPhase(),
          applyPermutation(invOrderUnsigned, enc.getOrder()), ctaLayout);
      return success();
    }

    if (auto enc = dyn_cast<NVMMASharedEncodingAttr>(operandEncoding)) {
      if (order == ArrayRef<int32_t>({1, 0})) {
        if (failed(checkRank(enc.getRank())))
          return failure();

        CTALayoutAttr ctaLayout =
            permuteCTALayout(ctx, enc.getCTALayout(), order);
        resultEncoding = NVMMASharedEncodingAttr::get(
            ctx, enc.getSwizzlingByteWidth(), !enc.getTransposed(),
            enc.getElementBitWidth(), enc.getFp4Padded(), ctaLayout);
        return success();
      }
    }

    if (auto enc = dyn_cast<BlockedEncodingAttr>(operandEncoding)) {
      if (failed(checkRank(enc.getRank())))
        return failure();

      CTALayoutAttr ctaLayout =
          permuteCTALayout(ctx, enc.getCTALayout(), order);
      resultEncoding = BlockedEncodingAttr::get(
          ctx, applyPermutation(enc.getSizePerThread(), order),
          applyPermutation(enc.getThreadsPerWarp(), order),
          applyPermutation(enc.getWarpsPerCTA(), order),
          applyPermutation(invOrderUnsigned, enc.getOrder()), ctaLayout);
      return success();
    }
    // Generic case
    auto padded = dyn_cast<PaddedSharedEncodingAttr>(operandEncoding);

    auto ll = padded ? padded.getLinearComponent()
                     : toLinearLayout(shape, operandEncoding);
    if (failed(checkRank(ll.getNumOutDims())))
      return failure();
    auto transposedLl = transposeLinearLayout(ll, order);
    if (isa<DistributedEncodingTrait>(operandEncoding)) {
      resultEncoding = LinearEncodingAttr::get(ctx, std::move(transposedLl));
    } else if (padded) {
      resultEncoding = PaddedSharedEncodingAttr::get(ctx, padded.getIntervals(),
                                                     padded.getPaddings(),
                                                     std::move(transposedLl));
    } else {
      auto shared = cast<SharedEncodingTrait>(operandEncoding);
      resultEncoding = SharedLinearEncodingAttr::get(
          ctx, std::move(transposedLl), shared.getAlignment());
    }
    return success();
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> location) const override {
    auto sliceEncoding = mlir::dyn_cast<SliceEncodingAttr>(operandEncoding);
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
    auto mmaRetEncoding = mlir::dyn_cast<NvidiaMmaEncodingAttr>(retEncoding);
    if (mmaRetEncoding && mmaRetEncoding.isHopper()) {
      auto dotOpEnc = mlir::dyn_cast<DotOperandEncodingAttr>(operandEncoding);
      if (!mlir::isa<NVMMASharedEncodingAttr>(operandEncoding) &&
          !(opIdx == 0 && dotOpEnc && dotOpEnc.getOpIdx() == 0 &&
            mlir::isa<NvidiaMmaEncodingAttr>(dotOpEnc.getParent()))) {
        return emitOptionalError(
            location, "unexpected operand layout for NvidiaMmaEncodingAttr v3");
      }
    } else if (auto dotOpEnc =
                   mlir::dyn_cast<DotOperandEncodingAttr>(operandEncoding)) {
      if (opIdx != dotOpEnc.getOpIdx())
        return emitOptionalError(location, "Wrong opIdx");
      if (retEncoding != dotOpEnc.getParent())
        return emitOptionalError(location, "Incompatible parent encoding");
    } else
      return emitOptionalError(
          location, "Dot's a/b's encoding should be of DotOperandEncodingAttr");
    return success();
  }

  LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const override {
    auto aEncoding =
        mlir::dyn_cast<triton::gpu::DotOperandEncodingAttr>(operandEncodingA);
    auto bEncoding =
        mlir::dyn_cast<triton::gpu::DotOperandEncodingAttr>(operandEncodingB);
    if (!aEncoding && !bEncoding)
      return mlir::success();
    auto mmaAEncoding =
        mlir::dyn_cast_or_null<NvidiaMmaEncodingAttr>(aEncoding.getParent());
    if (mmaAEncoding && mmaAEncoding.isHopper())
      return success();
    // Verify that the encodings are valid.
    if (!aEncoding || !bEncoding)
      return op->emitError("mismatching encoding between A and B operands");
    if (aEncoding.getKWidth() != bEncoding.getKWidth())
      return op->emitError("mismatching kWidth between A and B operands");
    return success();
  }

  // Given a src shape + encoding and a dst shape, our goal is to compute a dst
  // encoding that makes the reshape a "nop".  That is, if GPU thread [x,y,z]
  // contains elements [a,b,c,d] before the reshape, it contains those same
  // elements after the reshape, they're just "renamed".
  //
  // Using legacy layouts, a dst encoding that satisfies this property may not
  // exist.  Here are some positive and negative examples.
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
  // With linear layouts, we can always find a dst encoding that satisfies
  // this property. See inferReshapeOpEncoding.
  //
  // Users of this function require that it is symmetrical: if
  // (srcShape,srcEnc,dstShape) => dstEnc, then (dstShape,dstEnc,srcShape) =>
  // srcEnc.
  LogicalResult inferReshapeOpLegacyEncoding(ArrayRef<int64_t> srcShape,
                                             Attribute srcEnc,
                                             ArrayRef<int64_t> dstShape,
                                             Attribute &dstEnc) const {
    auto src = mlir::dyn_cast<BlockedEncodingAttr>(srcEnc);
    if (!src) {
      return failure();
    }

    // Nop reshape; we can always infer an encoding.
    if (srcShape == dstShape) {
      dstEnc = srcEnc;
      return success();
    }

    // default -> default encoding is always a nop.
    auto context = srcEnc.getContext();
    int32_t numWarps = product(src.getWarpsPerCTA());
    int32_t threadsPerWarp = product(src.getThreadsPerWarp());
    int32_t numCTAs = product(src.getCTALayout().getCTAsPerCGA());
    if (srcEnc == getDefaultBlockedEncoding(context, srcShape, numWarps,
                                            threadsPerWarp, numCTAs)) {
      dstEnc = getDefaultBlockedEncoding(context, dstShape, numWarps,
                                         threadsPerWarp, numCTAs);
      return success();
    }

    // Cowardly refuse to handle encodings with multiple CTAs.  CTAsPerCGA
    // should be like the other fields in blocked encoding, but I'm not sure how
    // to handle CTASplitNum.
    if (!all_of(src.getCTAsPerCGA(), [](int32_t x) { return x == 1; }) ||
        !all_of(src.getCTASplitNum(), [](int32_t x) { return x == 1; })) {
      return failure();
    }

    // Cowardly refuse to handle encodings where shape[dim] is not divisible by
    // sizePerThread[dim], threadsPerWarp[dim], and warpsPerCTA[dim].  (We make
    // an exception if the block is larger than the shape.)
    auto checkDivisibility = [&](StringRef name, ArrayRef<unsigned> subblock) {
      for (int dim = 0; dim < srcShape.size(); dim++) {
        if (srcShape[dim] >= subblock[dim] &&
            srcShape[dim] % subblock[dim] != 0) {
          return failure();
        }
      }
      return success();
    };
    if (!succeeded(
            checkDivisibility("sizePerThread", src.getSizePerThread())) ||
        !succeeded(
            checkDivisibility("threadsPerWarp", src.getThreadsPerWarp())) ||
        !succeeded(checkDivisibility("warpsPerCTA", src.getWarpsPerCTA()))) {
      return failure();
    }

    SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> decomp =
        getReshapeDecomposition(srcShape, dstShape);

    // enc.order[i] == j means that dimension j is the enc.order[i]'th most
    // minor. But what we usually want is the inverse: inverse(enc.order)[i] = j
    // means that dimension i is the j'th most minor (larger means more major).
    auto srcInvOrder = inversePermutation(src.getOrder());

    // If src dims [a,b,c] are to be merged, then they must be consecutive in
    // physical order, with `a` being the most major.
    for (const auto &[srcDims, dstDims] : decomp) {
      if (!isConsecutive(to_vector(reverse(gather(srcInvOrder, srcDims))))) {
        return failure();
      }
    }

    // If src dims [a,b,c] are to be merged, then `c` must fill up sizePerThread
    // / threadsPerWarp / blocksPerCTA before `b` can have any non-1 values.
    // Examples:
    //
    //  - NOT OK: shape=[4,4,4], sizePerThread=[1,2,2].
    //    The total sizePerThread for dim 2 is 2, which is less than dim 2's
    //    size of 4.  Therefore dim 1 cannot have non-1 sizePerThread.
    //
    //  - OK: shape=[4,4,4], sizePerThread=[1,2,4].
    //    Dim 2's sizePerThread covers its whole size, so dim 1 is allowed to
    //    have non-1 sizePerThread.
    //
    //  - NOT OK: shape=[4,4,4], sizePerThread=[2,1,4].
    //    Dim 1's sizePerThread does not cover its whole size, so dim 0 is not
    //    allowed to have non-1 sizePerThread.
    //
    //  - NOT OK: shape=[4,4,4], sizePerThread=[1,1,2],
    //            threadsPerWarp=[1,2,1].
    //    Dim 2 has 2 elems per thread and 1 thread per warp.  2*1 is less than
    //    dim 2's size.  Therefore dim 1 must have threadsPerWarp=1.
    //
    // In addition, the encoding's block can be larger than the shape, but only
    // in the most-major dimension of each decomposed chunk, and only after
    // we've "used up" the more minor dims.  Examples:
    //
    //  - OK: shape=[4,4,4], sizePerThread=[1,2,4], threadsPerWarp=[16,2,1],
    //        warpsPerCTA=[4,1,1].
    //    The whole size of dims 0 and 1 are covered by sizePerThread *
    //    threadsPerWarp.  Therefore dim 2 is allowed to have threadsPerWarp and
    //    warpsPerCTA larger than its size.
    for (const auto &[srcDims, dstDims] : decomp) {
      auto shapeRemaining = gather(srcShape, srcDims);
      auto checkSubblock = [&, srcDims = srcDims](ArrayRef<unsigned> subblock) {
        // Iterate minor-to-major (i==0 is most major).
        for (int i = srcDims.size() - 1; i >= 0; i--) {
          int dim = srcDims[i];
          if (subblock[dim] == 1) {
            continue;
          }

          // Check that more-minor dims all have 1 in shapeRemaining.
          for (int j = i + 1; j < srcDims.size(); j++) {
            if (shapeRemaining[j] != 1) {
              return failure();
            }
          }

          if (shapeRemaining[i] >= subblock[dim]) {
            assert(shapeRemaining[i] % subblock[dim] == 0); // checked earlier
            shapeRemaining[i] /= subblock[dim];
          } else {
            shapeRemaining[i] = 0;
          }

          // Is the block larger than the shape in this dimension?  This is OK
          // only if we're the most-major dimension of the chunk and in all
          // future chunks, only this most-major dim has a non-1 size.
          if (shapeRemaining[i] == 0 && i != 0) {
            return failure();
          }
        }
        return success();
      };
      if (!succeeded(checkSubblock(src.getSizePerThread())) ||
          !succeeded(checkSubblock(src.getThreadsPerWarp())) ||
          !succeeded(checkSubblock(src.getWarpsPerCTA()))) {
        return failure();
      }
    }

    // Given e.g. src.getSizePerThread(), computeSubblockSize computes e.g.
    // dst.getSizePerThread().  This should be called for each of sizePerThread,
    // threadsPerWarp, and warpsPerCTA, in that order.
    SmallVector<int64_t> dstShapeRemaining(dstShape);
    auto computeSubblockSize = [&](ArrayRef<unsigned> srcSubblock,
                                   SmallVector<unsigned> &dstSubblock,
                                   StringRef fieldName) -> LogicalResult {
      // The dst subblock is "filled up" greedily starting with the most minor
      // dim.  When we're done, we are left with a smaller shape, of size
      // dstShape / dstSubblock, which we store in dstShapeRemaining and use for
      // the next call to computeSubblockSize.
      dstSubblock.resize(dstShape.size());
      for (const auto &[srcDims, dstDims] : decomp) {
        int64_t subblockRemaining = product(gather(srcSubblock, srcDims));
        for (int i = dstDims.size() - 1; i >= 0; i--) {
          auto &val = dstSubblock[dstDims[i]];
          auto &shapeRemaining = dstShapeRemaining[dstDims[i]];
          val = std::min(subblockRemaining, shapeRemaining);

          assert(shapeRemaining % val == 0); // Checked earlier.
          subblockRemaining /= val;
          shapeRemaining /= val;
        }

        // If there are any elems remaining in the subblock, it must be because
        // the block is larger than the shape.  This excess goes into the
        // most-major dim of the subblock.
        dstSubblock[dstDims[0]] *= subblockRemaining;
      }
      return success();
    };

    SmallVector<unsigned> dstSizePerThread;
    SmallVector<unsigned> dstThreadsPerWarp;
    SmallVector<unsigned> dstWarpsPerCTA;
    if (!succeeded(computeSubblockSize(src.getSizePerThread(), dstSizePerThread,
                                       "sizePerThread")) ||
        !succeeded(computeSubblockSize(src.getThreadsPerWarp(),
                                       dstThreadsPerWarp, "threadsPerWarp")) ||
        !succeeded(computeSubblockSize(src.getWarpsPerCTA(), dstWarpsPerCTA,
                                       "warpsPerCTA"))) {
      return failure();
    }

    // Since we know that each set of srcDims is consecutive, we can
    // meaningfully sort decomp by the physical order of the src dimensions,
    // major-to-minor.  This will also be the order of the dst dimensions.
    llvm::sort(decomp, [&](const auto &a, const auto &b) {
      const auto &[srcDimsA, dstDimsA] = a;
      const auto &[srcDimsB, dstDimsB] = b;
      return srcInvOrder[srcDimsA.front()] < srcInvOrder[srcDimsB.front()];
    });

    // Compute the dst order.  Make the dimensions appear in the same order as
    // their corresponding src dimensions.
    SmallVector<unsigned> dstInvOrder(dstShape.size());
    int i = 0;
    for (const auto &[srcDims, dstDims] : decomp) {
      for (auto dim : reverse(dstDims)) {
        dstInvOrder[dim] = i++;
      }
    }
    auto dstOrder = inversePermutation(dstInvOrder);

    // CTALayout can be all 1's because we bailed on multi-CTA layouts above.
    auto CTALayout = CTALayoutAttr::get(
        src.getContext(),
        /*CTAsPerCGA=*/SmallVector<unsigned>(dstShape.size(), 1),
        /*CTASplitNum=*/SmallVector<unsigned>(dstShape.size(), 1),
        /*CTAOrder=*/llvm::to_vector(llvm::seq<unsigned>(dstShape.size())));

    dstEnc = BlockedEncodingAttr::get(src.getContext(), dstSizePerThread,
                                      dstThreadsPerWarp, dstWarpsPerCTA,
                                      dstOrder, CTALayout);

    return success();
  }

  LogicalResult
  verifyLayoutsAreEqual(ArrayRef<int64_t> shape, Attribute expected,
                        Attribute got,
                        std::optional<Location> loc) const override {
    if (expected == got) {
      return success();
    }
    if (!expected || !got)
      return failure();

    // Check whether the encodings are structurally the same.
    if (!areLayoutsEquivalent(shape, cast<LayoutEncodingTrait>(expected),
                              cast<LayoutEncodingTrait>(got))) {
      return emitOptionalError(loc, "Expected result encoding ", expected,
                               " but was ", got);
    }
    return success();
  }

  LogicalResult
  inferReshapeOpEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                         ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                         std::optional<Location> loc) const override {
    if (product(srcShape) != product(dstShape)) {
      return emitOptionalError(loc, "numel of dst shape does not match "
                                    "numel of src shape");
    }
    auto result =
        inferReshapeOpLegacyEncoding(srcShape, srcEnc, dstShape, dstEnc);
    if (succeeded(result)) {
      return result;
    }
    if (!isa<DistributedEncodingTrait>(srcEnc)) {
      return emitOptionalError(loc,
                               "Failed MemDescReshapeOp encoding inference");
    }
    // If the legacy encoding failed use LinearLayouts.
    // Once LinearLayouts are more widely used, we can remove
    // inferReshapeOpLegacyEncoding and simply use LLs.

    // HACK: We create a dummy tensor type to pass to inferReshapeLinearLayout.
    auto ctx = srcEnc.getContext();
    auto fp32Type = IntegerType::get(ctx, 32, IntegerType::Unsigned);
    auto srcTy = RankedTensorType::get(srcShape, fp32Type, srcEnc);
    LinearLayout ll =
        inferReshapeLinearLayout(cast<TensorOrMemDesc>(srcTy), dstShape);

    dstEnc = LinearEncodingAttr::get(srcEnc.getContext(), ll);
    return success();
  }

  LogicalResult
  inferDefaultJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                             ArrayRef<int64_t> shape,
                             std::optional<Location> loc) const override {
    auto ctx = getContext();
    if (auto enc = mlir::dyn_cast<SliceEncodingAttr>(srcEnc);
        enc && enc.getDim() == shape.size()) {
      SmallVector<int64_t> joinedShape(shape);
      joinedShape.push_back(2);
      auto parent = enc.getParent();
      auto parentLL = toLinearLayout(joinedShape, parent);

      Attribute splitEnc;
      auto result = inferSplitOpEncoding(parent, splitEnc, joinedShape, loc);
      if (succeeded(result) &&
          areLayoutsEquivalent(shape, cast<LayoutEncodingTrait>(splitEnc),
                               cast<LayoutEncodingTrait>(srcEnc))) {
        dstEnc = parent;
        return success();
      }
    } else if (auto enc = mlir::dyn_cast<BlockedEncodingAttr>(srcEnc)) {
      // JoinOp takes two tensors of shape AxBxC and generates a tensor of shape
      // AxBxCx2. The encoding is the same as the input, but with 2 elems per
      // thread in the new dimension. The new dimension is the fastest running
      // dimension.
      auto append = [](ArrayRef<unsigned> vals, int val) {
        SmallVector<unsigned> ret(vals);
        ret.push_back(val);
        return ret;
      };
      auto appendMajorDim = [](ArrayRef<unsigned> order) {
        SmallVector<unsigned> ret(order);
        ret.insert(ret.begin(), ret.size());
        return ret;
      };
      dstEnc = BlockedEncodingAttr::get(
          enc.getContext(), append(enc.getSizePerThread(), 2),
          append(enc.getThreadsPerWarp(), 1), append(enc.getWarpsPerCTA(), 1),
          appendMajorDim(enc.getOrder()),
          CTALayoutAttr::get(enc.getContext(), append(enc.getCTAsPerCGA(), 1),
                             append(enc.getCTASplitNum(), 1),
                             appendMajorDim(enc.getCTAOrder())));
      return success();
    }

    // Append dim to shape
    auto ll = toLinearLayout(shape, srcEnc);
    SmallVector<int64_t> dstShape(shape.begin(), shape.end());
    dstShape.push_back(1);
    ll = ll.reshapeOuts(standardOutDimPairs(ctx, dstShape));

    // Try join on last dim
    auto axis = dstShape.size() - 1;
    auto newLl = LinearLayout::empty();
    auto result =
        tryJoinOnAxis(ctx, ll, newLl, /*fwdInference=*/true, axis, loc);

    assert(result.succeeded());
    dstEnc = LinearEncodingAttr::get(ctx, newLl);
    return success();
  }

  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       ArrayRef<int64_t> shape,
                       std::optional<Location> loc) const override {
    // SplitOp takes a tensor of shape AxBxCx2 and generates two tensors of
    // shape AxBxC.  The input must have 2 elements per thread in the last
    // dimension, which must be the fastest running dimension. The result
    // encoding is the same as the input, but with the last dimension removed.
    auto enc = mlir::dyn_cast<BlockedEncodingAttr>(srcEnc);
    bool isSimpleSplit = (enc && (enc.getSizePerThread().back() == 2) &&
                          (enc.getThreadsPerWarp().back() == 1) &&
                          (enc.getWarpsPerCTA().back() == 1) &&
                          (enc.getCTAsPerCGA().back() == 1));
    if (isSimpleSplit) {
      SmallVector<unsigned> newOrder(enc.getOrder());
      int splitDim = newOrder.size() - 1;
      // Remove splitDim from order.
      newOrder.erase(std::remove(newOrder.begin(), newOrder.end(), splitDim),
                     newOrder.end());
      dstEnc = BlockedEncodingAttr::get(
          enc.getContext(), //
          ArrayRef(enc.getSizePerThread()).drop_back(1),
          ArrayRef(enc.getThreadsPerWarp()).drop_back(1),
          ArrayRef(enc.getWarpsPerCTA()).drop_back(1), ArrayRef(newOrder),
          CTALayoutAttr::get(enc.getContext(), //
                             ArrayRef(enc.getCTAsPerCGA()).drop_back(1),
                             ArrayRef(enc.getCTASplitNum()).drop_back(1),
                             ArrayRef(enc.getCTAOrder()).drop_front(1)));
      return success();
    }

    auto axis = shape.size() - 1;
    if (shape[axis] != 2) {
      return emitOptionalError(
          loc, "SplitOp input shape should have 2 in the last dim");
    }

    auto ctx = getContext();

    // Split on last dim
    auto ll = toLinearLayout(shape, srcEnc);
    auto newLl = LinearLayout::empty();
    auto result =
        tryJoinOnAxis(ctx, ll, newLl, /*fwdInference=*/false, axis, loc);
    if (!result.succeeded()) {
      return failure();
    }
    // Remove last dim from newLl (which should be 1)
    SmallVector<int64_t> dstShape(shape.begin(), shape.end());
    dstShape.pop_back();
    newLl = newLl.reshapeOuts(standardOutDimPairs(ctx, dstShape));
    dstEnc = LinearEncodingAttr::get(ctx, newLl);
    return success();
  }

  LogicalResult
  inferFp4ToFpOpEncoding(ArrayRef<int64_t> shape, int axis, Attribute inEnc,
                         Attribute &outEnc, bool fwdInference,
                         std::optional<Location> loc) const override {
    // We implement two legacy layout propagations
    // Once we fully migrate to LinearLayouts, we can remove these.
    auto *ctx = getContext();
    // The output encoding will only be a legacy encoding if the axis is the
    // fastest running dimension.
    // FIXME: We should make sure that there are enough elements along the axis
    // axis whenever fwdInference is false
    if (getOrder(cast<DistributedEncodingTrait>(inEnc), shape)[axis] == 0) {
      // Dot operand: double kWidth if kDim == axis.
      if (auto dotEnc = mlir::dyn_cast<DotOperandEncodingAttr>(inEnc)) {
        auto kWidth = dotEnc.getKWidth();
        if (fwdInference) {
          kWidth *= 2;
        } else {
          if (kWidth > 1) {
            // bwd inference
            kWidth /= 2;
          } else {
            return emitOptionalError(loc,
                                     "Fp4ToFpOp requires at least 2 elements "
                                     "per thread in the axis dimension");
          }
        }
        outEnc = DotOperandEncodingAttr::get(ctx, dotEnc.getOpIdx(),
                                             dotEnc.getParent(), kWidth);
        return success();
      }

      // Blocked layout: double elemsPerThread[axis].
      if (auto blockedEnc = mlir::dyn_cast<BlockedEncodingAttr>(inEnc)) {
        auto sizePerThread = llvm::to_vector(blockedEnc.getSizePerThread());
        if (fwdInference) {
          sizePerThread[axis] *= 2;
        } else {
          if (sizePerThread[axis] > 1) {
            sizePerThread[axis] /= 2;
          } else {
            return emitOptionalError(
                loc, "Fp4ToFpOp requires at least 2 elements per "
                     "thread in the axis dimension");
          }
        }
        outEnc = BlockedEncodingAttr::get(
            ctx, sizePerThread, blockedEnc.getThreadsPerWarp(),
            blockedEnc.getWarpsPerCTA(), blockedEnc.getOrder(),
            blockedEnc.getCTALayout());
        return success();
      }
    }

    auto ll = toLinearLayout(shape, inEnc);
    auto newLl = LinearLayout::empty();
    auto result = tryJoinOnAxis(ctx, ll, newLl, fwdInference, axis, loc);
    if (!result.succeeded())
      return result;
    outEnc = LinearEncodingAttr::get(ctx, newLl);
    return success();
  }
};

struct TritonGPUVerifyTensorLayoutInterface
    : public triton::DialectVerifyTensorLayoutInterface {
  using DialectVerifyTensorLayoutInterface::DialectVerifyTensorLayoutInterface;

  LogicalResult verifyTensorLayout(
      Attribute layout, RankedTensorType rankedTy, Operation *op,
      function_ref<InFlightDiagnostic()> makeErr) const override {
    auto distr = dyn_cast<triton::gpu::DistributedEncodingTrait>(layout);
    if (!distr)
      return makeErr()
             << "Non-distributed layout is not allowed in tensor type.";
    auto rank = distr.getRepOrder().size();
    if (rank != rankedTy.getRank())
      return makeErr() << "Layout has rank " << rank
                       << ", but the tensor it's attached to has rank "
                       << rankedTy.getRank() << ".";
    if (llvm::any_of(rankedTy.getShape(),
                     [](int64_t i) { return !llvm::isPowerOf2_64(i); })) {
      return makeErr() << "Layout has shape " << rankedTy.getShape()
                       << ", but the tensor it's attached to has shape "
                       << rankedTy.getShape()
                       << " which is not a power of two.";
    }
    auto ll = toLinearLayout(rankedTy);
    ModuleOp module = op->getParentOfType<ModuleOp>();

    // Number of threads per warp.
    auto kLane = StringAttr::get(module.getContext(), "lane");
    int moduleThreadsPerWarp = TritonGPUDialect::getThreadsPerWarp(module);
    if (ll.getInDimSize(kLane) != moduleThreadsPerWarp) {
      return makeErr() << layout << ".\nLayout has " << ll.getInDimSize(kLane)
                       << " threads per warp, but the module specifies "
                       << moduleThreadsPerWarp << " threads per warp.";
    }

    // Number of warps per CTA.
    std::optional<int> moduleWarpsPerCTA = maybeLookupNumWarps(op);
    if (!moduleWarpsPerCTA) {
      return makeErr()
             << "Could not determine the number of warps per CTA. Operation "
                "is not in a context with `ttg.num-warps`.";
    }
    auto kWarp = StringAttr::get(module.getContext(), "warp");
    if (ll.getInDimSize(kWarp) != *moduleWarpsPerCTA) {
      return makeErr() << layout << ".\nLayout has " << ll.getInDimSize(kWarp)
                       << " warps per CTA, but the context requires "
                       << *moduleWarpsPerCTA << " warps per CTA.";
    }

    // Number of CTAs per CGA.
    auto kBlock = StringAttr::get(module.getContext(), "block");
    int moduleCTAsPerCGA = TritonGPUDialect::getNumCTAs(module);
    if (ll.getInDimSize(kBlock) != moduleCTAsPerCGA) {
      return makeErr() << layout << ".\nLayout has " << ll.getInDimSize(kBlock)
                       << " CTAs per CGA, but the context requires "
                       << moduleCTAsPerCGA << " CTAs per CGA.";
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Layout debug printing
//===----------------------------------------------------------------------===//

// Return N-D delinearized indices from a linear index.
static SmallVector<int64_t> delinearizeIndex(int64_t idx,
                                             ArrayRef<int64_t> shape) {
  SmallVector<int64_t> ret(shape.size());
  for (int i = shape.size() - 1; i >= 0; i--) {
    ret[i] = idx % shape[i];
    idx /= shape[i];
  }
  return ret;
}

// Returns how many padding characters are needed for the string representation
// of value to be the same as max.
static int numCharacterPadding(int value, int max) {
  return std::to_string(max).size() - std::to_string(value).size();
}

// return the string padded to have the same length as max.
static std::string paddedString(int value, int max) {
  int nbChar = numCharacterPadding(value, max);
  std::string str;
  for (int i = 0; i < nbChar; i++)
    str += " ";
  str += std::to_string(value);
  return str;
}

std::string getSharedLayoutStr(RankedTensorType type, bool useHWPointOfView) {
  if (!type)
    return "";

  // This RankedTensorType is a MemDescType (?!)
  auto shape = type.getShape();
  auto layout = type.getEncoding();
  LinearLayout ll = triton::gpu::toLinearLayout(shape, layout);

  StringAttr kOffset = StringAttr::get(type.getContext(), "offset");
  StringAttr kBlock = StringAttr::get(type.getContext(), "block");
  int64_t tensorSize = product(type.getShape());
  auto enc = type.getEncoding();
  unsigned numBlocks = getNumCTAs(enc);
  int32_t blockSize = tensorSize / numBlocks;

  // elementMapping is for the non-hw layout, offsetMapping for hw-layout
  std::vector<std::string> elementMapping(tensorSize);
  std::vector<std::string> offsetMapping;

  // Shared layouts are a mapping of (block, offset) --> (...)

  // We can just use a single int to index into elementMapping because
  // the 'swizzle' operation rearranges the indices---and we want to keep it
  // that way
  int32_t idx = 0;
  // Enumerate all the offsets for each block
  for (int32_t block = 0; block < numBlocks; block++) {
    for (int32_t offset = 0; offset < blockSize; offset++) {
      SmallVector<std::pair<StringAttr, int32_t>> inputs = {
          {kBlock, block},
          {kOffset, offset},
      };

      SmallVector<std::pair<StringAttr, int32_t>> outputs = ll.apply(inputs);

      std::string sharedInfo = "(";
      std::string &value = elementMapping[idx];

      if (!value.empty())
        value += "|";

      value += "(";
      // We can build up both strings (for hw/non-hw layouts) concurrently
      for (int i = 0; i < outputs.size(); i++) {
        // Based on the formatting from LinearLayout::toString, the format for
        // the hw layout is slightly different. HW layouts use "," vs ":".
        if (i > 0) {
          sharedInfo += ",";
          value += ":";
        }
        auto index = paddedString(outputs[i].second, shape[i]);
        sharedInfo += index;
        value += index;
      }
      value += ")";
      sharedInfo += ")";

      offsetMapping.push_back(sharedInfo);

      idx++;
    }
  }

  std::string layoutStr;

  if (!useHWPointOfView) {
    int rank = type.getRank();
    bool newLine = true;
    for (int i = 0; i < tensorSize; i++) {
      auto indices = delinearizeIndex(i, shape);
      int numOpenBracket = 0;
      for (int j = rank - 1; j >= 0; j--) {
        if (indices[j] % shape[j] != 0)
          break;
        layoutStr += "[";
        numOpenBracket++;
      }
      if (newLine) {
        for (int j = 0; j < rank - numOpenBracket; j++)
          layoutStr += " ";
        newLine = false;
      }

      layoutStr += elementMapping[i];
      auto nextIndices = delinearizeIndex(i + 1, shape);
      for (int j = rank - 1; j >= 0; j--) {
        if (nextIndices[j] % shape[j] != 0)
          break;
        layoutStr += "]";
      }
      if (nextIndices.back() % shape.back() == 0) {
        layoutStr += "\n";
        newLine = true;
      } else {
        layoutStr += ",";
      }
    }
  } else {
    // For the HW view here, print the (block, offset) --> (r,c) mapping
    uint32_t idx = 0;
    for (int32_t block = 0; block < numBlocks; block++) {
      layoutStr += "Block: " + std::to_string(block) + ":\n";
      for (int32_t offset = 0; offset < (tensorSize / numBlocks); offset++) {
        layoutStr += "Offset: " + std::to_string(offset) + " -> ";
        layoutStr += offsetMapping[idx];
        layoutStr += "\n";
        idx++;
      }
    }
  }

  return layoutStr;
}

std::string getDistributedLayoutStr(RankedTensorType tensorType,
                                    bool useHWPointOfView) {
  auto layout = tensorType.getEncoding();
  if (!layout)
    return "";

  StringAttr kRegister = StringAttr::get(tensorType.getContext(), "register");
  StringAttr kLane = StringAttr::get(tensorType.getContext(), "lane");
  StringAttr kWarp = StringAttr::get(tensorType.getContext(), "warp");
  StringAttr kBlock = StringAttr::get(tensorType.getContext(), "block");

  LinearLayout ll = toLinearLayout(tensorType);
  int64_t tensorSize = product(tensorType.getShape());
  std::vector<std::string> elementMapping(tensorSize);
  std::vector<std::string> threadMapping;
  unsigned threadsPerWarp = ll.getInDimSize(kLane);
  unsigned numWarpsPerCTA = ll.getInDimSize(kWarp);
  unsigned numBlocks = ll.getInDimSize(kBlock);
  int numElementsPerThreads = ll.getInDimSize(kRegister);
  for (int blockId = 0; blockId < numBlocks; ++blockId) {
    for (int warpId = 0; warpId < numWarpsPerCTA; warpId++) {
      for (int tid = 0; tid < threadsPerWarp; ++tid) {
        for (int idx = 0; idx < numElementsPerThreads; ++idx) {
          SmallVector<std::pair<StringAttr, int32_t>> inputs = {
              {kBlock, blockId},
              {kWarp, warpId},
              {kLane, tid},
              {kRegister, idx}};
          SmallVector<std::pair<StringAttr, int32_t>> outputs =
              ll.apply(inputs);
          int32_t linearizedIdx = 0;
          int stride = 1;
          for (int i = outputs.size() - 1; i >= 0; i--) {
            linearizedIdx += outputs[i].second * stride;
            stride *= tensorType.getDimSize(i);
          }
          std::string &value = elementMapping[linearizedIdx];
          if (!value.empty())
            value += "|";
          int padding = numCharacterPadding(blockId, numBlocks) +
                        numCharacterPadding(tid + warpId * threadsPerWarp,
                                            numWarpsPerCTA * threadsPerWarp) +
                        numCharacterPadding(idx, numElementsPerThreads);
          for (int i = 0; i < padding; i++)
            value += " ";
          if (numBlocks > 1)
            value += "B" + std::to_string(blockId) + ":";
          value += "T" + std::to_string(tid + warpId * threadsPerWarp) + ":" +
                   std::to_string(idx);
          // Now also compute the thread mapping.
          std::string threadInfo = "(";
          for (int i = 0; i < outputs.size(); i++) {
            if (i > 0)
              threadInfo += ",";
            threadInfo +=
                paddedString(outputs[i].second, tensorType.getDimSize(i));
          }
          threadInfo += ")";
          threadMapping.push_back(threadInfo);
        }
      }
    }
  }
  std::string layoutStr;
  if (!useHWPointOfView) {
    // Printing the threads containing each elements of the tensor.
    int rank = tensorType.getRank();
    bool newLine = true;
    for (int i = 0; i < tensorSize; i++) {
      auto indices = delinearizeIndex(i, tensorType.getShape());
      int numOpenBracket = 0;
      for (int j = rank - 1; j >= 0; j--) {
        if (indices[j] % tensorType.getDimSize(j) != 0)
          break;
        layoutStr += "[";
        numOpenBracket++;
      }
      if (newLine) {
        for (int j = 0; j < rank - numOpenBracket; j++)
          layoutStr += " ";
        newLine = false;
      }

      layoutStr += elementMapping[i];
      auto nextIndices = delinearizeIndex(i + 1, tensorType.getShape());
      for (int j = rank - 1; j >= 0; j--) {
        if (nextIndices[j] % tensorType.getDimSize(j) != 0)
          break;
        layoutStr += "]";
      }
      if (nextIndices.back() % tensorType.getShape().back() == 0) {
        layoutStr += "\n";
        newLine = true;
      } else {
        layoutStr += ", ";
      }
    }
  } else {
    // Printing the elements in each physical reg/warps/threads.
    for (int blockId = 0; blockId < numBlocks; blockId++) {
      if (numBlocks > 1)
        layoutStr += "Block" + std::to_string(blockId) + ":\n";
      for (int warpId = 0; warpId < numWarpsPerCTA; warpId++) {
        layoutStr += "Warp" + std::to_string(warpId) + ":\n";
        for (int idx = 0; idx < numElementsPerThreads; ++idx) {
          for (int tid = 0; tid < threadsPerWarp; ++tid) {
            int linearizedIdx =
                blockId * numWarpsPerCTA * threadsPerWarp *
                    numElementsPerThreads +
                warpId * threadsPerWarp * numElementsPerThreads +
                tid * numElementsPerThreads + idx;
            layoutStr += threadMapping[linearizedIdx];
            if (tid < threadsPerWarp - 1)
              layoutStr += ", ";
          }
          layoutStr += "\n";
        }
      }
    }
  }
  return layoutStr;
}

template <typename T>
llvm::SmallVector<T>
mlir::triton::gpu::expandMatrixShapeWithBatch(llvm::ArrayRef<T> s) {
  auto rank = s.size();
  assert(rank == 2 || rank == 3);
  if (rank == 3)
    return llvm::SmallVector<T>(s);
  return {1, s[0], s[1]};
}

template llvm::SmallVector<int64_t>
mlir::triton::gpu::expandMatrixShapeWithBatch<int64_t>(
    llvm::ArrayRef<int64_t> s);

template llvm::SmallVector<unsigned>
mlir::triton::gpu::expandMatrixShapeWithBatch<unsigned>(
    llvm::ArrayRef<unsigned> s);

llvm::SmallVector<unsigned>
mlir::triton::gpu::expandMatrixOrderWithBatch(llvm::ArrayRef<unsigned> o) {
  int rank = o.size();
  assert(rank == 2 || rank == 3);
  if (rank == 3)
    return llvm::SmallVector<unsigned>(o);
  llvm::SmallVector<unsigned> expanded(3, 0);
  for (int i = 0; i < rank; ++i)
    expanded[i] += o[i] + 1;
  return expanded;
}

std::string mlir::triton::gpu::getLayoutStr(RankedTensorType tensorType,
                                            bool useHWPointOfView) {
  auto layout = tensorType.getEncoding();

  // tensorType is needed later on (e.g., getDimSize(j)), so we still have to
  // pass it as a param
  // TODO: Pass TensorOrMemDesc instead of RankedTensorType in
  // triton-tensor-layout.cpp
  if (mlir::isa<SharedEncodingTrait>(layout)) {
    return getSharedLayoutStr(tensorType, useHWPointOfView);
  } else if (mlir::isa<DistributedEncodingTrait>(layout)) {
    return getDistributedLayoutStr(tensorType, useHWPointOfView);
  }

  // else unimplemented, return error
  llvm::report_fatal_error("Unimplemented usage of getLayoutStr");
  return "";
}

void mlir::triton::gpu::dumpLayout(RankedTensorType tensorType) {
  llvm::errs() << getLayoutStr(tensorType, /*useHWPointOfView=*/false);
}

void mlir::triton::gpu::dumpHWLayout(RankedTensorType tensorType) {
  llvm::errs() << getLayoutStr(tensorType, /*useHWPointOfView=*/true);
}

namespace {
struct TensorModel
    : public triton::gpu::TensorOrMemDesc::ExternalModel<TensorModel,
                                                         RankedTensorType> {
  Type getElementType(Type pointer) const {
    return cast<RankedTensorType>(pointer).getElementType();
  }
  Attribute getEncoding(Type pointer) const {
    return cast<RankedTensorType>(pointer).getEncoding();
  }
  ArrayRef<int64_t> getShape(Type pointer) const {
    return cast<RankedTensorType>(pointer).getShape();
  }
  int64_t getRank(Type pointer) const {
    return cast<RankedTensorType>(pointer).getRank();
  }
  int64_t getElementTypeBitWidth(Type pointer) const {
    return cast<RankedTensorType>(pointer).getElementTypeBitWidth();
  }
};

struct MemDescModel
    : public triton::gpu::TensorOrMemDesc::ExternalModel<MemDescModel,
                                                         MemDescType> {
  Type getElementType(Type pointer) const {
    return cast<MemDescType>(pointer).getElementType();
  }
  Attribute getEncoding(Type pointer) const {
    return cast<MemDescType>(pointer).getEncoding();
  }
  ArrayRef<int64_t> getShape(Type pointer) const {
    return cast<MemDescType>(pointer).getShape();
  }
  int64_t getRank(Type pointer) const {
    return cast<MemDescType>(pointer).getShape().size();
  }
  int64_t getElementTypeBitWidth(Type pointer) const {
    return cast<MemDescType>(pointer).getElementType().getIntOrFloatBitWidth();
  }
};
} // namespace

void TritonGPUDialect::initialize() {
  registerTypes();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonGPU/IR/AttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"
#include "triton/Dialect/TritonGPU/IR/OpsEnums.cpp.inc"
      >();
  addInterfaces<TritonInlinerInterface>();
  addInterfaces<TritonGPUOpAsmInterface>();
  addInterfaces<TritonGPUInferLayoutInterface>();
  addInterfaces<TritonGPUVerifyTensorLayoutInterface>();

  RankedTensorType::attachInterface<TensorModel>(*getContext());
  MemDescType::attachInterface<MemDescModel>(*getContext());
}

LogicalResult TritonGPUDialect::verifyOperationAttribute(Operation *op,
                                                         NamedAttribute attr) {
  // Verify that dialect attributes are attached to the right ops.
  if (llvm::is_contained(
          {AttrNumCTAsName, AttrTargetName, AttrNumThreadsPerWarp},
          attr.getName()) &&
      !isa<ModuleOp>(op)) {
    return op->emitOpError("has unexpected attribute ")
           << attr.getName() << " which is expected only on `module` ops";
  }
  if (attr.getName() == AttrNumWarpsName && !isa<ModuleOp, FuncOp>(op)) {
    return op->emitOpError("has unexpected attribute ")
           << attr.getName()
           << " which is expected only on `module` or `tt.func` ops";
  }

  return success();
}

int TritonGPUDialect::getNumCTAs(ModuleOp module) {
  if (auto attr = module->getAttrOfType<IntegerAttr>(AttrNumCTAsName))
    return attr.getInt();
  return 1;
}

int TritonGPUDialect::getThreadsPerWarp(ModuleOp module) {
  if (auto attr = module->getAttrOfType<IntegerAttr>(AttrNumThreadsPerWarp))
    return attr.getInt();
  return 32;
}

std::optional<int> triton::gpu::maybeLookupNumWarps(Operation *op) {
  if (isa<ModuleOp, FuncOp>(op)) {
    if (auto attr = op->getAttrOfType<IntegerAttr>(AttrNumWarpsName))
      return attr.getInt();
  } else if (auto partitions =
                 dyn_cast<WarpSpecializePartitionsOp>(op->getParentOp())) {
    unsigned idx = op->getParentRegion()->getRegionNumber();
    return partitions.getParentOp().getPartitionNumWarps()[idx];
  }
  if (Operation *parent = op->getParentOp())
    return maybeLookupNumWarps(parent);
  return {};
}

int triton::gpu::lookupNumWarps(Operation *op) {
  std::optional<int> numWarps = maybeLookupNumWarps(op);
  if (!numWarps) {
    op->emitOpError(
        "is not contained within a context that specifies the number of warps");
    llvm::report_fatal_error("failed to lookup the number of warps, the "
                             "surrounding module should contain a " +
                             Twine(AttrNumWarpsName) + " attribute");
  }
  return *numWarps;
}

int triton::gpu::lookupNumWarps(Region *region) {
  if (auto partitions =
          dyn_cast<WarpSpecializePartitionsOp>(region->getParentOp())) {
    unsigned idx = region->getRegionNumber();
    return partitions.getParentOp().getPartitionNumWarps()[idx];
  }
  return lookupNumWarps(region->getParentOp());
}

int triton::gpu::lookupThreadsPerWarp(OpBuilder &rewriter) {
  assert(rewriter.getInsertionBlock() && "expected an insertion point");
  Operation *op =
      rewriter.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  assert(op && "cannot check threads per warp outside of module");
  return triton::gpu::TritonGPUDialect::getThreadsPerWarp(cast<ModuleOp>(op));
}

int triton::gpu::lookupNumCTAs(OpBuilder &rewriter) {
  assert(rewriter.getInsertionBlock() && "expected an insertion point");
  Operation *op =
      rewriter.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  assert(op && "cannot check number of CTAs outside of module");
  return triton::gpu::TritonGPUDialect::getNumCTAs(cast<ModuleOp>(op));
}

bool triton::gpu::areLayoutsEquivalent(ArrayRef<int64_t> shape,
                                       LayoutEncodingTrait lhs,
                                       LayoutEncodingTrait rhs) {
  auto lhsLL = triton::gpu::toLinearLayout(shape, lhs);
  auto rhsLL = triton::gpu::toLinearLayout(shape, rhs);
  return lhsLL == rhsLL;
}

bool triton::gpu::isInnermostContiguous(MemDescType type, unsigned numElems) {
  ArrayRef<int64_t> shape = type.getShape();
  Attribute enc = type.getEncoding();
  MLIRContext *ctx = enc.getContext();

  LinearLayout actual = toLinearLayout(type);
  StringAttr fastestIn = *actual.getInDimNames().begin();

  // Flatten actual outs in reverse order to produce a row-major flattening
  // of the layout
  auto outNames = actual.getOutDimNames();
  SmallVector<StringAttr> revOut(outNames.begin(), outNames.end());
  std::reverse(revOut.begin(), revOut.end());
  actual = actual.transposeOuts(revOut).flattenOuts();

  return actual.getNumConsecutiveInOut() >= numElems;
}

LinearLayout triton::gpu::inferReshapeLinearLayout(TensorOrMemDesc srcTy,
                                                   ArrayRef<int64_t> dstShape) {
  auto *ctx = srcTy.getContext();
  auto src = toLinearLayout(srcTy);
  assert(product(srcTy.getShape()) == product(dstShape));
  auto dst = reshapeLayout(ctx, src, dstShape);
  return dst;
}
