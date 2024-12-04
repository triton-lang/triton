#include "triton/Dialect/Triton/IR/Dialect.h"

#include <cstdint>
#include <numeric>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/TypeSwitch.h"

// Include TableGen'erated code
#include "triton/Dialect/TritonGPU/IR/Dialect.cpp.inc"
#include "triton/Dialect/TritonGPU/IR/TypeInterfaces.cpp.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// Utility
namespace mlir {
namespace triton {

static Type getI1SameShapeFromTensorOrTensorPtr(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return RankedTensorType::get(tensorType.getShape(), i1Type,
                                 tensorType.getEncoding());
  } else if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    Type pointeeType = ptrType.getPointeeType();
    if (auto tensorType = dyn_cast<RankedTensorType>(pointeeType)) {
      return RankedTensorType::get(tensorType.getShape(), i1Type,
                                   tensorType.getEncoding());
    }
  }
  return Type();
}

namespace gpu {

// TODO: Inheritance of layout attributes
// so that all distributed layouts implement
// these utilities

unsigned getTotalElemsPerThread(Attribute layout, ArrayRef<int64_t> shape,
                                Type eltTy) {
  if (auto tritonGPUAttr = mlir::dyn_cast<TritonGPU_AttrTrait>(layout)) {
    return tritonGPUAttr.getTotalElemsPerThread(shape, eltTy);
  } else {
    llvm::report_fatal_error("getTotalElemsPerThread not implemented");
    return 0;
  }
}

SmallVector<unsigned> getElemsPerThread(Attribute layout,
                                        ArrayRef<int64_t> shape, Type eltTy) {
  if (auto tritonGPUAttr = mlir::dyn_cast<TritonGPU_AttrTrait>(layout)) {
    return tritonGPUAttr.getElemsPerThread(shape, eltTy);
  } else {
    llvm::report_fatal_error("getElemsPerThread not implemented");
    return SmallVector<unsigned>();
  }
}

SmallVector<unsigned> getElemsPerThread(Type type) {
  if (type.isIntOrIndexOrFloat() || isa<triton::PointerType>(type))
    return SmallVector<unsigned>(1, 1);
  auto tensorType = cast<RankedTensorType>(type);
  return getElemsPerThread(tensorType.getEncoding(), tensorType.getShape(),
                           tensorType.getElementType());
}

unsigned getTotalElemsPerThread(Type type) {
  if (type.isIntOrIndexOrFloat() || isa<triton::PointerType>(type))
    return 1;
  auto tensorType = cast<RankedTensorType>(type);
  return getTotalElemsPerThread(tensorType.getEncoding(), tensorType.getShape(),
                                tensorType.getElementType());
}

SmallVector<unsigned> getThreadsPerWarp(Attribute layout) {
  if (auto distributedLayout = dyn_cast<DistributedEncodingTrait>(layout)) {
    return distributedLayout.getThreadsPerWarp();
  } else {
    llvm::report_fatal_error("getThreadsPerWarp not implemented");
    return SmallVector<unsigned>();
  }
}

unsigned getWarpSize(Attribute layout) {
  unsigned size = 1;
  auto threadsPerWarp = getThreadsPerWarp(layout);
  for (auto e : threadsPerWarp) {
    size *= e;
  }
  return size;
}

SmallVector<unsigned>
getThreadsPerWarpWithUniqueData(Attribute layout,
                                ArrayRef<int64_t> tensorShape) {
  if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(layout)) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(tensorShape);
    auto parentThreadsPerWarp =
        getThreadsPerWarpWithUniqueData(parentLayout, parentShape);
    SmallVector<unsigned> threadsPerWarp = parentThreadsPerWarp;
    threadsPerWarp.erase(threadsPerWarp.begin() + sliceLayout.getDim());
    return threadsPerWarp;
  }
  auto threadsPerWarp = getThreadsPerWarp(layout);
  assert(threadsPerWarp.size() == tensorShape.size() &&
         "layout and tensor shape must have the same rank");
  for (unsigned i = 0; i < threadsPerWarp.size(); i++) {
    threadsPerWarp[i] = std::min<unsigned>(threadsPerWarp[i], tensorShape[i]);
  }

  return threadsPerWarp;
}

SmallVector<unsigned> getWarpsPerCTA(Attribute layout) {
  if (auto distributedLayout =
          mlir::dyn_cast<DistributedEncodingTrait>(layout)) {
    return distributedLayout.getWarpsPerCTA();
  }

  llvm::report_fatal_error("getWarpsPerCTA not implemented");
  return SmallVector<unsigned>();
}

SmallVector<unsigned>
getWarpsPerCTAWithUniqueData(Attribute layout, ArrayRef<int64_t> tensorShape) {
  if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(layout)) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(tensorShape);
    auto parentWarpsPerCTA =
        getWarpsPerCTAWithUniqueData(parentLayout, parentShape);
    SmallVector<unsigned> warpsPerCTA = parentWarpsPerCTA;
    warpsPerCTA.erase(warpsPerCTA.begin() + sliceLayout.getDim());
    return warpsPerCTA;
  }
  auto warpsPerCTA = getWarpsPerCTA(layout);
  assert(warpsPerCTA.size() == tensorShape.size() &&
         "layout and tensor shape must have the same rank");
  for (unsigned i = 0; i < warpsPerCTA.size(); i++) {
    auto sizePerWarp =
        getSizePerThread(layout)[i] * getThreadsPerWarp(layout)[i];
    auto maxWarpsPerDim = ceil<unsigned>(tensorShape[i], sizePerWarp);
    warpsPerCTA[i] = std::min<unsigned>(warpsPerCTA[i], maxWarpsPerDim);
  }

  return warpsPerCTA;
}

SmallVector<unsigned> getSizePerThread(Attribute layout) {
  if (auto distributedLayout =
          mlir::dyn_cast<DistributedEncodingTrait>(layout)) {
    return distributedLayout.getSizePerThread();
  } else {
    llvm::report_fatal_error("getSizePerThread not implemented");
    return {};
  }
}

SmallVector<unsigned> getContigPerThread(Attribute layout) {
  if (auto distributedLayout = dyn_cast<DistributedEncodingTrait>(layout)) {
    return distributedLayout.getContigPerThread();
  } else {
    llvm::report_fatal_error("getContigPerThread not implemented");
    return {};
  }
}

SmallVector<unsigned> getUniqueContigPerThread(Attribute layout,
                                               ArrayRef<int64_t> shape) {
  // If slice layout, call recursively on parent layout, and drop
  // sliced dim
  if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(layout)) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(shape);
    auto parentUniqueContigPerThread =
        getUniqueContigPerThread(parentLayout, parentShape);
    parentUniqueContigPerThread.erase(parentUniqueContigPerThread.begin() +
                                      sliceLayout.getDim());
    return parentUniqueContigPerThread;
  }
  // Base case
  auto rank = shape.size();
  SmallVector<unsigned> ret(rank);
  auto contigPerThread = getContigPerThread(layout);
  assert(contigPerThread.size() == rank && "Unexpected contigPerThread size");
  for (int d = 0; d < rank; ++d) {
    ret[d] = std::min<unsigned>(shape[d], contigPerThread[d]);
  }
  return ret;
}
SmallVector<unsigned> getShapePerCTATile(Attribute layout) {
  if (auto distributedLayout =
          mlir::dyn_cast<DistributedEncodingTrait>(layout)) {
    auto sizePerThread = distributedLayout.getSizePerThread();
    auto threadsPerWarp = distributedLayout.getThreadsPerWarp();
    // ThreadsPerWarp does not align with this function for slice layout
    if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(layout)) {
      threadsPerWarp = getThreadsPerWarp(sliceLayout.getParent());
      threadsPerWarp.erase(threadsPerWarp.begin() + sliceLayout.getDim());
    }
    auto warpsPerCTA = distributedLayout.getWarpsPerCTA();
    assert(sizePerThread.size() == threadsPerWarp.size() &&
           sizePerThread.size() == warpsPerCTA.size());
    SmallVector<unsigned> shape;
    for (auto [size, thread, warp] :
         llvm::zip(sizePerThread, threadsPerWarp, warpsPerCTA)) {
      shape.push_back(size * thread * warp);
    }
    return shape;
  } else {
    llvm::report_fatal_error("getShapePerCTATile not implemented");
    return SmallVector<unsigned>();
  }
}

bool isExpensiveView(Type srcType, Type dstType) {
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
  assert(rank >= 2);
  SmallVector<unsigned> order(rank);
  std::iota(order.rbegin(), order.rend(), 0);
  if (!rowMajor) {
    std::swap(order[0], order[1]);
  }
  return order;
}

SmallVector<unsigned> getOrderForDotOperand(unsigned opIdx, unsigned rank,
                                            bool kMajor) {
  // kMajor: if true, the matrix is fastest-running on k,
  //         otherwise it is on m (resp. n)
  // opIdx=0: [batch, m, k] if rank == 3 else [m, k]
  // opIdx=1: [batch, k, n] if rank == 3 else [k, n]
  // batch (if rank == 3) is always the slowest running dimension
  assert(rank == 2 || rank == 3);
  assert(opIdx == 0 || opIdx == 1);
  auto rowMajor = bool(opIdx) != kMajor;
  return getMatrixOrder(rank, rowMajor);
}

SmallVector<unsigned> getRepOrder(Attribute layout) {
  if (auto distributedLayout = mlir::dyn_cast<DistributedEncodingTrait>(layout))
    return distributedLayout.getRepOrder();
  else
    llvm::report_fatal_error("Unimplemented usage of getRepOrder");
  return {};
}

SmallVector<unsigned> getWarpOrder(Attribute layout) {
  if (auto distributedLayout = mlir::dyn_cast<DistributedEncodingTrait>(layout))
    return distributedLayout.getWarpOrder();
  else
    llvm::report_fatal_error("Unimplemented usage of getThreadOrder");
  return {};
}

// Returns the order of the elements in a layout from the fastest running
// dimension to the slowest
SmallVector<unsigned> getOrder(Attribute layout) {
  if (auto blockedLayout = dyn_cast<BlockedEncodingAttr>(layout)) {
    return llvm::to_vector(blockedLayout.getOrder());
  }
  if (auto mmaLayout = dyn_cast<MmaEncodingTrait>(layout)) {
    auto distributedLayout = cast<DistributedEncodingTrait>(layout);
    auto rank = distributedLayout.getWarpsPerCTA().size();
    return getMatrixOrder(rank, /*rowMajor*/ true);
  }
  if (auto dotLayout = dyn_cast<DotOperandEncodingAttr>(layout)) {
    auto rank = dotLayout.getWarpsPerCTA().size();
    return getOrderForDotOperand(dotLayout.getOpIdx(), rank, /*kMajor*/ true);
  }
  if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(layout)) {
    SmallVector<unsigned> parentOrder = getOrder(sliceLayout.getParent());
    unsigned dim = sliceLayout.getDim();
    SmallVector<unsigned> order;
    for (unsigned d : parentOrder) {
      if (d != dim)
        order.push_back(d > dim ? d - 1 : d);
    }
    return order;
  }
  if (auto sharedLayout = mlir::dyn_cast<SharedEncodingAttr>(layout)) {
    return llvm::to_vector(sharedLayout.getOrder());
  }
  if (auto linearLayout = mlir::dyn_cast<LinearEncodingAttr>(layout)) {
    return linearLayout.getOrder();
  }

  llvm::report_fatal_error("Unimplemented usage of getOrder");
  return {};
}

SmallVector<unsigned> getThreadOrder(Attribute layout) {
  if (auto distributedLayout = mlir::dyn_cast<DistributedEncodingTrait>(layout))
    return distributedLayout.getThreadOrder();
  else
    llvm::report_fatal_error("Unimplemented usage of getThreadOrder");
  return {};
}

CTALayoutAttr getCTALayout(Attribute layout) {
  if (auto distributedLayout =
          mlir::dyn_cast<DistributedEncodingTrait>(layout)) {
    return CTALayoutAttr::get(
        layout.getContext(), getCTAsPerCGA(distributedLayout),
        getCTASplitNum(distributedLayout), getCTAOrder(distributedLayout));
  } else if (auto sharedLayout = mlir::dyn_cast<SharedEncodingAttr>(layout))
    return sharedLayout.getCTALayout();
  else
    llvm::report_fatal_error("Unimplemented usage of getCTALayout");
  return {};
}

SmallVector<unsigned> getCTAsPerCGA(Attribute layout) {
  ArrayRef<unsigned> ref;
  if (auto distributedLayout = mlir::dyn_cast<DistributedEncodingTrait>(layout))
    return distributedLayout.getCTAsPerCGA();
  else if (auto sharedLayout = mlir::dyn_cast<SharedEncodingAttr>(layout))
    ref = sharedLayout.getCTALayout().getCTAsPerCGA();
  else
    llvm::report_fatal_error("Unimplemented usage of getCTAsPerCGA");
  return SmallVector<unsigned>(ref.begin(), ref.end());
}

SmallVector<unsigned> getCTASplitNum(Attribute layout) {
  SmallVector<unsigned> res;
  if (auto distributedLayout =
          mlir::dyn_cast<DistributedEncodingTrait>(layout)) {
    return distributedLayout.getCTASplitNum();
  } else if (auto sharedLayout = mlir::dyn_cast<SharedEncodingAttr>(layout)) {
    res.assign(sharedLayout.getCTALayout().getCTASplitNum().begin(),
               sharedLayout.getCTALayout().getCTASplitNum().end());
  } else {
    assert(false && "Unimplemented usage of getCTASplitNum");
  }
  return res;
}

SmallVector<unsigned> getCTAOrder(Attribute layout) {
  SmallVector<unsigned> res;
  if (auto distributedLayout =
          mlir::dyn_cast<DistributedEncodingTrait>(layout)) {
    res = distributedLayout.getCTAOrder();
  } else if (auto sharedLayout = mlir::dyn_cast<SharedEncodingAttr>(layout)) {
    res = SmallVector<unsigned>(sharedLayout.getCTALayout().getCTAOrder());
  } else {
    llvm::report_fatal_error("Unimplemented usage of getCTAOrder");
  }
  return res;
}

SmallVector<int64_t> getShapePerCTA(ArrayRef<unsigned> CTASplitNum,
                                    ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  SmallVector<int64_t> shapePerCTA(rank);
  for (unsigned i = 0; i < rank; ++i) {
    // This wrapping rule must be consistent with emitCTAOffsetForLayout
    unsigned splitNum = std::min<unsigned>(shape[i], CTASplitNum[i]);
    shapePerCTA[i] = shape[i] / splitNum;
  }
  return shapePerCTA;
}

SmallVector<int64_t> getShapePerCTA(Attribute layout, ArrayRef<int64_t> shape) {
  if (auto sharedLayout = mlir::dyn_cast<SharedEncodingAttr>(layout)) {
    // Special logic for pipeline pass, where shape is 3D and CTALayout is 2D.
    // The first dim of shape is numStages. This is a work around, otherwise
    // too many places would have to be modified in pipeline pass. Maybe we
    // need to refactor this logic in the future.
    auto CTASplitNum = sharedLayout.getCTALayout().getCTASplitNum();
    if (shape.size() == CTASplitNum.size() + 1) {
      auto res = getShapePerCTA(CTASplitNum, shape.drop_front());
      res.insert(res.begin(), shape.front());
      return res;
    }
  }
  return getShapePerCTA(getCTASplitNum(layout), shape);
}

SmallVector<int64_t> getShapePerCTA(Type type) {
  auto tensorType = cast<TensorOrMemDesc>(type);
  return getShapePerCTA(tensorType.getEncoding(), tensorType.getShape());
}

unsigned getNumWarpsPerCTA(Attribute layout) {
  SmallVector<unsigned> warpsPerCTA;
  if (auto blockedLayout = dyn_cast<BlockedEncodingAttr>(layout))
    warpsPerCTA = blockedLayout.getWarpsPerCTA();
  else if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(layout))
    return getNumWarpsPerCTA(sliceLayout.getParent());
  else if (auto mmaLayout = dyn_cast<MmaEncodingTrait>(layout)) {
    // Use the distributed layout interface to get the number of warps per
    // CTA.
    auto distributedLayout = cast<DistributedEncodingTrait>(layout);
    warpsPerCTA = distributedLayout.getWarpsPerCTA();
  } else if (auto mfmaLayout = dyn_cast<AMDMfmaEncodingAttr>(layout))
    warpsPerCTA = mfmaLayout.getWarpsPerCTA();
  else if (auto wmmaLayout = dyn_cast<AMDWmmaEncodingAttr>(layout))
    warpsPerCTA = wmmaLayout.getWarpsPerCTA();
  else if (auto dotLayout = dyn_cast<DotOperandEncodingAttr>(layout))
    warpsPerCTA = dotLayout.getWarpsPerCTA();
  else if (auto sharedLayout = dyn_cast<SharedEncodingAttr>(layout))
    llvm::report_fatal_error("Cannot get numWarps from SharedEncodingAttr");
  else
    llvm::report_fatal_error("Unimplemented usage of getNumWarpsPerCTA");
  return product<unsigned>(warpsPerCTA);
}

unsigned getNumCTAs(Attribute layout) {
  return product<unsigned>(getCTAsPerCGA(layout));
}

template <typename T> bool hasEncoding(Value value) {
  auto type = value.getType();
  if (auto tensorType = dyn_cast<TensorOrMemDesc>(type)) {
    auto encoding = tensorType.getEncoding();
    return encoding && isa<T>(encoding);
  }
  return false;
}

bool hasDotOperandEncoding(Value value) {
  return hasEncoding<triton::gpu::DotOperandEncodingAttr>(value);
}

bool isExpensiveCat(CatOp cat, Attribute targetEncoding) {
  // If the new elements per thread is less than the old one, we will need to
  // do convert encoding that goes through shared memory anyway. So we
  // consider it as expensive.
  RankedTensorType tensorTy = cat.getType();
  auto totalElemsPerThread = gpu::getTotalElemsPerThread(tensorTy);
  auto shape = tensorTy.getShape();
  auto elemTy = tensorTy.getElementType();
  auto newTotalElemsPerThread =
      gpu::getTotalElemsPerThread(targetEncoding, shape, elemTy);
  return newTotalElemsPerThread < totalElemsPerThread;
}

LogicalResult CTALayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<unsigned> CTAsPerCGA,
    ArrayRef<unsigned> CTASplitNum, ArrayRef<unsigned> CTAOrder) {
  if (CTAsPerCGA.size() != CTASplitNum.size() ||
      CTASplitNum.size() != CTAOrder.size()) {
    return emitError() << "CTAsPerCGA, CTASplitNum, and CTAOrder must all have "
                          "the same rank.";
  }

  if (!isPermutationOfIota(CTAOrder)) {
    return emitError()
           << "CTAOrder must be a permutation of 0..(rank-1), but was ["
           << CTAOrder << "]";
  }

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
  if (sizePerThread.size() != threadsPerWarp.size() ||
      threadsPerWarp.size() != warpsPerCTA.size() ||
      warpsPerCTA.size() != order.size()) {
    return emitError() << "sizePerThread, threadsPerWarp, warpsPerCTA, and "
                          "order must all have the same rank.";
  }

  // Empty CTALayout is allowed, but if it's present its rank must match the
  // BlockedEncodingAttr's rank.
  if (CTALayout.getCTASplitNum().size() != 0 &&
      sizePerThread.size() != CTALayout.getCTASplitNum().size()) {
    return emitError() << "BlockedEncodingAttr and CTALayout's fields must "
                          "have the same rank.";
  }
  if (!isPermutationOfIota(order)) {
    return emitError()
           << "order must be a permutation of 0..(rank-1), but was [" << order
           << "]";
  }
  return success();
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

LinearLayout
ensureLayoutNotLargerThan(const LinearLayout &layout,
                          const llvm::SmallDenseMap<StringAttr, int64_t> &shape,
                          bool broadcastRegisters) {
  assert(shape.size() == layout.getNumOutDims());
  if (shape.empty()) {
    return layout;
  }
  MLIRContext *ctx = shape.begin()->first.getContext();

  auto bases = layout.getBases();

  auto kRegister = StringAttr::get(ctx, "register");
  std::set<int32_t> broadcastedDims;

  for (auto outDim : llvm::enumerate(layout.getOutDimNames())) {
    auto outDimName = outDim.value();
    int32_t actualSize = layout.getOutDimSize(outDimName);
    int32_t desiredSize = shape.lookup(outDimName);
    if (actualSize <= desiredSize) {
      continue;
    }
    assert(actualSize % desiredSize == 0);
    // <inDimName, basisIdx, outValue>
    std::vector<std::tuple<StringAttr, int, int>> sortedBases;
    for (auto [inDimName, basis] : bases) {
      for (size_t basisIdx = 0; basisIdx < basis.size(); basisIdx++) {
        auto outValue = basis[basisIdx][outDim.index()];
        if (outValue == 0) {
          continue;
        }
        assert(llvm::isPowerOf2_32(outValue));
        sortedBases.emplace_back(inDimName, basisIdx, outValue);
      }
    }
    // From the largest basis to the smallest.
    llvm::sort(sortedBases,
               [](auto a, auto b) { return std::get<2>(a) > std::get<2>(b); });
    for (auto [inDimName, basisIdx, outValue] : sortedBases) {
      if (actualSize <= desiredSize) {
        break;
      }
      if (!broadcastRegisters && inDimName == kRegister) {
        broadcastedDims.insert(basisIdx);
      } else {
        bases[inDimName][basisIdx][outDim.index()] = 0;
      }
      actualSize >>= 1;
    }
  }
  if (!broadcastRegisters) {
    // Remove broadcasted registers
    std::vector<std::vector<int32_t>> newBasesRegister;
    for (auto [idx, basis] : llvm::enumerate(bases[kRegister])) {
      // Remove if it's broadcasted
      if (broadcastedDims.find(idx) == broadcastedDims.end()) {
        newBasesRegister.push_back(std::move(basis));
      }
    }
    bases[kRegister] = std::move(newBasesRegister);
  }

  return LinearLayout(std::move(bases),
                      llvm::to_vector(layout.getOutDimNames()));
}

// For each out-dim d, ensure the layout's out-size (i.e. its codomain) is no
// smaller than shape[d].  Do this by increasing the size of the layout's inputs
// along its most-minor dimension ("register" for register layouts, "offset" for
// shared layouts).
//
// This function is invariant to the order of the layout's input dimensions, but
// it cares about the order of the output dims, which should be minor-to-major.
LinearLayout ensureLayoutNotSmallerThan(
    const LinearLayout &layout,
    const llvm::SmallDenseMap<StringAttr, int64_t> &shape) {
  assert(shape.size() == layout.getNumOutDims());
  if (shape.empty()) {
    return layout;
  }

  MLIRContext *ctx = shape.begin()->first.getContext();
  StringAttr kDim = *layout.getInDimNames().begin();
  assert(kDim == "register" || kDim == "offset");

  LinearLayout ret = layout;
  for (StringAttr outDimName : layout.getOutDimNames()) {
    int32_t actualSize = layout.getOutDimSize(outDimName);
    int32_t desiredSize = shape.lookup(outDimName);
    assert(actualSize > desiredSize || desiredSize % actualSize == 0);
    ret *= LinearLayout::identity1D(desiredSize / actualSize, kDim, outDimName);
    assert(ret.getOutDimSize(outDimName) >= desiredSize);
  }
  return ret;
}

// Returns ["dim0", "dim1", ..., "dim<rank-1>"].
SmallVector<StringAttr> standardOutDimNames(MLIRContext *ctx, int rank) {
  SmallVector<StringAttr> ret;
  for (int i = 0; i < rank; i++) {
    ret.push_back(StringAttr::get(ctx, "dim" + llvm::Twine(i)));
  }
  return ret;
}

// Returns a 1D -> ND layout into [dim0, dim1, ...] that's equivalent to
// creating a 1D -> 1D mapping of size product(shape) and then reshaping to
// permute(shape, order).
LinearLayout identityStandardND(StringAttr inDimName, ArrayRef<unsigned> shape,
                                ArrayRef<unsigned> order) {
  assert(shape.size() == order.size());
  MLIRContext *ctx = inDimName.getContext();
  auto rank = shape.size();

  // The order in triton is written wrt. [dim0, dim1, ...].
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  LinearLayout ret = LinearLayout::empty();
  for (int i = 0; i < shape.size(); i++) {
    // Start with the most-minor dimension, which is order[0].
    int dim = order[i];
    ret *= LinearLayout::identity1D(shape[dim], inDimName, outDimNames[dim]);
  }
  return ret;
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

SliceEncodingAttr BlockedEncodingAttr::squeeze(int axis) {
  return SliceEncodingAttr::get(getContext(), axis, *this);
}
SmallVector<unsigned>
BlockedEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                       Type eltTy) const {
  size_t rank = shape.size();
  auto sizePerThread = getSizePerThread();
  auto warpsPerCTA = getWarpsPerCTA();
  auto threadsPerWarp = getThreadsPerWarp();
  auto shapePerCTA = getShapePerCTA(*this, shape);
  assert(rank == sizePerThread.size() &&
         "unexpected rank in BlockedEncodingAttr::getElemsPerThread");
  SmallVector<unsigned> elemsPerThread(rank);
  for (size_t i = 0; i < rank; ++i) {
    unsigned t = sizePerThread[i] * threadsPerWarp[i] * warpsPerCTA[i];
    elemsPerThread[i] = ceil<unsigned>(shapePerCTA[i], t) * sizePerThread[i];
  }
  return elemsPerThread;
}
unsigned BlockedEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                     Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}

// If we only had BlockedEncodingAttr, we could simply return ArrayRefs here.
// But we need to have a consistent interface with e.g. SliceEncodingAttr, which
// computes some of these fields.
SmallVector<unsigned> BlockedEncodingAttr::getRepOrder() const {
  return SmallVector<unsigned>(getOrder());
}
SmallVector<unsigned> BlockedEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> BlockedEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> BlockedEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}
SmallVector<unsigned> BlockedEncodingAttr::getWarpsPerCTA() const {
  return SmallVector<unsigned>(getWarpsPerCTA__());
}
SmallVector<unsigned> BlockedEncodingAttr::getWarpOrder() const {
  return SmallVector<unsigned>(getOrder());
}
SmallVector<unsigned> BlockedEncodingAttr::getThreadsPerWarp() const {
  return SmallVector<unsigned>(getThreadsPerWarp__());
}
SmallVector<unsigned> BlockedEncodingAttr::getThreadOrder() const {
  return SmallVector<unsigned>(getOrder());
}
SmallVector<unsigned> BlockedEncodingAttr::getSizePerThread() const {
  return SmallVector<unsigned>(getSizePerThread__());
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

SmallVector<unsigned>
SliceEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                     Type eltTy) const {
  auto parent = getParent();
  auto parentElemsPerThread =
      ::getElemsPerThread(parent, paddedShape(shape), eltTy);
  parentElemsPerThread.erase(parentElemsPerThread.begin() + getDim());
  return parentElemsPerThread;
}
unsigned SliceEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                   Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}
SmallVector<unsigned> SliceEncodingAttr::getRepOrder() const {
  auto parentRepOrder = ::getRepOrder(getParent());
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
SmallVector<unsigned> SliceEncodingAttr::getWarpsPerCTA() const {
  auto parent = getParent();
  auto parentWarpsPerCTA = ::getWarpsPerCTA(parent);
  SmallVector<unsigned> warpsPerCTA = parentWarpsPerCTA;
  warpsPerCTA.erase(warpsPerCTA.begin() + getDim());
  int32_t nextDim = getDim() < warpsPerCTA.size() ? getDim() : getDim() - 1;
  warpsPerCTA[nextDim] *= parentWarpsPerCTA[getDim()];
  return warpsPerCTA;
}
SmallVector<unsigned> SliceEncodingAttr::getWarpOrder() const {
  auto parentWarpOrder = ::getWarpOrder(getParent());
  return eraseOrder(parentWarpOrder, getDim());
}
SmallVector<unsigned> SliceEncodingAttr::getThreadsPerWarp() const {
  auto parent = getParent();
  auto parentThreadsPerWarp = ::getThreadsPerWarp(parent);
  SmallVector<unsigned> threadsPerWarp = parentThreadsPerWarp;
  threadsPerWarp.erase(threadsPerWarp.begin() + getDim());
  int32_t nextDim = getDim() < threadsPerWarp.size() ? getDim() : getDim() - 1;
  threadsPerWarp[nextDim] *= parentThreadsPerWarp[getDim()];
  return threadsPerWarp;
}
SmallVector<unsigned> SliceEncodingAttr::getThreadOrder() const {
  auto parentThreadOrder = ::getThreadOrder(getParent());
  return eraseOrder(parentThreadOrder, getDim());
}
SmallVector<unsigned> SliceEncodingAttr::getSizePerThread() const {
  auto sizePerThread = ::getSizePerThread(getParent());
  sizePerThread.erase(sizePerThread.begin() + getDim());
  return sizePerThread;
}

//

SmallVector<unsigned>
AMDMfmaEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                       Type eltTy) const {
  size_t rank = shape.size();
  assert((rank == 2 || rank == 3) && "Unexpected rank of mfma layout");

  SmallVector<unsigned> elemsPerThread(rank);
  auto nonKDim = getMDim();
  auto elemsPerThreadPerTile = (nonKDim == 16 ? 4 : 16);
  if (rank == 3)
    elemsPerThread[0] = ceil<unsigned>(shape[0], getWarpsPerCTA()[0]);
  if (getIsTransposed()) {
    unsigned elemsCol =
        ceil<unsigned>(shape[rank - 1], nonKDim * getWarpsPerCTA()[rank - 1]) *
        elemsPerThreadPerTile;
    unsigned elemsRow =
        ceil<unsigned>(shape[rank - 2], nonKDim * getWarpsPerCTA()[rank - 2]);
    elemsPerThread[rank - 2] = elemsRow;
    elemsPerThread[rank - 1] = elemsCol;
  } else {
    unsigned elemsCol =
        ceil<unsigned>(shape[rank - 1], nonKDim * getWarpsPerCTA()[rank - 1]);
    unsigned elemsRow =
        ceil<unsigned>(shape[rank - 2], nonKDim * getWarpsPerCTA()[rank - 2]) *
        elemsPerThreadPerTile;
    elemsPerThread[rank - 2] = elemsRow;
    elemsPerThread[rank - 1] = elemsCol;
  }
  return elemsPerThread;
}

unsigned AMDMfmaEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                     Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}

// Wmma encoding

SmallVector<unsigned>
AMDWmmaEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                       Type eltTy) const {
  size_t rank = shape.size();
  assert((rank == 2 || rank == 3) && "Unexpected rank of wmma layout");

  SmallVector<unsigned> elemsPerThread(rank);
  auto mnkDim = getMNKDimPerInstr();
  auto elemsPerThreadPerTile = getSizePerThread();
  auto warpsPerCTA = getWarpsPerCTA();

  if (rank == 3)
    elemsPerThread[0] = ceil<unsigned>(shape[0], getWarpsPerCTA()[0]);
  elemsPerThread[rank - 2] =
      ceil<unsigned>(shape[rank - 2], mnkDim[0] * warpsPerCTA[rank - 2]) *
      elemsPerThreadPerTile[rank - 2];
  elemsPerThread[rank - 1] =
      ceil<unsigned>(shape[rank - 1], mnkDim[1] * warpsPerCTA[rank - 1]) *
      elemsPerThreadPerTile[rank - 1];
  return elemsPerThread;
}

unsigned AMDWmmaEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                     Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}

SmallVector<unsigned>
NvidiaMmaEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                         Type eltTy) const {
  size_t rank = shape.size();
  assert(rank == 2 ||
         (rank == 3 && isAmpere()) && "Unexpected rank of mma layout");
  assert((isAmpere() || isHopper()) &&
         "For NvidiaMmaEncodingAttr only version 1~3 is supported");

  auto shapePerCTA = getShapePerCTA(getCTALayout().getCTASplitNum(), shape);

  SmallVector<unsigned> elemsPerThread(rank);
  if (isAmpere()) {
    unsigned elemsRow =
        ceil<unsigned>(shapePerCTA[rank - 2], 16 * getWarpsPerCTA()[rank - 2]) *
        2;
    unsigned elemsCol =
        ceil<unsigned>(shapePerCTA[rank - 1], 8 * getWarpsPerCTA()[rank - 1]) *
        2;
    if (rank == 3)
      elemsPerThread[0] = ceil<unsigned>(shapePerCTA[0], getWarpsPerCTA()[0]);
    elemsPerThread[rank - 2] = elemsRow;
    elemsPerThread[rank - 1] = elemsCol;
  } else if (isHopper()) {
    auto wpt = getWarpsPerCTA();
    auto instrMNK = getInstrShape();
    int repM = ceil<unsigned>(shapePerCTA[0], instrMNK[0] * wpt[0]);
    int repN = ceil<unsigned>(shapePerCTA[1], instrMNK[1] * wpt[1]);
    elemsPerThread[0] = 2 * repM;
    elemsPerThread[1] = (instrMNK[1] / 4) * repN;
  } else {
    llvm_unreachable("Unexpected mma version");
  }

  return elemsPerThread;
}

unsigned NvidiaMmaEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                       Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}

//

SmallVector<unsigned>
SharedEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                      Type eltTy) const {
  llvm_unreachable("getElemsPerThread is not supported for shared layout");
  return SmallVector<unsigned>();
}
unsigned SharedEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                    Type eltTy) const {
  llvm_unreachable("getElemsPerThread is not supported for shared layout");
  return 0;
}

SmallVector<unsigned>
DotOperandEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                          Type eltTy) const {
  auto rank = shape.size();
  assert(rank == 2 || rank == 3);

  auto idx = getOpIdx();
  assert(idx == 0 || idx == 1);

  SmallVector<unsigned> elemsPerThread(rank);
  auto parent = getParent();
  auto kWidth = getKWidth();

  if (auto mfma = mlir::dyn_cast<AMDMfmaEncodingAttr>(parent)) {
    auto rep = mfma.getRepForOperand(shape, kWidth, idx);
    if (rank == 3)
      elemsPerThread[0] = rep[0];
    elemsPerThread[rank - 2] = (idx == 0) ? rep[1] : rep[1] * kWidth;
    elemsPerThread[rank - 1] = (idx == 0) ? rep[2] * kWidth : rep[2];
    return elemsPerThread;
  } else if (auto mma = mlir::dyn_cast<NvidiaMmaEncodingAttr>(parent)) {
    assert(getCTALayout(*this) ==
               CTALayoutAttr::getDefault(getContext(), rank) &&
           "NYI");
    auto sizePerThread = getSizePerThread();
    auto threadsPerWarp = getThreadsPerWarp();
    auto warpsPerCTA = getWarpsPerCTA();
    SmallVector<unsigned> regs;
    for (auto [n, nsize, nThread, nWarp] :
         llvm::zip(shape, sizePerThread, threadsPerWarp, warpsPerCTA)) {
      regs.push_back(std::max<int64_t>(nsize, n / (nThread * nWarp)));
    }
    return regs;
  }

  llvm_unreachable("getElemsPerThread is not supported for dot operand");
  return SmallVector<unsigned>();
}

unsigned DotOperandEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                        Type eltTy) const {
  if (auto mmaParent = mlir::dyn_cast<MmaEncodingTrait>(getParent())) {
    if (auto nvidiaMmaParent =
            mlir::dyn_cast<NvidiaMmaEncodingAttr>(mmaParent)) {
      return product<unsigned>(getElemsPerThread(shape, eltTy));
    }
    if (auto amdMfmaParent = mlir::dyn_cast<AMDMfmaEncodingAttr>(getParent())) {
      return amdMfmaParent.getTotalElemsPerThreadForOperand(
          shape, eltTy, getKWidth(), getOpIdx());
    }
    if (auto amdWmmaParent = mlir::dyn_cast<AMDWmmaEncodingAttr>(getParent())) {
      return amdWmmaParent.getTotalElemsPerThreadForOperand(
          shape, eltTy, getKWidth(), getOpIdx());
    }
  }
  if (auto blockedLayout = mlir::dyn_cast<BlockedEncodingAttr>(getParent())) {
    auto shapePerCTA = getShapePerCTA(*this, shape);
    auto shapePerCTATile = getShapePerCTATile(blockedLayout);
    auto order = blockedLayout.getOrder();
    auto sizePerThread = blockedLayout.getSizePerThread();

    int K = getOpIdx() == 0 ? shapePerCTA[1] : shapePerCTA[0];
    int otherDim = getOpIdx() == 1 ? shapePerCTA[1] : shapePerCTA[0];

    bool isM = getOpIdx() == 0;

    int mSizePerThread =
        order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
    int nSizePerThread =
        order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];
    int sizePerThreadMN = isM ? mSizePerThread : nSizePerThread;

    int mShapePerCTATile =
        order[0] == 1 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
    int nShapePerCTATile =
        order[0] == 0 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
    int shapePerCTAMNTile = isM ? mShapePerCTATile : nShapePerCTATile;

    return K * std::max<int>(otherDim / shapePerCTAMNTile, 1) * sizePerThreadMN;
  }
  llvm_unreachable("unknown dot operand parent layout");
  return 0;
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
SmallVector<unsigned> DotOperandEncodingAttr::getWarpsPerCTA() const {
  auto distributedLayout = mlir::cast<DistributedEncodingTrait>(getParent());
  auto warps = distributedLayout.getWarpsPerCTA();
  auto rank = warps.size();
  auto kDim = getOpIdx() == 0 ? rank - 1 : rank - 2;
  warps[kDim] = 1;
  return warps;
}
SmallVector<unsigned> DotOperandEncodingAttr::getWarpOrder() const {
  // FIXME(Lezcano): Preexisting. Do we want to have this path at all?
  if (mlir::isa<AMDMfmaEncodingAttr>(getParent())) {
    return ::getWarpOrder(getParent());
  }
  // It's quite weird to talk about warp order when that the warps
  // are broadcasted along the K dimension
  llvm::report_fatal_error("DotOperandEncoding::getWarpOrder not implemented");
  return {};
}
SmallVector<unsigned> DotOperandEncodingAttr::getThreadOrder() const {
  return getOrderForDotOperand(getOpIdx(), getWarpsPerCTA().size(),
                               /*kMajor*/ true);
}

LogicalResult DotOperandEncodingAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    unsigned opIdx, Attribute parent, unsigned kWidth) {
  if (opIdx != 0 && opIdx != 1) {
    return emitError() << "ttg.dot_op opIdx paramenter can be 0 or 1, got: "
                       << opIdx;
  }
  if (!parent) {
    return emitError() << "ttg.dot_op parent paramenter cannot be null";
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
    if (kWidth != 16 && parentAttr.getVersion() == 1 ||
        kWidth != 8 && parentAttr.getVersion() == 2)
      return emitError() << "ttg.dot_op kWidth parameter must be 16 for "
                            "gfx11 and 8 for gfx12";
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

void LinearEncodingAttr::print(mlir::AsmPrinter &printer) const {
  // We don't use the default implementation as it's a bit too verbose
  // This prints in the following format that is shape agnostic, in the sense
  // that we don't print explicitly the outShape of the LL
  // We always assume LLs to be surjective
  // <{register = [[0, 1], [8, 0], [0, 8], [64, 0]],
  //   lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
  //   warp = [[16, 0], [32, 0]],
  //   block = []}>
  auto ll = getLinearLayout();
  printer << "<{" << join(ll.getBases(), ", ", [](const auto &base) {
    return base.first.str() + " = " + "[" +
           join(base.second, ", ",
                [](const std::vector<int32_t> &vec) {
                  return "[" + join(vec, ", ") + "]";
                }) +
           "]";
  }) << "}>";
}

Attribute LinearEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};

  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};

  if (parser.parseGreater().failed())
    return {};

  LinearLayout::BasesT bases;

  // Parse the basis names in order (the order is relevant)
  std::vector<std::string> inDimNames = {"register", "lane", "warp", "block"};

  for (const auto &inDimNameStr : inDimNames) {
    auto inDimName = StringAttr::get(parser.getContext(), inDimNameStr);
    Attribute value = dict.get(inDimName);

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
  LinearLayout linearLayout(std::move(bases), std::move(outDimNames));

  // Create and return the LinearEncodingAttr
  return parser.getChecked<LinearEncodingAttr>(parser.getContext(),
                                               std::move(linearLayout));
}

SmallVector<unsigned> basesPerDim(const LinearLayout::BasesT &namedBases,
                                  StringAttr dimName, size_t rank,
                                  bool skipBroadcast = true) {
  const auto &bases = namedBases.find(dimName)->second;

  if (bases.empty()) {
    return SmallVector<unsigned>(rank, 1);
  }

  SmallVector<unsigned> ret(rank, 1);
  auto nonZero = [](auto val) { return val != 0; };
  int nonZeroIdx = -1;
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
      assert(nonZeroIdx != -1);
      ret[nonZeroIdx] *= 2;
    }
  }
  return ret;
}

SmallVector<unsigned> basesPerDim(const LinearLayout &ll, StringAttr dimName,
                                  bool skipBroadcast = true) {
  auto shapeIter = ll.getOutDimSizes();
  auto rank = std::distance(shapeIter.begin(), shapeIter.end());
  return basesPerDim(ll.getBases(), dimName, rank, skipBroadcast);
}

SmallVector<unsigned> orderPerDim(const LinearLayout &ll, StringAttr dimName,
                                  ArrayRef<unsigned> defaultOrder) {
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
  return SmallVector<unsigned>(order.begin(), order.end());
}

// [Note. Divergence of methods wrt. legacy layouts]
// For smaller shapes where the CTATile is larger than the output
// tensor, some methods return different values than the legacy layouts. I think
// this is benign tho. An example: what is the the vector of `warpsPerCTA` if
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
  return basesPerDim(getLinearLayout(), StringAttr::get(getContext(), "block"),
                     /*skipBroadcast=*/false);
}
SmallVector<unsigned> LinearEncodingAttr::getCTAOrder() const {
  return orderPerDim(getLinearLayout(), StringAttr::get(getContext(), "block"),
                     getOrder());
}
SmallVector<unsigned> LinearEncodingAttr::getCTASplitNum() const {
  return basesPerDim(getLinearLayout(), StringAttr::get(getContext(), "block"));
}
SmallVector<unsigned> LinearEncodingAttr::getWarpsPerCTA() const {
  return basesPerDim(getLinearLayout(), StringAttr::get(getContext(), "warp"));
}
SmallVector<unsigned> LinearEncodingAttr::getWarpOrder() const {
  return orderPerDim(getLinearLayout(), StringAttr::get(getContext(), "warp"),
                     getOrder());
}
SmallVector<unsigned> LinearEncodingAttr::getThreadsPerWarp() const {
  return basesPerDim(getLinearLayout(), StringAttr::get(getContext(), "lane"));
}
SmallVector<unsigned> LinearEncodingAttr::getThreadOrder() const {
  return orderPerDim(getLinearLayout(), StringAttr::get(getContext(), "lane"),
                     getOrder());
}
SmallVector<unsigned> LinearEncodingAttr::getSizePerThread() const {
  auto rank = getRepOrder().size();
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
  auto &registers = bases[StringAttr::get(ctx, "register")];
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
  return basesPerDim(bases, kRegister, rank);
}

SmallVector<unsigned> LinearEncodingAttr::getOrder() const {
  auto rank = getLinearLayout().getNumOutDims();
  SmallVector<unsigned> order(rank);
  // Choose [rank-1, rank-2, ... 0] as the default order in case
  // there are dims that do not move in the register
  // This order is as good as any really
  std::iota(order.rbegin(), order.rend(), 0);

  return orderPerDim(getLinearLayout(),
                     StringAttr::get(getContext(), "register"), order);
}

std::optional<LinearLayout>
LinearEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
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
LinearEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape, Type) const {
  // We can relax this assert by calling toLinearLayout rather than
  // getLinearLayout
  SmallVector<int32_t> shapeVec(shape.begin(), shape.end());
  assert(shapeVec == llvm::to_vector(getLinearLayout().getOutDimSizes()));
  auto ll = getLinearLayout();
  return basesPerDim(ll, StringAttr::get(getContext(), "register"));
}

// Start of Selection
SmallVector<unsigned> LinearEncodingAttr::getContigPerThread() const {
  auto ll = getLinearLayout();
  const auto &regs =
      ll.getBases().find(StringAttr::get(getContext(), "register"))->second;
  auto order = getOrder();
  auto rank = order.size();

  SmallVector<unsigned> contig(rank, 1);
  auto regIt = regs.begin();
  for (unsigned dim : order) {
    std::vector<int32_t> basis(rank, 0);
    basis[dim] = 1;

    while (regIt != regs.end() && *regIt == basis) {
      contig[dim] *= 2;
      basis[dim] *= 2;
      ++regIt;
    }
  }
  return contig;
}

unsigned LinearEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                    Type eltTy) const {
  return product(getElemsPerThread(shape, eltTy));
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
                      /*rank=*/getWarpsPerCTA().size());

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

  unsigned versionMajor = 0;
  unsigned versionMinor = 0;
  SmallVector<unsigned> warpsPerCTA;
  SmallVector<unsigned> instrShape;
  bool isTransposed;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;

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
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/warpsPerCTA.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<AMDMfmaEncodingAttr>(
      parser.getContext(), versionMajor, versionMinor, warpsPerCTA,
      instrShape[0], instrShape[1], isTransposed, *CTALayout);
}

void AMDMfmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "versionMajor = " << getVersionMajor()                      //
          << ", versionMinor = " << getVersionMinor()                    //
          << ", warpsPerCTA = [" << ArrayRef(getWarpsPerCTA()) << "]"    //
          << ", instrShape = [" << ArrayRef{getMDim(), getNDim()} << "]" //
          << ", isTransposed = " << getIsTransposed();
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getWarpsPerCTA().size());
  printer << "}>";
}

LogicalResult
AMDMfmaEncodingAttr::verify(function_ref<mlir::InFlightDiagnostic()> emitError,
                            unsigned versionMajor, unsigned versionMinor,
                            llvm::ArrayRef<unsigned int> warpsPerCTA,
                            unsigned mDim, unsigned nDim, bool isTransposed,
                            mlir::triton::gpu::CTALayoutAttr) {
  if (!(versionMajor >= 0 && versionMajor <= 3)) {
    return emitError() << "major version must be in the [0, 3] range";
  }
  if (versionMinor != 0) {
    return emitError() << "minor version must be 0";
  }
  if (!((mDim == 32 && nDim == 32) || (mDim == 16 && nDim == 16))) {
    return emitError()
           << "(M, N) cases other than (32, 32) or (16, 16) unimplemented";
  }

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
  SmallVector<unsigned> warpsPerCTA;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "version") {
      if (parseUInt(parser, attr, version, "version").failed())
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
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/warpsPerCTA.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<AMDWmmaEncodingAttr>(parser.getContext(), version,
                                                warpsPerCTA, *CTALayout);
}

void AMDWmmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "version = " << getVersion() << ", warpsPerCTA = ["
          << ArrayRef(getWarpsPerCTA()) << "]";
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getWarpsPerCTA().size());
  printer << "}>";
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
  SmallVector<unsigned> order;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;
  bool hasLeadingOffset = false;

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
    } else if (attr.getName() == "hasLeadingOffset") {
      if (parseBool(parser, attr, hasLeadingOffset, "hasLeadingOffset")
              .failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/order.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<SharedEncodingAttr>(parser.getContext(), vec,
                                               perPhase, maxPhase, order,
                                               *CTALayout, hasLeadingOffset);
}

void SharedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "vec = " << getVec() //
          << ", perPhase = " << getPerPhase()
          << ", maxPhase = " << getMaxPhase() //
          << ", order = [" << getOrder() << "]";
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getOrder().size());
  printer << ", hasLeadingOffset = " << getHasLeadingOffset() << "}>";
}

//===----------------------------------------------------------------------===//
// Mfma encoding
//===----------------------------------------------------------------------===//
// TODO: there is a lot of common code with MmaEncoding here

SmallVector<unsigned> AMDMfmaEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> AMDMfmaEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> AMDMfmaEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}
SmallVector<unsigned> AMDMfmaEncodingAttr::getWarpsPerCTA() const {
  return SmallVector<unsigned>(getWarpsPerCTA__());
}
SmallVector<unsigned> AMDMfmaEncodingAttr::getWarpOrder() const {
  return ::getOrder(*this);
}
SmallVector<unsigned> AMDMfmaEncodingAttr::getThreadOrder() const {
  auto order = ::getOrder(*this);
  if (getIsTransposed())
    std::swap(order[0], order[1]);
  return order;
}
SmallVector<unsigned> AMDMfmaEncodingAttr::getThreadsPerWarp() const {
  unsigned rows, cols;
  auto rank = ::getOrder(*this).size();
  SmallVector<unsigned> res(rank, 1);
  if (getMDim() == 32) {
    cols = 2;
    rows = 32;
  } else {
    assert(getMDim() == 16);
    cols = 4;
    rows = 16;
  }
  if (getIsTransposed()) {
    res[rank - 1] = cols;
    res[rank - 2] = rows;
  } else {
    res[rank - 1] = rows;
    res[rank - 2] = cols;
  }
  return res;
}

SmallVector<unsigned> AMDMfmaEncodingAttr::getSizePerThread() const {
  unsigned rows, cols;
  auto rank = ::getOrder(*this).size();
  SmallVector<unsigned> res(rank, 1);
  if (getMDim() == 32) {
    rows = 16;
    cols = 1;
  } else if (getMDim() == 16) {
    rows = 4;
    cols = 1;
  } else
    llvm_unreachable("Unexpected mfma non-k dim");

  if (getIsTransposed()) {
    res[rank - 1] = rows;
    res[rank - 2] = cols;
  } else {
    res[rank - 1] = cols;
    res[rank - 2] = rows;
  }
  return res;
}

SmallVector<int64_t>
AMDMfmaEncodingAttr::getInstrShapeForOperand(int kWidth, int opIdx) const {
  unsigned mDim = getMDim();
  unsigned nDim = getNDim();
  assert((mDim == nDim) && (mDim == 32 || mDim == 16 || mDim == 4) ||
         (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));
  constexpr int warpSize = 64; // MFMA is always based on the 64-wide warps.
  int kGroups = -1;
  if (mDim == nDim)
    kGroups = warpSize / mDim;
  if (mDim == 64 && nDim == 4 || mDim == 4 && nDim == 64)
    kGroups = 1;
  int64_t kDim = kWidth * kGroups;
  if (opIdx == 0)
    return {mDim, kDim};
  else
    assert(opIdx == 1);
  return {kDim, nDim};
}

SmallVector<unsigned> AMDMfmaEncodingAttr::getRepOrder() const {
  auto rank = getWarpsPerCTA().size();
  return getMatrixOrder(rank, /*rowMajor*/ true);
}

SmallVector<unsigned>
AMDMfmaEncodingAttr::getRepOrderForOperand(int opIdx) const {
  auto rank = getWarpsPerCTA().size();
  return getOrderForDotOperand(opIdx, rank, /*kMajor*/ true);
}

SmallVector<int64_t>
AMDMfmaEncodingAttr::getRepForOperand(ArrayRef<int64_t> operandShape,
                                      int kWidth, int opIdx) const {
  auto operandTileShape = getInstrShapeForOperand(kWidth, opIdx);
  auto rank = operandShape.size();
  auto warpsPerCTA = getWarpsPerCTA();
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

unsigned AMDMfmaEncodingAttr::getTotalElemsPerThreadForOperand(
    ArrayRef<int64_t> shape, Type eltTy, int kWidth, int opIdx) const {
  auto rep = getRepForOperand(shape, kWidth, opIdx);
  return product(rep) * kWidth;
}

SmallVector<unsigned>
AMDMfmaEncodingAttr::getSizePerThreadForOperand(int kWidth, int opIdx) const {
  auto rank = getWarpsPerCTA().size();
  auto sizePerThread = SmallVector<unsigned>(rank, 1);
  if (opIdx == 0) {
    sizePerThread[rank - 2] = 1;
    sizePerThread[rank - 1] = kWidth;
  } else if (opIdx == 1) {
    sizePerThread[rank - 2] = kWidth;
    sizePerThread[rank - 1] = 1;
  } else {
    llvm::report_fatal_error("DotOperandEncodingAttr opIdx must be 0 or 1");
  }
  return sizePerThread;
}

//===----------------------------------------------------------------------===//
// Wmma encoding
//===----------------------------------------------------------------------===//

SmallVector<unsigned> AMDWmmaEncodingAttr::getRepOrder() const {
  auto rank = getWarpsPerCTA().size();
  return getMatrixOrder(rank, /*rowMajor*/ true);
}

SmallVector<unsigned>
AMDWmmaEncodingAttr::getRepOrderForOperand(int opIdx) const {
  auto rank = getWarpsPerCTA().size();
  return getOrderForDotOperand(opIdx, rank, /*kMajor*/ true);
}

SmallVector<unsigned> AMDWmmaEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getWarpsPerCTA() const {
  return SmallVector<unsigned>(getWarpsPerCTA__());
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getWarpOrder() const {
  return ::getOrder(*this);
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getThreadOrder() const {
  return ::getOrder(*this);
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getThreadsPerWarp() const {
  auto rank = getWarpsPerCTA().size();
  SmallVector<unsigned> threads(rank, 1);
  auto mnkInstr = getMNKDimPerInstr();
  threads[rank - 2] = mnkInstr[0] / getSizePerThread()[rank - 2];
  threads[rank - 1] = mnkInstr[1] / getSizePerThread()[rank - 1];
  return threads;
}

SmallVector<unsigned> AMDWmmaEncodingAttr::getSizePerThread() const {
  auto rank = getWarpsPerCTA().size();
  SmallVector<unsigned> sizePerThread(rank, 1);
  sizePerThread[rank - 2] = 8;
  sizePerThread[rank - 1] = 1;
  return sizePerThread;
}
SmallVector<unsigned>
AMDWmmaEncodingAttr::getSizePerThreadForOperand(int kWidth, int opIdx) const {
  auto rank = getWarpsPerCTA().size();
  SmallVector<unsigned> sizePerThread(rank, 1);
  auto numReplicated = getVersion() == 1 ? 2 : 1;
  auto elemsPerInstr = numReplicated * product(getElemsPerInstrForOperands()) /
                       product(getThreadsPerWarp());
  if (opIdx == 0) {
    sizePerThread[rank - 2] = 1;
    sizePerThread[rank - 1] = elemsPerInstr;
  } else if (opIdx == 1) {
    sizePerThread[rank - 2] = elemsPerInstr;
    sizePerThread[rank - 1] = 1;
  } else {
    llvm::report_fatal_error("DotOperandEncodingAttr opIdx must be 0 or 1");
  }
  return sizePerThread;
}

unsigned AMDWmmaEncodingAttr::getTotalElemsPerThreadForOperand(
    ArrayRef<int64_t> shape, Type eltTy, int kWidth, int opIdx) const {
  auto rep = getRepForOperand(shape, eltTy, kWidth, opIdx);
  return product(rep) * kWidth;
}

SmallVector<int64_t> AMDWmmaEncodingAttr::getElemsPerInstrForOperands() const {
  return {16, 16};
}

SmallVector<int64_t>
AMDWmmaEncodingAttr::getRepForOperand(ArrayRef<int64_t> operandShape,
                                      Type elemType, int kWidth,
                                      int opIdx) const {
  auto operandTileShape = getElemsPerInstrForOperands();
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

SmallVector<unsigned> AMDWmmaEncodingAttr::getMNKDimPerInstr() {
  // TODO: move magic numbers out of the code
  return {16, 16, 16};
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
  auto rank = getWarpsPerCTA().size();
  return getMatrixOrder(rank, /*rowMajor*/ true);
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getWarpsPerCTA() const {
  return SmallVector<unsigned>(getWarpsPerCTA__());
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getWarpOrder() const {
  auto rank = getWarpsPerCTA().size();
  // Hopper (wgmma) uses column-major as this is embeded in the instruction
  // For Ampere we can choose either row-major or column-major.
  // We choose row-major as the legacy path did so
  return getMatrixOrder(rank, /*rowMajor*/ !isHopper());
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getThreadsPerWarp() const {
  auto rank = getWarpsPerCTA().size();
  SmallVector<unsigned> res(rank, 1);
  if (isAmpere()) {
    res[rank - 2] = 8;
    res[rank - 1] = 4;
    return res;
  }
  if (isHopper()) {
    res[rank - 2] = 8;
    res[rank - 1] = 4;
    return res;
  }
  llvm::report_fatal_error(
      "getThreadsPerWarp not implemented for unknown Mma version ");
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getThreadOrder() const {
  auto rank = getWarpsPerCTA().size();
  return getMatrixOrder(rank, /*rowMajor*/ true);
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getSizePerThread() const {
  auto rank = getWarpsPerCTA().size();
  SmallVector<unsigned> res(rank, 1);
  if (isAmpere()) {
    res[rank - 2] = 2;
    res[rank - 1] = 2;
    return res;
  }
  if (isHopper()) {
    auto instrShape = getInstrShape();
    // WGMMA instructions have an order of [0, 1] with 4 warps, each with 8
    // unique thread ids (32 in a warp group) per column. It is 1 warp wide with
    // 4 unique thread ids in the row. So the size per thread is the instruction
    // size divided by the number of unique thread ids.
    return SmallVector<unsigned>{instrShape[0] * 4 / 32, instrShape[1] / 4};
  }
  llvm_unreachable("Unexpected mma version");
}

SmallVector<unsigned>
NvidiaMmaEncodingAttr::getRepOrderForOperand(int opIdx) const {
  auto rank = getWarpsPerCTA().size();
  return getOrderForDotOperand(opIdx, rank, /*kMajor*/ true);
}

SmallVector<int64_t>
NvidiaMmaEncodingAttr::getRepForOperand(ArrayRef<int64_t> shape, int bitwidth,
                                        int kWidth, int opIdx) const {
  assert(
      kWidth >= 32 / bitwidth &&
      "kWidth must be >= 32 / bitwidth for this function to be well-defined");
  auto rank = shape.size();
  // Broadcast long K
  auto warpsPerCTA = getWarpsPerCTA();
  auto kDim = opIdx == 0 ? rank - 1 : rank - 2;
  warpsPerCTA[kDim] = 1;

  SmallVector<int> tileSize;
  if (rank == 3) {
    tileSize.push_back(1);
  }
  if (opIdx == 0) {
    // m x k
    tileSize.push_back(16);
    tileSize.push_back(4 * 64 / bitwidth);
  } else {
    // k x n
    // Hopper path never uses the n value, since this method is only invoked
    // for in-RF (dotOpEnc) operands, but WGMMA only supports in A to be in RF
    // so it's fine if the n is incorrect here
    tileSize.push_back(4 * 64 / bitwidth);
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

SmallVector<unsigned>
NvidiaMmaEncodingAttr::getSizePerThreadForOperand(int kWidth, int opIdx) const {
  auto rank = getWarpsPerCTA().size();
  auto sizePerThread = SmallVector<unsigned>(rank, 1);
  if (opIdx == 0) {
    sizePerThread[rank - 2] = 2;
    sizePerThread[rank - 1] = 2 * kWidth;
  } else {
    assert(opIdx == 1);
    sizePerThread[rank - 2] = 2 * kWidth;
    sizePerThread[rank - 1] = 1;
  }
  return sizePerThread;
}

//===----------------------------------------------------------------------===//
// DotOperand Encoding
//===----------------------------------------------------------------------===//
SmallVector<unsigned> DotOperandEncodingAttr::getRepOrder() const {
  if (auto mma = mlir::dyn_cast<MmaEncodingTrait>(getParent())) {
    return mma.getRepOrderForOperand(getOpIdx());
  }
  llvm::report_fatal_error(
      "getRepOrder not implemented for DotOperandEncodingAttr");
  return {};
}

SmallVector<unsigned> DotOperandEncodingAttr::getThreadsPerWarp() const {
  auto parent = getParent();
  if (auto mma = mlir::dyn_cast<NvidiaMmaEncodingAttr>(parent)) {
    auto threadsPerWarp = mma.getThreadsPerWarp();
    auto rank = threadsPerWarp.size();
    if (getOpIdx() == 1)
      std::swap(threadsPerWarp[rank - 2], threadsPerWarp[rank - 1]);
    return threadsPerWarp;
  }
  llvm::report_fatal_error(
      "getThreadsPerWarp not implemented for DotOperandEncodingAttr");
}
SmallVector<unsigned> DotOperandEncodingAttr::getSizePerThread() const {
  auto parentLayout = getParent();
  assert(parentLayout && "DotOperandEncodingAttr must have a parent");
  if (auto parentMmaLayout = mlir::dyn_cast<MmaEncodingTrait>(parentLayout)) {
    return parentMmaLayout.getSizePerThreadForOperand(getKWidth(), getOpIdx());
  } else {
    llvm::report_fatal_error(
        "DotOperandEncodingAttr non-NvidiaMmaEncodingAttr parent not "
        "supported yet");
    return {};
  }
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
    } else if (auto sharedAttr = mlir::dyn_cast<SharedEncodingAttr>(attr)) {
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
                        Attribute &resultEncoding) const override {
    resultEncoding = SliceEncodingAttr::get(getDialect()->getContext(), axis,
                                            operandEncoding);
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
    // Note: inferFooOpEncoding should not crash if given invalid inputs, which
    // happens when someone creates invalid IR.  If we return failure() on
    // error, then MLIR will generate a helpful error message.

    auto invOrder = inversePermutation(order);
    SmallVector<unsigned> invOrderUnsigned(invOrder.begin(), invOrder.end());

    auto permuteCTALayout =
        [&](const CTALayoutAttr &layout) -> FailureOr<CTALayoutAttr> {
      auto n = order.size();
      if (layout.getCTAsPerCGA().size() != n ||
          layout.getCTASplitNum().size() != n ||
          layout.getCTAOrder().size() != n) {
        return failure();
      }

      return CTALayoutAttr::get(
          getDialect()->getContext(),
          applyPermutation(layout.getCTAsPerCGA(), order),
          applyPermutation(layout.getCTASplitNum(), order),
          applyPermutation(invOrderUnsigned, layout.getCTAOrder()));
    };

    if (auto enc = mlir::dyn_cast<SharedEncodingAttr>(operandEncoding)) {
      if (enc.getOrder().size() != order.size()) {
        return failure();
      }
      FailureOr<CTALayoutAttr> ctaLayout = permuteCTALayout(enc.getCTALayout());
      if (failed(ctaLayout)) {
        return failure();
      }
      resultEncoding = SharedEncodingAttr::get(
          getDialect()->getContext(), enc.getVec(), enc.getPerPhase(),
          enc.getMaxPhase(), applyPermutation(invOrderUnsigned, enc.getOrder()),
          *ctaLayout, enc.getHasLeadingOffset());
      return success();
    }

    if (auto enc = mlir::dyn_cast<BlockedEncodingAttr>(operandEncoding)) {
      auto n = order.size();
      if (enc.getSizePerThread().size() != n ||
          enc.getThreadsPerWarp().size() != n ||
          enc.getWarpsPerCTA().size() != n || enc.getOrder().size() != n) {
        return failure();
      }
      FailureOr<CTALayoutAttr> ctaLayout = permuteCTALayout(enc.getCTALayout());
      if (failed(ctaLayout)) {
        return failure();
      }
      resultEncoding = BlockedEncodingAttr::get(
          getDialect()->getContext(),
          applyPermutation(enc.getSizePerThread(), order),
          applyPermutation(enc.getThreadsPerWarp(), order),
          applyPermutation(enc.getWarpsPerCTA(), order),
          applyPermutation(invOrderUnsigned, enc.getOrder()), *ctaLayout);
      return success();
    }

    return failure(); // unhandled encoding
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
      if (!mlir::isa<SharedEncodingAttr>(operandEncoding) &&
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
    auto src = mlir::dyn_cast<BlockedEncodingAttr>(srcEnc);
    if (!src) {
      return emitOptionalError(
          loc, "Non-reordering reshape only supports BlockedEncoding");
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

    // Feature flag to disable this routine while it's relatively new.
    // TODO(jlebar): Remove this once we're confident in the code.
    if (triton::tools::getBoolEnv(
            "TRITON_DISABLE_RESHAPE_ENCODING_INFERENCE")) {
      return failure();
    }

    // Cowardly refuse to handle encodings with multiple CTAs.  CTAsPerCGA
    // should be like the other fields in blocked encoding, but I'm not sure how
    // to handle CTASplitNum.
    if (!all_of(src.getCTAsPerCGA(), [](int32_t x) { return x == 1; }) ||
        !all_of(src.getCTASplitNum(), [](int32_t x) { return x == 1; })) {
      return emitOptionalError(
          loc, "Non-reordering reshape does not currently support multi-CTA "
               "layouts other than the default layout.");
    }

    // Cowardly refuse to handle encodings where shape[dim] is not divisible by
    // sizePerThread[dim], threadsPerWarp[dim], and warpsPerCTA[dim].  (We make
    // an exception if the block is larger than the shape.)
    auto checkDivisibility = [&](StringRef name, ArrayRef<unsigned> subblock) {
      for (int dim = 0; dim < srcShape.size(); dim++) {
        if (srcShape[dim] >= subblock[dim] &&
            srcShape[dim] % subblock[dim] != 0) {
          return emitOptionalError(loc,
                                   "Can't do a non-reordering reshape because "
                                   "the size of dimension ",
                                   dim, " (", srcShape[dim], ")",
                                   " is not divisible by ", name, "[", dim, "]",
                                   " = ", subblock[dim]);
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
        return emitOptionalError(loc,
                                 "Cannot do a non-reordering reshape given "
                                 "this src encoding order.  Dimensions [",
                                 join(srcDims),
                                 "] must be physically consecutive.");
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
              return emitOptionalError(
                  loc,
                  "Invalid src encoding for non-reordering reshape.  Must use "
                  "up sizePerThread / threadsPerWarp / warpsPerCTA for "
                  "more-minor dimensions before more major-dims can use them.");
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
            return emitOptionalError(
                loc,
                "Invalid src encoding for non-reordering reshape.  Block "
                "size in dimension ",
                dim,
                " is larger than the shape that dimension, but this is only "
                "allowed for the most-major dimension of a reshape chunk");
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
  inferJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                      std::optional<Location> loc) const override {
    auto enc = mlir::dyn_cast<BlockedEncodingAttr>(srcEnc);
    if (!enc) {
      return emitOptionalError(loc,
                               "JoinOp can only operate on BlockedEncoding");
    }

    // JoinOp takes two tensors of shape AxBxC and generates a tensor of shape
    // AxBxCx2.  The encoding is the same as the input, but with 2 elems per
    // thread in the new dimension.  The new dimension is most-minor.
    auto append = [](ArrayRef<unsigned> vals, int val) {
      SmallVector<unsigned> ret(vals);
      ret.push_back(val);
      return ret;
    };
    auto appendMinorDim = [](ArrayRef<unsigned> order) {
      SmallVector<unsigned> ret(order);
      ret.insert(ret.begin(), ret.size());
      return ret;
    };
    dstEnc = BlockedEncodingAttr::get(
        enc.getContext(),                    //
        append(enc.getSizePerThread(), 2),   //
        append(enc.getThreadsPerWarp(), 1),  //
        append(enc.getWarpsPerCTA(), 1),     //
        appendMinorDim(enc.getOrder()),      //
        CTALayoutAttr::get(enc.getContext(), //
                           append(enc.getCTAsPerCGA(), 1),
                           append(enc.getCTASplitNum(), 1),
                           appendMinorDim(enc.getCTAOrder())));
    return success();
  }

  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       std::optional<Location> loc) const override {
    auto enc = mlir::dyn_cast<BlockedEncodingAttr>(srcEnc);
    if (!enc) {
      return emitOptionalError(loc,
                               "SplitOp can only operate on BlockedEncoding");
    }

    // SplitOp takes a tensor of shape AxBxCx2 and generates two tensors of
    // shape AxBxC.  The input must have 2 elements per thread in the last
    // dimension, which must be most-minor.  The result encoding is the same as
    // the input, but with the last dimension removed.
    if (enc.getSizePerThread().back() != 2) {
      return emitOptionalError(loc,
                               "SplitOp requires 2 elements per thread in the "
                               "last dimension of the input");
    }
    if (enc.getThreadsPerWarp().back() != 1 ||
        enc.getWarpsPerCTA().back() != 1 || enc.getCTAsPerCGA().back() != 1) {
      return emitOptionalError(
          loc, "SplitOp requires threadsPerWarp, warpsPerCTA, "
               "and CTAsPerCGA = 1 for the last dimension of the input");
    }
    if (enc.getCTALayout().getCTAsPerCGA().back() != 1) {
      return emitOptionalError(
          loc,
          "SplitOp requires the last dimension to be most-minor in CTAOrder");
    }
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
};

//===----------------------------------------------------------------------===//
// Canonicalizer
//===----------------------------------------------------------------------===//

// reshape(cvt) -> reshape
struct CanonicalizeConvertFromReshape
    : public mlir::OpRewritePattern<triton::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::ReshapeOp op,
                  PatternRewriter &rewriter) const override {
    auto convert = op.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return failure();
    if (isExpensiveView(convert.getSrc().getType(), op.getType()))
      return failure();
    if (!op.getAllowReorder() || op.getEfficientLayout())
      return failure();

    rewriter.replaceOpWithNewOp<triton::ReshapeOp>(
        op, op.getType(), convert.getSrc(), op.getAllowReorder());
    return mlir::success();
  }
};

// histogram(cvt) -> histogram
struct CanonicalizeConvertFromHistogram
    : public mlir::OpRewritePattern<triton::HistogramOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::HistogramOp op,
                  PatternRewriter &rewriter) const override {
    auto convert = op.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return failure();
    rewriter.replaceOpWithNewOp<triton::HistogramOp>(
        op, op->getResult(0).getType(), convert.getSrc());
    return mlir::success();
  }
};

// alloc(cvt) -> alloc
struct CanonicalizeConvertFromAlloc
    : public mlir::OpRewritePattern<triton::gpu::LocalAllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op,
                  PatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    auto convert = op.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return failure();
    rewriter.replaceOpWithNewOp<triton::gpu::LocalAllocOp>(
        op, op->getResult(0).getType(), convert.getSrc());
    return mlir::success();
  }
};

// local_store(cvt) -> local_store
struct CanonicalizeConvertFromLocalStore
    : public mlir::OpRewritePattern<triton::gpu::LocalStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op,
                  PatternRewriter &rewriter) const override {
    auto convert = op.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return failure();
    rewriter.replaceOpWithNewOp<triton::gpu::LocalStoreOp>(op, convert.getSrc(),
                                                           op.getDst());
    return mlir::success();
  }
};

struct CanonicalizeConvertFromSplit
    : public mlir::OpRewritePattern<triton::SplitOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::SplitOp op,
                  PatternRewriter &rewriter) const override {
    auto convert = op.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return failure();
    auto srcEncoding = convert.getSrc().getType().getEncoding();
    // Multiple source layout can give the same output layout, if the source
    // layout of the convert gives the same destination layout we can skip the
    // convert.
    auto dstEncoding = inferDstEncoding(op, srcEncoding);
    if (dstEncoding != op.getOutLHS().getType().getEncoding())
      return failure();
    rewriter.replaceOpWithNewOp<triton::SplitOp>(op, convert.getSrc());
    return mlir::success();
  }
};

struct CanonicalizeConvertFromConvert
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConvertLayoutOp op,
                  PatternRewriter &rewriter) const override {
    // Convert to the same layout is redundant.
    if (op->getResultTypes() == op->getOperandTypes()) {
      rewriter.replaceOp(op, op->getOperands());
      return success();
    }

    // We don't handle conversions to DotOperandEncodingAttr.  This is a
    // heuristic to accommodate fused attention.
    auto srcType = op.getSrc().getType();
    auto dstType = op.getType();
    if (mlir::isa<DotOperandEncodingAttr>(dstType.getEncoding()) &&
        mlir::isa<NvidiaMmaEncodingAttr>(srcType.getEncoding()))
      return failure();

    // for hopper MMAv3
    if (mlir::isa<SharedEncodingAttr>(dstType.getEncoding()) &&
        mlir::isa<NvidiaMmaEncodingAttr>(srcType.getEncoding()) &&
        llvm::any_of(op.getResult().getUsers(), [](Operation *dot) {
          return dot->hasTrait<OpTrait::DotLike>();
        })) {
      return failure();
    }

    Operation *arg = op.getSrc().getDefiningOp();
    if (!arg)
      return failure();

    // cvt(reshape) -> reshape
    if (auto reshape = dyn_cast<ReshapeOp>(arg)) {
      if (!reshape.getAllowReorder() || reshape.getEfficientLayout() ||
          isExpensiveView(reshape.getSrc().getType(), op.getType()))
        return failure();

      // In TritonGPUToLLVM phase, ViewOp is converted to unpacking and packing
      // operations, which requires the element type to match between unpacking
      // and packing. However, part of values with dot operand encoding will be
      // packed/unpacked as i32 elements instead of the underlying element type.
      // To avoid errors, skip this folding when either the operand or result
      // of view has a dot operand encoding.
      if (hasDotOperandEncoding(op->getOperand(0)) ||
          hasDotOperandEncoding(op->getResult(0)))
        return failure();

      rewriter.replaceOpWithNewOp<ReshapeOp>(op, op->getResult(0).getType(),
                                             reshape.getResult(),
                                             reshape.getAllowReorder());
      return success();
    }

    // cvt(histogram) -> histogram
    if (auto histogram = dyn_cast<HistogramOp>(arg)) {
      // For histogram ops the input and output layouts are independent, so we
      // can always fold convert into the histogram op.
      rewriter.replaceOpWithNewOp<HistogramOp>(op, op->getResult(0).getType(),
                                               histogram.getSrc());
      return success();
    }

    // cvt(local_load) -> local_load.
    if (auto sharedLoad = dyn_cast<LocalLoadOp>(arg)) {
      // Shared_load can load to any layout so we can always fold convert into
      // it.
      // We insert at the point of the original op as there could be ops with
      // memory side-effects between the LocalLoad op and the ConvertLayout op
      rewriter.setInsertionPoint(arg);
      rewriter.replaceOpWithNewOp<LocalLoadOp>(op, op->getResult(0).getType(),
                                               sharedLoad.getSrc());

      return success();
    }

    // cvt(cat) -> cat
    if (auto cat = dyn_cast<CatOp>(arg)) {
      if (isExpensiveCat(cat, op.getType().getEncoding()))
        return failure();

      rewriter.replaceOpWithNewOp<CatOp>(op, op->getResult(0).getType(),
                                         cat.getOperands());
      return success();
    }

    // cvt(cvt(x, type1), type2) -> cvt(x, type2)
    if (auto cvt = dyn_cast<ConvertLayoutOp>(arg)) {
      auto srcType = op.getSrc().getType();
      rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
          op, op->getResultTypes().front(), cvt.getSrc());
      return success();
    }

    // cvt(type1, splat(type2, x)) -> splat(type1, x)
    if (auto splat = dyn_cast<triton::SplatOp>(arg)) {
      rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op->getResultTypes(),
                                                   splat.getSrc());
      return success();
    }

    // cvt(type1, make_range(type2, x)) -> make_range(type1, x)
    if (auto range = dyn_cast<MakeRangeOp>(arg)) {
      rewriter.replaceOpWithNewOp<MakeRangeOp>(
          op, op->getResultTypes(), range.getStart(), range.getEnd());
      return success();
    }

    // cvt(type, constant) -> constant
    if (auto cst = llvm::dyn_cast<arith::ConstantOp>(arg))
      if (auto ret = dyn_cast<SplatElementsAttr>(cst.getValue())) {
        auto ty = cast<ShapedType>(op->getResultTypes().front());
        auto newRet =
            SplatElementsAttr::get(ty, ret.getSplatValue<Attribute>());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newRet);
        return success();
      }
    return failure();
  }
};

void ConvertLayoutOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<CanonicalizeConvertFromConvert>(context);
  patterns.add<CanonicalizeConvertFromReshape>(context);
  patterns.add<CanonicalizeConvertFromHistogram>(context);
  patterns.add<CanonicalizeConvertFromAlloc>(context);
  patterns.add<CanonicalizeConvertFromLocalStore>(context);
  patterns.add<CanonicalizeConvertFromSplit>(context);
}

// LocalAllocOp
void LocalAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  Operation *op = getOperation();
  // If allocation is immutable, mark it as no side effect allow things like
  // CSE, DCE to work in early compiler passes.
  // After the memory offset is computed, we attach the true side effect to the
  // op.
  if (!getType().getMutableMemory() && !op->hasAttr("allocation.offset"))
    return;
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       mlir::triton::gpu::SharedMemory::get());
  if (getSrc())
    effects.emplace_back(MemoryEffects::Write::get(),
                         getOperation()->getOpResult(0),
                         mlir::triton::gpu::SharedMemory::get());
}

OpFoldResult LocalAllocOp::fold(FoldAdaptor adaptor) {
  if (getType().getMutableMemory())
    return {};
  auto src = getSrc();
  if (!src)
    return {};
  auto localLoadOp = src.getDefiningOp<LocalLoadOp>();
  if (!localLoadOp)
    return {};
  auto loadSrc = localLoadOp.getSrc();
  if (loadSrc.getType() != getType())
    return {};
  return loadSrc;
}

LogicalResult LocalAllocOp::verify() {
  if (!getSrc()) {
    if (!getType().getMutableMemory())
      return emitError("uninitialized alloc must have a mutable memdesc type");
    return success();
  }
  auto srcTy = getSrc().getType();
  auto dstTy = getType();

  if (srcTy.getElementType() != dstTy.getElementType()) {
    return emitError("result element type must match desc element type");
  }
  return success();
}

// LocalLoadOp
void LocalLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// LocalStoreOp
LogicalResult LocalStoreOp::verify() {
  if (!getDst().getType().getMutableMemory())
    return emitOpError("Cannot store into immutable memory");
  return success();
}

void LocalStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// AsyncCopyGlobalToLocalOp
LogicalResult AsyncCopyGlobalToLocalOp::verify() {
  if (!getResult().getType().getMutableMemory())
    return emitOpError("Cannot store into immutable memory");
  return success();
}

void AsyncCopyGlobalToLocalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       mlir::triton::GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getResultMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

LogicalResult MemDescSubviewOp::verify() {
  auto srcTy = getSrc().getType();
  auto dstTy = getType();

  if (srcTy.getElementType() != dstTy.getElementType()) {
    return emitError("result element type must match desc element type");
  }
  if (getOffsets().size() != srcTy.getRank()) {
    return emitError("offsets must have the same rank as input");
  }
  if (srcTy.getRank() < dstTy.getRank()) {
    return emitError("result rank must be less than or equal to input rank");
  }
  auto rankDiff = srcTy.getRank() - dstTy.getRank();
  for (int i = 0; i < dstTy.getRank(); i++) {
    if (dstTy.getDimSize(i) > srcTy.getDimSize(i + rankDiff)) {
      return emitError(
                 "result shape cannot be larger than input shape at dimension ")
             << i;
    }
  }

  auto srcEnc = srcTy.getEncoding();
  auto dstEnc = dstTy.getEncoding();
  if (!!srcEnc != !!dstEnc) {
    return emitError("src and result must both have or not have an encoding");
  }

  if (!isa<SharedEncodingAttr>(srcEnc)) {
    return emitError("src encoding must be SharedEncodingAttr");
  }
  if (!isa<SharedEncodingAttr>(dstEnc)) {
    return emitError("result encoding must be SharedEncodingAttr");
  }

  // TODO(jlebar): Currently we generate illegal encodings, so we can't add a
  // verifier for them.  In particular, we use the same encoding for the src and
  // dst of a subview op, when the subview removes a dimension.  That generates
  // an illegal shared encoding (because the size of `order` doesn't match the
  // rank of the tensor), but it's not checked anywhere, and we believe the
  // resulting code ultimately works.

  return success();
}

// -- LocalAllocOp --

int32_t LocalAllocOp::getAlignmentOrDefault() {
  auto align = getAlignment();
  if (align) {
    return *align;
  }

  auto ty = getType();
  auto shapePerCTA = triton::gpu::getShapePerCTA(ty);
  auto bytes =
      product<int64_t>(shapePerCTA) * (ty.getElementTypeBitWidth() / 8);

  // XXX(Keren): magic numbers 256 and 1024
  // Software swizzling calculates phase based on offset, while hardware
  // swizzling do that based on physical address. Thus only by setting the
  // alignment to 1024 can ensure the correctness.
  return bytes > 256 ? 1024 : 8;
}

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

std::string getSharedLayoutStr(RankedTensorType tensorType,
                               bool useHWPointOfView) {
  auto layout = tensorType.getEncoding();
  if (!layout)
    return "";

  std::optional<LinearLayout> ll =
      triton::gpu::toLinearLayout(tensorType.getShape(), layout);
  if (!ll.has_value())
    llvm::report_fatal_error("Failed to convert layout to linear layout");

  StringAttr kOffset = StringAttr::get(tensorType.getContext(), "offset");
  StringAttr kBlock = StringAttr::get(tensorType.getContext(), "block");
  int64_t tensorSize = product(tensorType.getShape());
  unsigned numBlocks = getNumCTAs(layout);
  int32_t blockSize = tensorSize / numBlocks;

  // elementMapping is for the non-hw layout, offsetMapping for hw-layout
  std::vector<std::string> elementMapping(tensorSize);
  std::vector<std::string> offsetMapping;

  // Shared layouts are a mapping of (block, offset) --> (...)

  // We can just use a single int to index into elementMapping because
  // the 'swizzle' operation rearranges the indicies---and we want to keep it
  // that way
  int32_t idx = 0;
  // Enumerate all the offsets for each block
  for (int32_t block = 0; block < numBlocks; block++) {
    for (int32_t offset = 0; offset < blockSize; offset++) {
      SmallVector<std::pair<StringAttr, int32_t>> inputs = {
          {kBlock, block},
          {kOffset, offset},
      };

      SmallVector<std::pair<StringAttr, int32_t>> outputs = ll->apply(inputs);

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
        auto index = paddedString(outputs[i].second, tensorType.getDimSize(i));
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

  std::optional<LinearLayout> ll =
      triton::gpu::toLinearLayout(tensorType.getShape(), layout);
  if (!ll.has_value())
    llvm::report_fatal_error("Failed to convert layout to linear layout");
  int64_t tensorSize = product(tensorType.getShape());
  std::vector<std::string> elementMapping(tensorSize);
  std::vector<std::string> threadMapping;
  unsigned threadsPerWarp = ll->getInDimSize(kLane);
  unsigned numWarpsPerCTA = ll->getInDimSize(kWarp);
  unsigned numBlocks = ll->getInDimSize(kBlock);
  int numElementsPerThreads = ll->getInDimSize(kRegister);
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
              ll->apply(inputs);
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

std::string mlir::triton::gpu::getLayoutStr(RankedTensorType tensorType,
                                            bool useHWPointOfView) {
  auto layout = tensorType.getEncoding();

  // tensorType is needed later on (e.g., getDimSize(j)), so we still have to
  // pass it as a param
  if (auto sharedLayout = mlir::dyn_cast<SharedEncodingAttr>(layout)) {
    return getSharedLayoutStr(tensorType, useHWPointOfView);
  } else if (auto distributedLayout =
                 mlir::dyn_cast<DistributedEncodingTrait>(layout)) {
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
  addInterfaces<TritonGPUOpAsmInterface>();
  addInterfaces<TritonGPUInferLayoutInterface>();

  RankedTensorType::attachInterface<TensorModel>(*getContext());
  MemDescType::attachInterface<MemDescModel>(*getContext());
}

// verify TritonGPU ops
LogicalResult TritonGPUDialect::verifyOperationAttribute(Operation *op,
                                                         NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
