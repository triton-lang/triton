#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVM.h"
#include "../PassDetail.h"
#include "./DotHelpers.h"
#include "./Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/PtxAsmFormat.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPU.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include <memory>
#include <numeric>
#include <string>

using namespace mlir;
using namespace mlir::triton;
using ::mlir::LLVM::DotOpFMAConversionHelper;
using ::mlir::LLVM::DotOpMmaV1ConversionHelper;
using ::mlir::LLVM::DotOpMmaV2ConversionHelper;
using ::mlir::LLVM::getElementsFromStruct;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::LLVM::getStructFromElements;
using ::mlir::LLVM::MMA16816ConversionHelper;
using ::mlir::LLVM::SharedMemoryObject;
using ::mlir::LLVM::shflSync;
using ::mlir::LLVM::storeShared;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getThreadsPerCTA;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

namespace mlir {
namespace LLVM {

static StringRef getStructAttrsAttrName() { return "llvm.struct_attrs"; }

// A helper function for using printf in LLVM conversion.
void vprintf(StringRef msg, ValueRange args,
             ConversionPatternRewriter &rewriter);

void vprintf_array(Value thread, ArrayRef<Value> arr, std::string info,
                   std::string elem_repr, ConversionPatternRewriter &builder);

} // namespace LLVM
} // namespace mlir

namespace {

namespace type = mlir::triton::type;

class TritonGPUToLLVMTypeConverter;

// TODO[goostavz]: Remove these methods after we have better debug log utilities
template <typename T>
void printArray(ArrayRef<T> array, const std::string &info) {
  std::cout << info << ": ";
  for (const T &e : array)
    std::cout << e << ",";
  std::cout << std::endl;
}
template <typename T> void printScalar(const T &e, const std::string &info) {
  std::cout << info << ": " << e << std::endl;
}

// FuncOpConversion/FuncOpConversionBase is borrowed from
// https://github.com/llvm/llvm-project/blob/fae656b2dd80246c3c6f01e9c77c49560368752c/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp#L276
// since it is not exposed on header files in mlir v14
// TODO(Superjomn) Remove the code when mlir v15.0 is included.
// All the rights are reserved by LLVM community.

/// Only retain those attributes that are not constructed by
/// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
/// attributes.
void filterFuncAttributes(ArrayRef<NamedAttribute> attrs, bool filterArgAttrs,
                          SmallVectorImpl<NamedAttribute> &result) {
  for (const auto &attr : attrs) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == FunctionOpInterface::getTypeAttrName() ||
        attr.getName() == "std.varargs" ||
        (filterArgAttrs &&
         attr.getName() == FunctionOpInterface::getArgDictAttrName()))
      continue;
    result.push_back(attr);
  }
}

/// Helper function for wrapping all attributes into a single DictionaryAttr
auto wrapAsStructAttrs(OpBuilder &b, ArrayAttr attrs) {
  return DictionaryAttr::get(
      b.getContext(), b.getNamedAttr(LLVM::getStructAttrsAttrName(), attrs));
}

struct FuncOpConversionBase : public ConvertOpToLLVMPattern<FuncOp> {
protected:
  using ConvertOpToLLVMPattern<FuncOp>::ConvertOpToLLVMPattern;

  // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
  // to this legalization pattern.
  LLVM::LLVMFuncOp
  convertFuncOpToLLVMFuncOp(FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) const {
    // Convert the original function arguments. They are converted using the
    // LLVMTypeConverter provided to this legalization pattern.
    auto varargsAttr = funcOp->getAttrOfType<BoolAttr>("func.varargs");
    TypeConverter::SignatureConversion result(funcOp.getNumArguments());
    auto llvmType = getTypeConverter()->convertFunctionSignature(
        funcOp.getType(), varargsAttr && varargsAttr.getValue(), result);
    if (!llvmType)
      return nullptr;

    // Propagate argument/result attributes to all converted arguments/result
    // obtained after converting a given original argument/result.
    SmallVector<NamedAttribute, 4> attributes;
    filterFuncAttributes(funcOp->getAttrs(), /*filterArgAndResAttrs=*/true,
                         attributes);
    if (ArrayAttr resAttrDicts = funcOp.getAllResultAttrs()) {
      assert(!resAttrDicts.empty() && "expected array to be non-empty");
      auto newResAttrDicts =
          (funcOp.getNumResults() == 1)
              ? resAttrDicts
              : rewriter.getArrayAttr(
                    {wrapAsStructAttrs(rewriter, resAttrDicts)});
      attributes.push_back(rewriter.getNamedAttr(
          FunctionOpInterface::getResultDictAttrName(), newResAttrDicts));
    }
    if (ArrayAttr argAttrDicts = funcOp.getAllArgAttrs()) {
      SmallVector<Attribute, 4> newArgAttrs(
          llvmType.cast<LLVM::LLVMFunctionType>().getNumParams());
      for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
        auto mapping = result.getInputMapping(i);
        assert(mapping && "unexpected deletion of function argument");
        for (size_t j = 0; j < mapping->size; ++j)
          newArgAttrs[mapping->inputNo + j] = argAttrDicts[i];
      }
      attributes.push_back(
          rewriter.getNamedAttr(FunctionOpInterface::getArgDictAttrName(),
                                rewriter.getArrayAttr(newArgAttrs)));
    }
    for (const auto &pair : llvm::enumerate(attributes)) {
      if (pair.value().getName() == "llvm.linkage") {
        attributes.erase(attributes.begin() + pair.index());
        break;
      }
    }

    // Create an LLVM function, use external linkage by default until MLIR
    // functions have linkage.
    LLVM::Linkage linkage = LLVM::Linkage::External;
    if (funcOp->hasAttr("llvm.linkage")) {
      auto attr =
          funcOp->getAttr("llvm.linkage").dyn_cast<mlir::LLVM::LinkageAttr>();
      if (!attr) {
        funcOp->emitError()
            << "Contains llvm.linkage attribute not of type LLVM::LinkageAttr";
        return nullptr;
      }
      linkage = attr.getLinkage();
    }
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), llvmType, linkage,
        /*dsoLocal*/ false, attributes);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &result)))
      return nullptr;

    return newFuncOp;
  }
};

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(LLVMTypeConverter &converter, int numWarps,
                   PatternBenefit benefit)
      : FuncOpConversionBase(converter, benefit), NumWarps(numWarps) {}

  LogicalResult
  matchAndRewrite(FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (!newFuncOp)
      return failure();

    auto ctx = funcOp->getContext();

    // Set an attribute to indicate this function is a kernel entry.
    newFuncOp->setAttr(NVVMMetadataField::Kernel,
                       rewriter.getIntegerAttr(type::u1Ty(ctx), 1));

    // Set an attribute for maxntidx, it could be used in latter LLVM codegen
    // for `nvvm.annotation` metadata.
    newFuncOp->setAttr(NVVMMetadataField::MaxNTid,
                       rewriter.getIntegerAttr(i32_ty, 32 * NumWarps));

    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  int NumWarps{0};
};

struct ReturnOpConversion : public ConvertOpToLLVMPattern<::mlir::ReturnOp> {
  using ConvertOpToLLVMPattern<ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op.getNumOperands();

    // Currently, Triton kernel function always return nothing.
    // TODO(Superjomn) add support for non-inline device function
    if (numArguments > 0) {
      return rewriter.notifyMatchFailure(
          op, "Only kernel function with nothing returned is supported.");
    }

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                op->getAttrs());
    return success();
  }
};

// delinearize supposing order is [0, 1, .. , n]
template <typename T>
SmallVector<T> getMultiDimIndexImpl(T linearIndex, ArrayRef<T> shape) {
  // shape: {a, b, c, d}  ->  accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearRemain = linearIndex;
  SmallVector<T> multiDimIndex(rank);
  for (int i = rank - 1; i >= 0; --i) {
    multiDimIndex[i] = linearRemain / accMul;
    linearRemain = linearRemain % accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return multiDimIndex;
}

template <typename T>
SmallVector<T> getMultiDimIndex(T linearIndex, ArrayRef<T> shape,
                                ArrayRef<unsigned> order) {
  size_t rank = shape.size();
  assert(rank == order.size());
  auto reordered = reorder(shape, order);
  auto reorderedMultiDim = getMultiDimIndexImpl<T>(linearIndex, reordered);
  SmallVector<T> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

// linearize supposing order is [0, 1, .. , n]
template <typename T>
T getLinearIndexImpl(ArrayRef<T> multiDimIndex, ArrayRef<T> shape) {
  assert(multiDimIndex.size() == shape.size());
  // shape: {a, b, c, d}  ->  accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearIndex = 0;
  for (int i = rank - 1; i >= 0; --i) {
    linearIndex += multiDimIndex[i] * accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return linearIndex;
}

template <typename T>
T getLinearIndex(ArrayRef<T> multiDimIndex, ArrayRef<T> shape,
                 ArrayRef<unsigned> order) {
  assert(shape.size() == order.size());
  return getLinearIndexImpl<T>(reorder(multiDimIndex, order),
                               reorder(shape, order));
}

struct ConvertTritonGPUOpToLLVMPatternBase {
  static Value
  getStructFromSharedMemoryObject(Location loc,
                                  const SharedMemoryObject &smemObj,
                                  ConversionPatternRewriter &rewriter) {
    auto elems = smemObj.getElems();
    auto types = smemObj.getTypes();
    auto structTy =
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
    return getStructFromElements(loc, elems, rewriter, structTy);
  }
};

template <typename SourceOp>
class ConvertTritonGPUOpToLLVMPattern
    : public ConvertOpToLLVMPattern<SourceOp>,
      public ConvertTritonGPUOpToLLVMPatternBase {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ConvertTritonGPUOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                           PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit) {}

  explicit ConvertTritonGPUOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                           const Allocation *allocation,
                                           Value smem,
                                           PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        allocation(allocation), smem(smem) {}

  Value getThreadId(ConversionPatternRewriter &rewriter, Location loc) const {
    auto llvmIndexTy = this->getTypeConverter()->getIndexType();
    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{llvmIndexTy},
        ValueRange{rewriter.create<::mlir::gpu::ThreadIdOp>(
            loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x)});
    Value threadId = cast.getResult(0);
    return threadId;
  }

  Value createIndexConst(ConversionPatternRewriter &rewriter, Location loc,
                         int64_t value) const {
    return rewriter.create<LLVM::ConstantOp>(
        loc, this->getTypeConverter()->getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), value));
  }

  // -----------------------------------------------------------------------
  // Utilities
  // -----------------------------------------------------------------------

  // Convert an \param index to a multi-dim coordinate given \param shape and
  // \param order.
  SmallVector<Value> delinearize(ConversionPatternRewriter &rewriter,
                                 Location loc, Value linear,
                                 ArrayRef<unsigned> shape,
                                 ArrayRef<unsigned> order) const {
    unsigned rank = shape.size();
    assert(rank == order.size());
    auto reordered = reorder(shape, order);
    auto reorderedMultiDim = delinearize(rewriter, loc, linear, reordered);
    SmallVector<Value> multiDim(rank);
    for (unsigned i = 0; i < rank; ++i) {
      multiDim[order[i]] = reorderedMultiDim[i];
    }
    return multiDim;
  }

  SmallVector<Value> delinearize(ConversionPatternRewriter &rewriter,
                                 Location loc, Value linear,
                                 ArrayRef<unsigned> shape) const {
    unsigned rank = shape.size();
    assert(rank > 0);
    SmallVector<Value> multiDim(rank);
    if (rank == 1) {
      multiDim[0] = linear;
    } else {
      Value remained = linear;
      for (auto &&en : llvm::enumerate(shape.drop_back())) {
        Value dimSize = idx_val(en.value());
        multiDim[en.index()] = urem(remained, dimSize);
        remained = udiv(remained, dimSize);
      }
      multiDim[rank - 1] = remained;
    }
    return multiDim;
  }

  Value linearize(ConversionPatternRewriter &rewriter, Location loc,
                  ArrayRef<Value> multiDim, ArrayRef<unsigned> shape,
                  ArrayRef<unsigned> order) const {
    return linearize(rewriter, loc, reorder<Value>(multiDim, order),
                     reorder<unsigned>(shape, order));
  }

  Value linearize(ConversionPatternRewriter &rewriter, Location loc,
                  ArrayRef<Value> multiDim, ArrayRef<unsigned> shape) const {
    int rank = multiDim.size();
    Value linear = idx_val(0);
    if (rank > 0) {
      linear = multiDim.back();
      for (auto [dim, shape] :
           llvm::reverse(llvm::zip(multiDim.drop_back(), shape.drop_back()))) {
        Value dimSize = idx_val(shape);
        linear = add(mul(linear, dimSize), dim);
      }
    }
    return linear;
  }

  Value dot(ConversionPatternRewriter &rewriter, Location loc,
            ArrayRef<Value> offsets, ArrayRef<Value> strides) const {
    assert(offsets.size() == strides.size());
    Value ret = idx_val(0);
    for (auto [offset, stride] : llvm::zip(offsets, strides)) {
      ret = add(ret, mul(offset, stride));
    }
    return ret;
  }

  // -----------------------------------------------------------------------
  // Blocked layout indices
  // -----------------------------------------------------------------------

  // Get an index-base for each dimension for a \param blocked_layout.
  SmallVector<Value>
  emitBaseIndexForBlockedLayout(Location loc,
                                ConversionPatternRewriter &rewriter,
                                const BlockedEncodingAttr &blocked_layout,
                                ArrayRef<int64_t> shape) const {
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = idx_val(32);
    Value laneId = urem(threadId, warpSize);
    Value warpId = udiv(threadId, warpSize);
    auto sizePerThread = blocked_layout.getSizePerThread();
    auto threadsPerWarp = blocked_layout.getThreadsPerWarp();
    auto warpsPerCTA = blocked_layout.getWarpsPerCTA();
    auto order = blocked_layout.getOrder();
    unsigned rank = shape.size();

    // delinearize threadId to get the base index
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, order);
    SmallVector<Value> multiDimThreadId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);

    SmallVector<Value> multiDimBase(rank);
    for (unsigned k = 0; k < rank; ++k) {
      // Wrap around multiDimWarpId/multiDimThreadId incase
      // shape[k] > shapePerCTA[k]
      unsigned maxWarps =
          ceil<unsigned>(shape[k], sizePerThread[k] * threadsPerWarp[k]);
      unsigned maxThreads = ceil<unsigned>(shape[k], sizePerThread[k]);
      multiDimWarpId[k] = urem(multiDimWarpId[k], idx_val(maxWarps));
      multiDimThreadId[k] = urem(multiDimThreadId[k], idx_val(maxThreads));
      // multiDimBase[k] = (multiDimThreadId[k] +
      //                    multiDimWarpId[k] * threadsPerWarp[k]) *
      //                   sizePerThread[k];
      Value threadsPerWarpK = idx_val(threadsPerWarp[k]);
      Value sizePerThreadK = idx_val(sizePerThread[k]);
      multiDimBase[k] =
          mul(sizePerThreadK, add(multiDimThreadId[k],
                                  mul(multiDimWarpId[k], threadsPerWarpK)));
    }
    return multiDimBase;
  }

  SmallVector<SmallVector<unsigned>>
  emitOffsetForBlockedLayout(const BlockedEncodingAttr &blockedLayout,
                             ArrayRef<int64_t> shape) const {
    auto sizePerThread = blockedLayout.getSizePerThread();
    auto threadsPerWarp = blockedLayout.getThreadsPerWarp();
    auto warpsPerCTA = blockedLayout.getWarpsPerCTA();
    auto order = blockedLayout.getOrder();

    unsigned rank = shape.size();
    SmallVector<unsigned> shapePerCTA = getShapePerCTA(blockedLayout);
    SmallVector<unsigned> tilesPerDim(rank);
    for (unsigned k = 0; k < rank; ++k)
      tilesPerDim[k] = ceil<unsigned>(shape[k], shapePerCTA[k]);

    SmallVector<SmallVector<unsigned>> offset(rank);
    for (unsigned k = 0; k < rank; ++k) {
      // 1 block in minimum if shape[k] is less than shapePerCTA[k]
      for (unsigned blockOffset = 0; blockOffset < tilesPerDim[k];
           ++blockOffset)
        for (unsigned warpOffset = 0; warpOffset < warpsPerCTA[k]; ++warpOffset)
          for (unsigned threadOffset = 0; threadOffset < threadsPerWarp[k];
               ++threadOffset)
            for (unsigned elemOffset = 0; elemOffset < sizePerThread[k];
                 ++elemOffset)
              offset[k].push_back(blockOffset * sizePerThread[k] *
                                      threadsPerWarp[k] * warpsPerCTA[k] +
                                  warpOffset * sizePerThread[k] *
                                      threadsPerWarp[k] +
                                  threadOffset * sizePerThread[k] + elemOffset);
    }

    unsigned elemsPerThread = blockedLayout.getElemsPerThread(shape);
    unsigned totalSizePerThread = product<unsigned>(sizePerThread);
    SmallVector<SmallVector<unsigned>> reorderedOffset(elemsPerThread);
    for (unsigned n = 0; n < elemsPerThread; ++n) {
      unsigned linearNanoTileId = n / totalSizePerThread;
      unsigned linearNanoTileElemId = n % totalSizePerThread;
      SmallVector<unsigned> multiDimNanoTileId =
          getMultiDimIndex<unsigned>(linearNanoTileId, tilesPerDim, order);
      SmallVector<unsigned> multiDimNanoTileElemId = getMultiDimIndex<unsigned>(
          linearNanoTileElemId, sizePerThread, order);
      for (unsigned k = 0; k < rank; ++k) {
        unsigned reorderedMultiDimId =
            multiDimNanoTileId[k] *
                (sizePerThread[k] * threadsPerWarp[k] * warpsPerCTA[k]) +
            multiDimNanoTileElemId[k];
        reorderedOffset[n].push_back(offset[k][reorderedMultiDimId]);
      }
    }
    return reorderedOffset;
  }

  // -----------------------------------------------------------------------
  // Mma layout indices
  // -----------------------------------------------------------------------

  SmallVector<Value>
  emitBaseIndexForMmaLayoutV1(Location loc, ConversionPatternRewriter &rewriter,
                              const MmaEncodingAttr &mmaLayout,
                              ArrayRef<int64_t> shape) const {
    llvm_unreachable("emitIndicesForMmaLayoutV1 not implemented");
  }

  SmallVector<SmallVector<unsigned>>
  emitOffsetForMmaLayoutV1(const MmaEncodingAttr &mmaLayout,
                           ArrayRef<int64_t> shape) const {
    SmallVector<SmallVector<unsigned>> ret;

    for (unsigned i = 0; i < shape[0]; i += getShapePerCTA(mmaLayout)[0]) {
      for (unsigned j = 0; j < shape[1]; j += getShapePerCTA(mmaLayout)[1]) {
        ret.push_back({i, j});
        ret.push_back({i, j + 1});
        ret.push_back({i + 2, j});
        ret.push_back({i + 2, j + 1});
        ret.push_back({i, j + 8});
        ret.push_back({i, j + 9});
        ret.push_back({i + 2, j + 8});
        ret.push_back({i + 2, j + 9});
      }
    }
    return ret;
  }

  SmallVector<Value>
  emitBaseIndexForMmaLayoutV2(Location loc, ConversionPatternRewriter &rewriter,
                              const MmaEncodingAttr &mmaLayout,
                              ArrayRef<int64_t> shape) const {
    auto _warpsPerCTA = mmaLayout.getWarpsPerCTA();
    assert(_warpsPerCTA.size() == 2);
    SmallVector<Value> warpsPerCTA = {idx_val(_warpsPerCTA[0]),
                                      idx_val(_warpsPerCTA[1])};
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = idx_val(32);
    Value laneId = urem(threadId, warpSize);
    Value warpId = udiv(threadId, warpSize);
    Value warpId0 = urem(warpId, warpsPerCTA[0]);
    Value warpId1 = urem(udiv(warpId, warpsPerCTA[0]), warpsPerCTA[1]);
    Value offWarp0 = mul(warpId0, idx_val(16));
    Value offWarp1 = mul(warpId1, idx_val(8));

    SmallVector<Value> multiDimBase(2);
    multiDimBase[0] = add(udiv(laneId, idx_val(4)), offWarp0);
    multiDimBase[1] = add(mul(idx_val(2), urem(laneId, idx_val(4))), offWarp1);
    return multiDimBase;
  }

  SmallVector<SmallVector<unsigned>>
  emitOffsetForMmaLayoutV2(const MmaEncodingAttr &mmaLayout,
                           ArrayRef<int64_t> shape) const {
    SmallVector<SmallVector<unsigned>> ret;

    for (unsigned i = 0; i < shape[0]; i += getShapePerCTA(mmaLayout)[0]) {
      for (unsigned j = 0; j < shape[1]; j += getShapePerCTA(mmaLayout)[1]) {
        ret.push_back({i, j});
        ret.push_back({i, j + 1});
        ret.push_back({i + 8, j});
        ret.push_back({i + 8, j + 1});
      }
    }
    return ret;
  }

  // -----------------------------------------------------------------------
  // Get offsets / indices for any layout
  // -----------------------------------------------------------------------

  SmallVector<Value> emitBaseIndexForLayout(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const Attribute &layout,
                                            ArrayRef<int64_t> shape) const {
    if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>())
      return emitBaseIndexForBlockedLayout(loc, rewriter, blockedLayout, shape);
    if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
      if (mmaLayout.isVolta())
        return emitBaseIndexForMmaLayoutV1(loc, rewriter, mmaLayout, shape);
      if (mmaLayout.isAmpere())
        return emitBaseIndexForMmaLayoutV2(loc, rewriter, mmaLayout, shape);
    }
    llvm_unreachable("unsupported emitBaseIndexForLayout");
  }

  SmallVector<SmallVector<unsigned>>
  emitOffsetForLayout(const Attribute &layout, ArrayRef<int64_t> shape) const {
    if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>())
      return emitOffsetForBlockedLayout(blockedLayout, shape);
    if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
      if (mmaLayout.isVolta())
        return emitOffsetForMmaLayoutV1(mmaLayout, shape);
      if (mmaLayout.isAmpere())
        return emitOffsetForMmaLayoutV2(mmaLayout, shape);
    }
    llvm_unreachable("unsupported emitOffsetForLayout");
  }

  // Emit indices calculation within each ConversionPattern, and returns a
  // [elemsPerThread X rank] index matrix.

  // TODO: [phil] redundant indices commputation do not appear to hurt
  // performance much, but they could still significantly slow down
  // computations.
  SmallVector<SmallVector<Value>> emitIndicesForDistributedLayout(
      Location loc, ConversionPatternRewriter &rewriter,
      const Attribute &layout, ArrayRef<int64_t> shape) const {

    // step 1, delinearize threadId to get the base index
    auto multiDimBase = emitBaseIndexForLayout(loc, rewriter, layout, shape);
    // step 2, get offset of each element
    auto offset = emitOffsetForLayout(layout, shape);
    // step 3, add offset to base, and reorder the sequence of indices to
    // guarantee that elems in the same sizePerThread are adjacent in order
    unsigned rank = shape.size();
    unsigned elemsPerThread = offset.size();
    SmallVector<SmallVector<Value>> multiDimIdx(elemsPerThread,
                                                SmallVector<Value>(rank));
    for (unsigned n = 0; n < elemsPerThread; ++n)
      for (unsigned k = 0; k < rank; ++k)
        multiDimIdx[n][k] = add(multiDimBase[k], idx_val(offset[n][k]));

    return multiDimIdx;
  }

  struct SmallVectorKeyInfo {
    static unsigned getHashValue(const SmallVector<unsigned> &key) {
      return llvm::hash_combine_range(key.begin(), key.end());
    }
    static bool isEqual(const SmallVector<unsigned> &lhs,
                        const SmallVector<unsigned> &rhs) {
      return lhs == rhs;
    }
    static SmallVector<unsigned> getEmptyKey() {
      return SmallVector<unsigned>();
    }
    static SmallVector<unsigned> getTombstoneKey() {
      return {std::numeric_limits<unsigned>::max()};
    }
  };

  SmallVector<SmallVector<Value>>
  emitIndicesForSliceLayout(Location loc, ConversionPatternRewriter &rewriter,
                            const SliceEncodingAttr &sliceLayout,
                            ArrayRef<int64_t> shape) const {
    auto parent = sliceLayout.getParent();
    unsigned dim = sliceLayout.getDim();
    size_t rank = shape.size();
    auto parentIndices =
        emitIndices(loc, rewriter, parent, sliceLayout.paddedShape(shape));
    unsigned numIndices = parentIndices.size();
    SmallVector<SmallVector<Value>> resultIndices;
    for (unsigned i = 0; i < numIndices; ++i) {
      SmallVector<Value> indices = parentIndices[i];
      indices.erase(indices.begin() + dim);
      resultIndices.push_back(indices);
    }
    return resultIndices;
  }

  // -----------------------------------------------------------------------
  // Emit indices
  // -----------------------------------------------------------------------
  SmallVector<SmallVector<Value>> emitIndices(Location loc,
                                              ConversionPatternRewriter &b,
                                              const Attribute &layout,
                                              ArrayRef<int64_t> shape) const {
    if (auto blocked = layout.dyn_cast<BlockedEncodingAttr>()) {
      return emitIndicesForDistributedLayout(loc, b, blocked, shape);
    } else if (auto mma = layout.dyn_cast<MmaEncodingAttr>()) {
      return emitIndicesForDistributedLayout(loc, b, mma, shape);
    } else if (auto slice = layout.dyn_cast<SliceEncodingAttr>()) {
      return emitIndicesForSliceLayout(loc, b, slice, shape);
    } else {
      assert(0 && "emitIndices for layouts other than blocked & slice not "
                  "implemented yet");
      return {};
    }
  }

  // -----------------------------------------------------------------------
  // Shared memory utilities
  // -----------------------------------------------------------------------

  template <typename T>
  Value getSharedMemoryBase(Location loc, ConversionPatternRewriter &rewriter,
                            T value) const {
    auto ptrTy = LLVM::LLVMPointerType::get(
        this->getTypeConverter()->convertType(rewriter.getI8Type()), 3);
    auto bufferId = allocation->getBufferId(value);
    assert(bufferId != Allocation::InvalidBufferId && "BufferId not found");
    size_t offset = allocation->getOffset(bufferId);
    Value offVal = idx_val(offset);
    Value base = gep(ptrTy, smem, offVal);
    return base;
  }

protected:
  const Allocation *allocation;
  Value smem;
};

Value convertSplatLikeOpWithMmaLayout(const MmaEncodingAttr &layout,
                                      Type resType, Type elemType,
                                      Value constVal,
                                      TypeConverter *typeConverter,
                                      ConversionPatternRewriter &rewriter,
                                      Location loc);

// Convert SplatOp or arith::ConstantOp with SplatElementsAttr to a
// LLVM::StructType value.
//
// @elemType: the element type in operand.
// @resType: the return type of the Splat-like op.
// @constVal: a LLVM::ConstantOp or other scalar value.
Value convertSplatLikeOp(Type elemType, Type resType, Value constVal,
                         TypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc) {
  auto tensorTy = resType.cast<RankedTensorType>();
  if (tensorTy.getEncoding().isa<BlockedEncodingAttr>() ||
      tensorTy.getEncoding().isa<SliceEncodingAttr>()) {
    auto srcType = typeConverter->convertType(elemType);
    auto llSrc = bitcast(constVal, srcType);
    size_t elemsPerThread = getElemsPerThread(tensorTy);
    llvm::SmallVector<Value> elems(elemsPerThread, llSrc);
    llvm::SmallVector<Type> elemTypes(elems.size(), srcType);
    auto structTy =
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(), elemTypes);

    return getStructFromElements(loc, elems, rewriter, structTy);
  } else if (auto mmaLayout =
                 tensorTy.getEncoding().dyn_cast<MmaEncodingAttr>()) {
    return convertSplatLikeOpWithMmaLayout(
        mmaLayout, resType, elemType, constVal, typeConverter, rewriter, loc);
  } else
    assert(false && "Unsupported layout found in ConvertSplatLikeOp");

  return Value{};
}

struct SplatOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::SplatOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::SplatOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto src = adaptor.src();
    auto llStruct = convertSplatLikeOp(src.getType(), op.getType(), src,
                                       getTypeConverter(), rewriter, loc);
    rewriter.replaceOp(op, {llStruct});
    return success();
  }
};

// This pattern helps to convert arith::ConstantOp(with SplatElementsAttr),
// the logic is the same as triton::SplatOp, so the underlying implementation
// is reused.
struct ArithConstantSplatOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<arith::ConstantOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      arith::ConstantOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = op.getValue();
    if (!value.dyn_cast<SplatElementsAttr>())
      return failure();

    auto loc = op->getLoc();

    LLVM::ConstantOp arithConstantOp;
    auto values = op.getValue().dyn_cast<SplatElementsAttr>();
    auto elemType = values.getElementType();

    Attribute val;
    if (type::isInt(elemType)) {
      val = values.getValues<IntegerAttr>()[0];
    } else if (type::isFloat(elemType)) {
      val = values.getValues<FloatAttr>()[0];
    } else {
      llvm::errs() << "ArithConstantSplatOpConversion get unsupported type: "
                   << value.getType() << "\n";
      return failure();
    }

    auto constOp = rewriter.create<LLVM::ConstantOp>(loc, elemType, val);
    auto llStruct = convertSplatLikeOp(elemType, op.getType(), constOp,
                                       getTypeConverter(), rewriter, loc);
    rewriter.replaceOp(op, llStruct);

    return success();
  }
};

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase : public ConvertTritonGPUOpToLLVMPatternBase {
  explicit LoadStoreConversionBase(AxisInfoAnalysis &axisAnalysisPass)
      : axisAnalysisPass(axisAnalysisPass) {}

  // Get corresponding LLVM element values of \param value.
  static SmallVector<Value> getLLVMElems(Value value, Value llValue,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc) {
    if (!value)
      return {};
    if (!llValue.getType().isa<LLVM::LLVMStructType>())
      return {llValue};
    // Here, we assume that all inputs should have a blockedLayout
    auto valueVals = getElementsFromStruct(loc, llValue, rewriter);
    return valueVals;
  }

  unsigned getVectorSize(Value ptr) const {
    return axisAnalysisPass.getPtrVectorSize(ptr);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  AxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::LoadOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::LoadOp>::ConvertTritonGPUOpToLLVMPattern;

  LoadOpConversion(LLVMTypeConverter &converter,
                   AxisInfoAnalysis &axisAnalysisPass, PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // original values
    Value ptr = op.ptr();
    Value mask = op.mask();
    Value other = op.other();

    // adaptor values
    Value llPtr = adaptor.ptr();
    Value llMask = adaptor.mask();
    Value llOther = adaptor.other();

    // Determine the vectorization size
    Type valueTy = op.getResult().getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned vec = getVectorSize(ptr);
    unsigned numElems = getElemsPerThread(ptr.getType());
    if (llMask)
      vec = std::min<size_t>(vec, getMaskAlignment(mask));

    // Get the LLVM values for pointers
    auto ptrElems = getLLVMElems(ptr, llPtr, rewriter, loc);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = getLLVMElems(mask, llMask, rewriter, loc);
      assert(maskElems.size() == numElems);
    }

    // Get the LLVM values for `other`
    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && valueElemTy.isa<IntegerType>() &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat()) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    auto otherElems = getLLVMElems(other, llOther, rewriter, loc);

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNbits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      // TODO: optimization when ptr is GEP with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNbits);
      const size_t totalWidth = valueElemNbits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNbits;
      assert(wordNElems * nWords * numVecs == numElems);

      // TODO(Superjomn) Add cache policy fields to StoreOp.
      // TODO(Superjomn) Deal with cache policy here.
      const bool hasL2EvictPolicy = false;

      PTXBuilder ptxBuilder;

      Value pred = mask ? maskElems[vecStart] : int_val(1, 1);

      const std::string readConstraint =
          (width == 64) ? "l" : ((width == 32) ? "r" : "c");
      const std::string writeConstraint =
          (width == 64) ? "=l" : ((width == 32) ? "=r" : "=c");

      // prepare asm operands
      auto *dstsOpr = ptxBuilder.newListOperand();
      for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
        auto *opr = ptxBuilder.newOperand(writeConstraint); // =r operations
        dstsOpr->listAppend(opr);
      }

      auto *addrOpr =
          ptxBuilder.newAddrOperand(ptrElems[vecStart], "l", in_off);

      // Define the instruction opcode
      auto &ld = ptxBuilder.create<>("ld")
                     ->o("volatile", op.isVolatile())
                     .global()
                     .o("ca", op.cache() == triton::CacheModifier::CA)
                     .o("cg", op.cache() == triton::CacheModifier::CG)
                     .o("L1::evict_first",
                        op.evict() == triton::EvictionPolicy::EVICT_FIRST)
                     .o("L1::evict_last",
                        op.evict() == triton::EvictionPolicy::EVICT_LAST)
                     .o("L1::cache_hint", hasL2EvictPolicy)
                     .v(nWords)
                     .b(width);

      PTXBuilder::Operand *evictOpr{};

      // Here lack a mlir::Value to bind to this operation, so disabled.
      // if (has_l2_evict_policy)
      //   evictOpr = ptxBuilder.newOperand(l2Evict, "l");

      if (!evictOpr)
        ld(dstsOpr, addrOpr).predicate(pred, "b");
      else
        ld(dstsOpr, addrOpr, evictOpr).predicate(pred, "b");

      if (other) {
        for (size_t ii = 0; ii < nWords; ++ii) {
          // PTX doesn't support mov.u8, so we need to use mov.u16
          auto movWidth = width < 16 ? 16 : width;
          PTXInstr &mov =
              ptxBuilder.create<>("mov")->o("u" + std::to_string(movWidth));

          size_t size = width / valueElemNbits;

          auto vecTy = LLVM::getFixedVectorType(valueElemTy, size);
          Value v = undef(vecTy);
          for (size_t s = 0; s < size; ++s) {
            Value falseVal = otherElems[vecStart + ii * size + s];
            Value sVal = createIndexAttrConstant(
                rewriter, loc, this->getTypeConverter()->getIndexType(), s);
            v = insert_element(vecTy, v, falseVal, sVal);
          }
          v = bitcast(v, IntegerType::get(getContext(), width));

          PTXInstr::Operand *opr{};
          if (otherIsSplatConstInt)
            opr = ptxBuilder.newConstantOperand(splatVal);
          else
            opr = ptxBuilder.newOperand(v, readConstraint);

          mov(dstsOpr->listGet(ii), opr).predicateNot(pred, "b");
        }
      }

      // ---
      // create inline ASM signature
      // ---
      SmallVector<Type> retTys(nWords, IntegerType::get(getContext(), width));
      Type retTy = retTys.size() > 1
                       ? LLVM::LLVMStructType::getLiteral(getContext(), retTys)
                       : retTys[0];

      // TODO: if (has_l2_evict_policy)
      // auto asmDialectAttr =
      // LLVM::AsmDialectAttr::get(rewriter.getContext(),
      //                                                 LLVM::AsmDialect::AD_ATT);
      Value ret = ptxBuilder.launch(rewriter, loc, retTy);

      // ---
      // extract and store return values
      // ---
      SmallVector<Value> rets;
      for (unsigned int ii = 0; ii < nWords; ++ii) {
        Value curr;
        if (retTy.isa<LLVM::LLVMStructType>()) {
          curr = extract_val(IntegerType::get(getContext(), width), ret,
                             rewriter.getI64ArrayAttr(ii));
        } else {
          curr = ret;
        }
        curr = bitcast(curr, LLVM::getFixedVectorType(valueElemTy,
                                                      width / valueElemNbits));
        rets.push_back(curr);
      }
      int tmp = width / valueElemNbits;
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, this->getTypeConverter()->getIndexType(), ii % tmp);
        Value loaded = extract_element(valueElemTy, rets[ii / tmp], vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct =
        getStructFromElements(loc, loadedVals, rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::StoreOp>::ConvertTritonGPUOpToLLVMPattern;

  StoreOpConversion(LLVMTypeConverter &converter,
                    AxisInfoAnalysis &axisAnalysisPass, PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.ptr();
    Value mask = op.mask();
    Value value = op.value();

    Value llPtr = adaptor.ptr();
    Value llMask = adaptor.mask();
    Value llValue = adaptor.value();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    unsigned vec = getVectorSize(ptr);
    unsigned numElems = getElemsPerThread(ptr.getType());

    auto ptrElems = getLLVMElems(ptr, llPtr, rewriter, loc);
    auto valueElems = getLLVMElems(value, llValue, rewriter, loc);
    assert(ptrElems.size() == valueElems.size());

    // Determine the vectorization size
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = getLLVMElems(mask, llMask, rewriter, loc);
      assert(valueElems.size() == maskElems.size());

      unsigned maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    const size_t dtsize =
        std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNbits = dtsize * 8;

    const int numVecs = numElems / vec;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      // TODO: optimization when ptr is AddPtr with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNbits);
      const size_t totalWidth = valueElemNbits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNbits;
      assert(wordNElems * nWords * numVecs == numElems);

      // TODO(Superjomn) Add cache policy fields to StoreOp.
      // TODO(Superjomn) Deal with cache policy here.

      Type valArgTy = IntegerType::get(ctx, width);
      auto wordTy = vec_ty(valueElemTy, wordNElems);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
        // llWord is a width-len composition
        Value llWord = undef(wordTy);
        // Insert each value element to the composition
        for (size_t elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
          const size_t elemOffset = vecStart + wordIdx * wordNElems + elemIdx;
          assert(elemOffset < valueElems.size());
          Value elem = valueElems[elemOffset];
          if (elem.getType().isInteger(1))
            elem = rewriter.create<LLVM::SExtOp>(loc, type::i8Ty(ctx), elem);
          elem = bitcast(elem, valueElemTy);

          Type u32Ty = typeConverter->convertType(type::u32Ty(ctx));
          llWord = insert_element(wordTy, llWord, elem, i32_val(elemIdx));
        }
        llWord = bitcast(llWord, valArgTy);
        std::string constraint =
            (width == 64) ? "l" : ((width == 32) ? "r" : "c");
        asmArgs.emplace_back(llWord, constraint);
      }

      // Prepare the PTX inline asm.
      PTXBuilder ptxBuilder;
      auto *asmArgList = ptxBuilder.newListOperand(asmArgs);

      Value maskVal = llMask ? maskElems[vecStart] : int_val(1, 1);

      auto *asmAddr =
          ptxBuilder.newAddrOperand(ptrElems[vecStart], "l", in_off);

      auto &ptxStoreInstr =
          ptxBuilder.create<>("st")->global().v(nWords).b(width);
      ptxStoreInstr(asmAddr, asmArgList).predicate(maskVal, "b");

      Type boolTy = getTypeConverter()->convertType(rewriter.getIntegerType(1));
      llvm::SmallVector<Type> argTys({boolTy, ptr.getType()});
      argTys.insert(argTys.end(), nWords, valArgTy);

      auto ASMReturnTy = void_ty(ctx);

      ptxBuilder.launch(rewriter, loc, ASMReturnTy);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct BroadcastOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::BroadcastOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::BroadcastOp>::ConvertTritonGPUOpToLLVMPattern;

  // Following the order of indices in the legacy code, a broadcast of:
  //   [s(0), s(1) ... s(k-1),    1, s(k+1), s(k+2) ... s(n-1)]
  // =>
  //   [s(0), s(1) ... s(k-1), s(k), s(k+1), s(k+2) ... s(n-1)]
  //
  // logically maps to a broadcast within a thread's scope:
  //   [cta(0)..cta(k-1),     1,cta(k+1)..cta(n-1),spt(0)..spt(k-1),
  //   1,spt(k+1)..spt(n-1)]
  // =>
  //   [cta(0)..cta(k-1),cta(k),cta(k+1)..cta(n-1),spt(0)..spt(k-1),spt(k),spt(k+1)..spt(n-1)]
  //
  // regardless of the order of the layout
  //
  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value src = adaptor.src();
    Value result = op.result();
    auto srcTy = op.src().getType().cast<RankedTensorType>();
    auto resultTy = result.getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    auto resultLayout = resultTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultShape = resultTy.getShape();
    unsigned rank = srcTy.getRank();
    assert(rank == resultTy.getRank());
    auto order = triton::gpu::getOrder(srcLayout);
    auto srcOffsets = emitOffsetForLayout(srcLayout, srcShape);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultShape);
    SmallVector<Value> srcVals = getElementsFromStruct(loc, src, rewriter);
    DenseMap<SmallVector<unsigned>, Value, SmallVectorKeyInfo> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }
    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      for (size_t j = 0; j < srcShape.size(); j++)
        if (srcShape[j] == 1)
          offset[j] = 0;
      resultVals.push_back(srcValues.lookup(offset));
    }
    auto llvmStructTy = getTypeConverter()->convertType(resultTy);
    Value resultStruct =
        getStructFromElements(loc, resultVals, rewriter, llvmStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

/// ====================== reduce codegen begin ==========================

struct ReduceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::ReduceOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      triton::ReduceOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  void accumulate(ConversionPatternRewriter &rewriter, Location loc,
                  RedOp redOp, Value &acc, Value cur, bool isFirst) const;

  void accumulateWithIndex(ConversionPatternRewriter &rewriter, Location loc,
                           RedOp redOp, Value &acc, Value &accIndex, Value cur,
                           Value curIndex, bool isFirst) const;

  // Use shared memory for reduction within warps and across warps
  LogicalResult matchAndRewriteBasic(triton::ReduceOp op, OpAdaptor adaptor,
                                     ConversionPatternRewriter &rewriter) const;

  // Use warp shuffle for reduction within warps and shared memory for data
  // exchange across warps
  LogicalResult matchAndRewriteFast(triton::ReduceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const;
};

LogicalResult
ReduceOpConversion::matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  if (ReduceOpHelper(op).isFastReduction())
    return matchAndRewriteFast(op, adaptor, rewriter);
  return matchAndRewriteBasic(op, adaptor, rewriter);
}

void ReduceOpConversion::accumulate(ConversionPatternRewriter &rewriter,
                                    Location loc, RedOp redOp, Value &acc,
                                    Value cur, bool isFirst) const {
  if (isFirst) {
    acc = cur;
    return;
  }
  switch (redOp) {
  case RedOp::ADD:
    acc = add(acc, cur);
    break;
  case RedOp::FADD:
    acc = fadd(acc.getType(), acc, cur);
    break;
  case RedOp::MIN:
    acc = smin(acc, cur);
    break;
  case RedOp::MAX:
    acc = smax(acc, cur);
    break;
  case RedOp::UMIN:
    acc = umin(acc, cur);
    break;
  case RedOp::UMAX:
    acc = umax(acc, cur);
    break;
  case RedOp::FMIN:
    acc = fmin(acc, cur);
    break;
  case RedOp::FMAX:
    acc = fmax(acc, cur);
    break;
  case RedOp::XOR:
    acc = xor_(acc, cur);
    break;
  case RedOp::ARGMIN:
  case RedOp::ARGMAX:
  case RedOp::ARGUMIN:
  case RedOp::ARGUMAX:
  case RedOp::ARGFMIN:
  case RedOp::ARGFMAX:
    llvm::report_fatal_error(
        "This accumulate implementation is not for argmin / argmax");
  default:
    llvm::report_fatal_error("Unsupported reduce op");
  }
}

void ReduceOpConversion::accumulateWithIndex(
    ConversionPatternRewriter &rewriter, Location loc, RedOp redOp, Value &acc,
    Value &accIndex, Value cur, Value curIndex, bool isFirst) const {
  if (isFirst) {
    acc = cur;
    accIndex = curIndex;
    return;
  }
  switch (redOp) {
  case RedOp::ARGMIN:
    accIndex =
        select(icmp_slt(acc, cur), accIndex,
               select(icmp_sgt(acc, cur), curIndex, smin(accIndex, curIndex)));
    acc = smin(acc, cur);
    break;
  case RedOp::ARGMAX:
    accIndex =
        select(icmp_sgt(acc, cur), accIndex,
               select(icmp_slt(acc, cur), curIndex, smin(accIndex, curIndex)));
    acc = smax(acc, cur);
    break;
  case RedOp::ARGUMIN:
    accIndex =
        select(icmp_ult(acc, cur), accIndex,
               select(icmp_ugt(acc, cur), curIndex, smin(accIndex, curIndex)));
    acc = umin(acc, cur);
    break;
  case RedOp::ARGUMAX:
    accIndex =
        select(icmp_ugt(acc, cur), accIndex,
               select(icmp_ult(acc, cur), curIndex, smin(accIndex, curIndex)));
    acc = umax(acc, cur);
    break;
  case RedOp::ARGFMIN:
    accIndex =
        select(fcmp_olt(acc, cur), accIndex,
               select(fcmp_ogt(acc, cur), curIndex, smin(accIndex, curIndex)));
    acc = fmin(acc, cur);
    break;
  case RedOp::ARGFMAX:
    accIndex =
        select(fcmp_ogt(acc, cur), accIndex,
               select(fcmp_olt(acc, cur), curIndex, smin(accIndex, curIndex)));
    acc = fmax(acc, cur);
    break;
  case RedOp::ADD:
  case RedOp::FADD:
  case RedOp::MIN:
  case RedOp::MAX:
  case RedOp::UMIN:
  case RedOp::UMAX:
  case RedOp::FMIN:
  case RedOp::FMAX:
  case RedOp::XOR:
    llvm::report_fatal_error(
        "This accumulate implementation is only for argmin / argmax");
  default:
    llvm::report_fatal_error("Unsupported reduce op");
  }
}

LogicalResult ReduceOpConversion::matchAndRewriteBasic(
    triton::ReduceOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  unsigned axis = op.axis();
  bool withIndex = triton::ReduceOp::withIndex(op.redOp());

  auto srcTy = op.operand().getType().cast<RankedTensorType>();
  auto srcLayout = srcTy.getEncoding().cast<BlockedEncodingAttr>();
  auto srcOrd = srcLayout.getOrder();
  auto srcShape = srcTy.getShape();

  auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
  auto llvmIndexTy = getTypeConverter()->getIndexType();
  auto elemPtrTy = LLVM::LLVMPointerType::get(llvmElemTy, 3);
  auto indexPtrTy = LLVM::LLVMPointerType::get(llvmIndexTy, 3);
  Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
  smemBase = bitcast(smemBase, elemPtrTy);

  ReduceOpHelper helper(op);
  auto smemShape = helper.getScratchConfigBasic();
  unsigned elems = product<unsigned>(smemShape);
  Value indexSmemBase = gep(elemPtrTy, smemBase, i32_val(elems));
  indexSmemBase = bitcast(indexSmemBase, indexPtrTy);

  unsigned srcElems = getElemsPerThread(srcTy);
  auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcShape);
  auto srcValues = getElementsFromStruct(loc, adaptor.operand(), rewriter);

  SmallVector<SmallVector<unsigned>> offset =
      emitOffsetForBlockedLayout(srcLayout, srcShape);

  std::map<SmallVector<unsigned>, Value> accs;
  std::map<SmallVector<unsigned>, Value> accIndices;
  std::map<SmallVector<unsigned>, SmallVector<Value>> indices;

  // reduce within threads
  for (unsigned i = 0; i < srcElems; ++i) {
    SmallVector<unsigned> key = offset[i];
    key[axis] = 0;
    bool isFirst = accs.find(key) == accs.end();
    if (!withIndex) {
      accumulate(rewriter, loc, op.redOp(), accs[key], srcValues[i], isFirst);
    } else {
      Value curIndex = srcIndices[i][axis];
      accumulateWithIndex(rewriter, loc, op.redOp(), accs[key], accIndices[key],
                          srcValues[i], curIndex, isFirst);
    }
    if (isFirst)
      indices[key] = srcIndices[i];
  }

  // cached int32 constants
  std::map<int, Value> ints;
  ints[0] = i32_val(0);
  for (int N = smemShape[axis] / 2; N > 0; N >>= 1)
    ints[N] = i32_val(N);
  Value sizePerThread = i32_val(srcLayout.getSizePerThread()[axis]);

  // reduce across threads
  for (auto it : accs) {
    const SmallVector<unsigned> &key = it.first;
    Value acc = it.second;
    Value accIndex;
    if (withIndex)
      accIndex = accIndices[key];
    SmallVector<Value> writeIdx = indices[key];

    writeIdx[axis] = udiv(writeIdx[axis], sizePerThread);
    Value writeOffset = linearize(rewriter, loc, writeIdx, smemShape, srcOrd);
    Value writePtr = gep(elemPtrTy, smemBase, writeOffset);
    Value indexWritePtr = gep(indexPtrTy, indexSmemBase, writeOffset);
    store(acc, writePtr);
    if (withIndex)
      store(accIndex, indexWritePtr);

    SmallVector<Value> readIdx(writeIdx.size(), ints[0]);
    for (int N = smemShape[axis] / 2; N > 0; N >>= 1) {
      readIdx[axis] = ints[N];
      Value readMask = icmp_slt(writeIdx[axis], ints[N]);
      Value readOffset =
          select(readMask, linearize(rewriter, loc, readIdx, smemShape, srcOrd),
                 ints[0]);
      Value readPtr = gep(elemPtrTy, writePtr, readOffset);
      barrier();
      if (!withIndex) {
        Value cur = load(readPtr);
        accumulate(rewriter, loc, op.redOp(), acc, cur, false);
        store(acc, writePtr);
      } else {
        Value cur = load(readPtr);
        Value indexReadPtr = gep(indexPtrTy, indexWritePtr, readOffset);
        Value curIndex = load(indexReadPtr);
        accumulateWithIndex(rewriter, loc, op.redOp(), acc, accIndex, cur,
                            curIndex, false);
        store(acc, writePtr);
        store(accIndex, indexWritePtr);
      }
    }
  }

  barrier();

  // set output values
  if (auto resultTy = op.getType().dyn_cast<RankedTensorType>()) {
    // nd-tensor where n >= 1
    auto resultLayout = resultTy.getEncoding();
    auto resultShape = resultTy.getShape();

    unsigned resultElems = getElemsPerThread(resultTy);
    auto resultIndices = emitIndices(loc, rewriter, resultLayout, resultShape);
    assert(resultIndices.size() == resultElems);

    SmallVector<Value> resultVals(resultElems);
    for (unsigned i = 0; i < resultElems; ++i) {
      SmallVector<Value> readIdx = resultIndices[i];
      readIdx.insert(readIdx.begin() + axis, ints[0]);
      Value readOffset = linearize(rewriter, loc, readIdx, smemShape, srcOrd);
      Value readPtr = gep(elemPtrTy, smemBase, readOffset);
      Value indexReadPtr = gep(indexPtrTy, indexSmemBase, readOffset);
      resultVals[i] = withIndex ? load(indexReadPtr) : load(readPtr);
    }

    SmallVector<Type> resultTypes(resultElems,
                                  withIndex ? llvmIndexTy : llvmElemTy);
    Type structTy =
        LLVM::LLVMStructType::getLiteral(this->getContext(), resultTypes);
    Value ret = getStructFromElements(loc, resultVals, rewriter, structTy);
    rewriter.replaceOp(op, ret);
  } else {
    // 0d-tensor -> scalar
    Value resultVal = withIndex ? load(indexSmemBase) : load(smemBase);
    rewriter.replaceOp(op, resultVal);
  }

  return success();
}

LogicalResult ReduceOpConversion::matchAndRewriteFast(
    triton::ReduceOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  unsigned axis = adaptor.axis();
  bool withIndex = triton::ReduceOp::withIndex(op.redOp());

  auto srcTy = op.operand().getType().cast<RankedTensorType>();
  auto srcLayout = srcTy.getEncoding();
  auto srcShape = srcTy.getShape();
  auto srcRank = srcTy.getRank();
  auto order = getOrder(srcLayout);

  auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcLayout);
  auto warpsPerCTA = triton::gpu::getWarpsPerCTA(srcLayout);

  auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
  auto llvmIndexTy = getTypeConverter()->getIndexType();
  auto elemPtrTy = LLVM::LLVMPointerType::get(llvmElemTy, 3);
  auto indexPtrTy = LLVM::LLVMPointerType::get(llvmIndexTy, 3);
  Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
  smemBase = bitcast(smemBase, elemPtrTy);

  ReduceOpHelper helper(op);
  auto smemShapes = helper.getScratchConfigsFast();
  unsigned elems = product<unsigned>(smemShapes[0]);
  unsigned maxElems = std::max(elems, product<unsigned>(smemShapes[1]));
  Value indexSmemBase = gep(elemPtrTy, smemBase, i32_val(maxElems));
  indexSmemBase = bitcast(indexSmemBase, indexPtrTy);

  unsigned sizeIntraWarps = helper.getIntraWarpSize();
  unsigned sizeInterWarps = helper.getInterWarpSize();

  unsigned srcElems = getElemsPerThread(srcTy);
  auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcShape);
  auto srcValues = getElementsFromStruct(loc, adaptor.operand(), rewriter);

  SmallVector<SmallVector<unsigned>> offset =
      emitOffsetForLayout(srcLayout, srcShape);

  std::map<SmallVector<unsigned>, Value> accs;
  std::map<SmallVector<unsigned>, Value> accIndices;
  std::map<SmallVector<unsigned>, SmallVector<Value>> indices;

  // reduce within threads
  for (unsigned i = 0; i < srcElems; ++i) {
    SmallVector<unsigned> key = offset[i];
    key[axis] = 0;
    bool isFirst = accs.find(key) == accs.end();
    if (!withIndex) {
      accumulate(rewriter, loc, op.redOp(), accs[key], srcValues[i], isFirst);
    } else {
      Value curIndex = srcIndices[i][axis];
      accumulateWithIndex(rewriter, loc, op.redOp(), accs[key], accIndices[key],
                          srcValues[i], curIndex, isFirst);
    }
    if (isFirst)
      indices[key] = srcIndices[i];
  }

  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(32);
  Value warpId = udiv(threadId, warpSize);
  Value laneId = urem(threadId, warpSize);

  SmallVector<Value> multiDimLaneId =
      delinearize(rewriter, loc, laneId, threadsPerWarp, order);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, warpsPerCTA, order);

  Value laneIdAxis = multiDimLaneId[axis];
  Value warpIdAxis = multiDimWarpId[axis];

  Value zero = i32_val(0);
  Value laneZero = icmp_eq(laneIdAxis, zero);
  Value warpZero = icmp_eq(warpIdAxis, zero);

  for (auto it : accs) {
    const SmallVector<unsigned> &key = it.first;
    Value acc = it.second;
    Value accIndex;
    if (withIndex)
      accIndex = accIndices[key];

    // reduce within warps
    for (unsigned N = sizeIntraWarps / 2; N > 0; N >>= 1) {
      Value shfl = shflSync(loc, rewriter, acc, N);
      if (!withIndex) {
        accumulate(rewriter, loc, op.redOp(), acc, shfl, false);
      } else {
        Value shflIndex = shflSync(loc, rewriter, accIndex, N);
        accumulateWithIndex(rewriter, loc, op.redOp(), acc, accIndex, shfl,
                            shflIndex, false);
      }
    }

    SmallVector<Value> writeIdx = indices[key];
    writeIdx[axis] = (sizeInterWarps == 1) ? zero : warpIdAxis;
    Value writeOffset =
        linearize(rewriter, loc, writeIdx, smemShapes[0], order);
    Value writePtr = gep(elemPtrTy, smemBase, writeOffset);
    storeShared(rewriter, loc, writePtr, acc, laneZero);
    if (withIndex) {
      Value indexWritePtr = gep(indexPtrTy, indexSmemBase, writeOffset);
      storeShared(rewriter, loc, indexWritePtr, accIndex, laneZero);
    }
  }

  barrier();

  // the second round of shuffle reduction
  //   now the problem size: sizeInterWarps, s1, s2, .. , sn
  //   where sizeInterWarps is 2^m
  //
  // each thread needs to process:
  //   elemsPerThread = sizeInterWarps * s1 * s2 .. Sn / numThreads
  unsigned numThreads =
      product<unsigned>(triton::gpu::getWarpsPerCTA(srcLayout)) * 32;
  unsigned elemsPerThread = std::max<unsigned>(elems / numThreads, 1);
  Value readOffset = threadId;
  for (unsigned round = 0; round < elemsPerThread; ++round) {
    Value readPtr = gep(elemPtrTy, smemBase, readOffset);
    // FIXME(Qingyi): need predicate icmp_slt(threadId, i32_val(sizeInerWarps))
    Value acc = load(readPtr);
    Value accIndex;
    if (withIndex) {
      Value readIndexPtr = gep(indexPtrTy, indexSmemBase, readOffset);
      accIndex = load(readIndexPtr);
    }

    for (unsigned N = sizeInterWarps / 2; N > 0; N >>= 1) {
      Value shfl = shflSync(loc, rewriter, acc, N);
      if (!withIndex) {
        accumulate(rewriter, loc, op.redOp(), acc, shfl, false);
      } else {
        Value shflIndex = shflSync(loc, rewriter, accIndex, N);
        accumulateWithIndex(rewriter, loc, op.redOp(), acc, accIndex, shfl,
                            shflIndex, false);
      }
    }

    // only the first thread in each sizeInterWarps is writing
    Value writeOffset = readOffset;
    Value writePtr = gep(elemPtrTy, smemBase, writeOffset);
    Value threadIsNeeded = icmp_slt(threadId, i32_val(elems));
    Value laneIdModSizeInterWarps = urem(laneId, i32_val(sizeInterWarps));
    Value laneIdModSizeInterWarpsIsZero =
        icmp_eq(laneIdModSizeInterWarps, zero);
    Value pred = and_(threadIsNeeded, laneIdModSizeInterWarpsIsZero);
    storeShared(rewriter, loc, writePtr, acc, pred);
    if (withIndex) {
      Value writeIndexPtr = gep(indexPtrTy, indexSmemBase, writeOffset);
      storeShared(rewriter, loc, writeIndexPtr, accIndex, pred);
    }

    if (round != elemsPerThread - 1) {
      readOffset = add(readOffset, i32_val(numThreads));
    }
  }

  // We could avoid this barrier in some of the layouts, however this is not
  // the general case. TODO: optimize the barrier incase the layouts are
  // accepted.
  barrier();

  // set output values
  if (auto resultTy = op.getType().dyn_cast<RankedTensorType>()) {
    // nd-tensor where n >= 1
    auto resultLayout = resultTy.getEncoding().cast<SliceEncodingAttr>();
    auto resultShape = resultTy.getShape();
    unsigned resultElems = getElemsPerThread(resultTy);
    auto resultIndices = emitIndices(loc, rewriter, resultLayout, resultShape);
    assert(resultIndices.size() == resultElems);

    SmallVector<Value> resultVals(resultElems);
    for (size_t i = 0; i < resultElems; ++i) {
      SmallVector<Value> readIdx = resultIndices[i];
      readIdx.insert(readIdx.begin() + axis, i32_val(0));
      Value readOffset =
          linearize(rewriter, loc, readIdx, smemShapes[0], order);
      Value readPtr = gep(elemPtrTy, smemBase, readOffset);
      Value indexReadPtr = gep(indexPtrTy, indexSmemBase, readOffset);
      resultVals[i] = withIndex ? load(indexReadPtr) : load(readPtr);
    }

    SmallVector<Type> resultTypes(resultElems,
                                  withIndex ? llvmIndexTy : llvmElemTy);
    Type structTy =
        LLVM::LLVMStructType::getLiteral(this->getContext(), resultTypes);
    Value ret = getStructFromElements(loc, resultVals, rewriter, structTy);
    rewriter.replaceOp(op, ret);
  } else {
    // 0d-tensor -> scalar
    Value resultVal = withIndex ? load(indexSmemBase) : load(smemBase);
    rewriter.replaceOp(op, resultVal);
  }

  return success();
}

/// ====================== reduce codegen end ==========================

/// ====================== cat codegen begin  ==========================

struct CatOpConversion : public ConvertTritonGPUOpToLLVMPattern<CatOp> {
  using OpAdaptor = typename CatOp::Adaptor;

  explicit CatOpConversion(LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<CatOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType().template cast<RankedTensorType>();
    unsigned elems = getElemsPerThread(resultTy);
    Type elemTy =
        this->getTypeConverter()->convertType(resultTy.getElementType());
    SmallVector<Type> types(elems, elemTy);
    // unpack input values
    auto lhsVals = getElementsFromStruct(loc, adaptor.lhs(), rewriter);
    auto rhsVals = getElementsFromStruct(loc, adaptor.rhs(), rewriter);
    // concatenate (and potentially reorder) values
    SmallVector<Value> retVals;
    for (Value v : lhsVals)
      retVals.push_back(v);
    for (Value v : rhsVals)
      retVals.push_back(v);
    // pack and replace
    Type structTy = LLVM::LLVMStructType::getLiteral(this->getContext(), types);
    Value ret = getStructFromElements(loc, retVals, rewriter, structTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

/// ====================== cat codegen end    ==========================

template <typename SourceOp>
struct ViewLikeOpConversion : public ConvertTritonGPUOpToLLVMPattern<SourceOp> {
  using OpAdaptor = typename SourceOp::Adaptor;
  explicit ViewLikeOpConversion(LLVMTypeConverter &typeConverter,
                                PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<SourceOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We cannot directly run
    //   `rewriter.replaceOp(op, adaptor.src())`
    // due to MLIR's restrictions
    Location loc = op->getLoc();
    auto resultTy = op.getType().template cast<RankedTensorType>();
    unsigned elems = getElemsPerThread(resultTy);
    Type elemTy =
        this->getTypeConverter()->convertType(resultTy.getElementType());
    SmallVector<Type> types(elems, elemTy);
    Type structTy = LLVM::LLVMStructType::getLiteral(this->getContext(), types);
    auto vals = getElementsFromStruct(loc, adaptor.src(), rewriter);
    Value view = getStructFromElements(loc, vals, rewriter, structTy);
    rewriter.replaceOp(op, view);
    return success();
  }
};

struct PrintfOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::PrintfOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::PrintfOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::PrintfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    SmallVector<Value, 16> operands;
    for (auto operand : adaptor.getOperands()) {
      auto sub_operands = getElementsFromStruct(loc, operand, rewriter);
      for (auto elem : sub_operands) {
        operands.push_back(elem);
      }
    }
    std::string formatStr;
    llvm::raw_string_ostream os(formatStr);
    os << op.prefix();
    if (!operands.empty()) {
      os << getFormatSubstr(operands[0]);
    }

    for (size_t i = 1; i < operands.size(); ++i) {
      os << ", " << getFormatSubstr(operands[i]);
    }
    llPrintf(formatStr, operands, rewriter);
    rewriter.eraseOp(op);
    return success();
  }
  // get format specific for each input value
  // currently support pointer, i8, i16, i32, i64, f16, bf16, f32, f64
  std::string getFormatSubstr(Value value) const {
    Type type = value.getType();
    if (type.isa<LLVM::LLVMPointerType>()) {
      return "%p";
    } else if (type.isBF16() || type.isF16() || type.isF32() || type.isF64()) {
      return "%f";
    } else if (type.isSignedInteger()) {
      return "%i";
    } else if (type.isUnsignedInteger() || type.isSignlessInteger()) {
      return "%u";
    }
    assert(false && "not supported type");
    return "";
  }

  // declare vprintf(i8*, i8*) as external function
  static LLVM::LLVMFuncOp
  getVprintfDeclaration(ConversionPatternRewriter &rewriter) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName("vprintf");
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    auto *context = rewriter.getContext();

    SmallVector<Type> argsType{ptr_ty(IntegerType::get(context, 8)),
                               ptr_ty(IntegerType::get(context, 8))};
    auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType);

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
                                             funcType);
  }

  // extend integer to int32, extend float to float64
  // this comes from vprintf alignment requirements.
  static std::pair<Type, Value>
  promoteValue(ConversionPatternRewriter &rewriter, Value value) {
    auto *context = rewriter.getContext();
    auto type = value.getType();
    Value newOp = value;
    Type newType = type;

    bool bUnsigned = type.isUnsignedInteger();
    if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
      if (bUnsigned) {
        newType = ui32_ty;
        newOp = rewriter.create<LLVM::ZExtOp>(UnknownLoc::get(context), newType,
                                              value);
      } else {
        newType = i32_ty;
        newOp = rewriter.create<LLVM::SExtOp>(UnknownLoc::get(context), newType,
                                              value);
      }
    } else if (type.isBF16() || type.isF16() || type.isF32()) {
      newType = f64_ty;
      newOp = rewriter.create<LLVM::FPExtOp>(UnknownLoc::get(context), newType,
                                             value);
    }

    return {newType, newOp};
  }

  static void llPrintf(StringRef msg, ValueRange args,
                       ConversionPatternRewriter &rewriter) {
    static const char formatStringPrefix[] = "printfFormat_";
    assert(!msg.empty() && "printf with empty string not support");
    Type int8Ptr = ptr_ty(i8_ty);

    auto *context = rewriter.getContext();
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    auto funcOp = getVprintfDeclaration(rewriter);

    Value one = rewriter.create<LLVM::ConstantOp>(
        UnknownLoc::get(context), i32_ty, rewriter.getI32IntegerAttr(1));
    Value zero = rewriter.create<LLVM::ConstantOp>(
        UnknownLoc::get(context), i32_ty, rewriter.getI32IntegerAttr(0));

    unsigned stringNumber = 0;
    SmallString<16> stringConstName;
    do {
      stringConstName.clear();
      (formatStringPrefix + Twine(stringNumber++)).toStringRef(stringConstName);
    } while (moduleOp.lookupSymbol(stringConstName));

    llvm::SmallString<64> formatString(msg);
    formatString.push_back('\n');
    formatString.push_back('\0');
    size_t formatStringSize = formatString.size_in_bytes();
    auto globalType = LLVM::LLVMArrayType::get(i8_ty, formatStringSize);

    LLVM::GlobalOp global;
    {
      ConversionPatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      global = rewriter.create<LLVM::GlobalOp>(
          UnknownLoc::get(context), globalType,
          /*isConstant=*/true, LLVM::Linkage::Internal, stringConstName,
          rewriter.getStringAttr(formatString));
    }

    Value globalPtr =
        rewriter.create<LLVM::AddressOfOp>(UnknownLoc::get(context), global);
    Value stringStart = rewriter.create<LLVM::GEPOp>(
        UnknownLoc::get(context), int8Ptr, globalPtr,
        SmallVector<Value>({zero, zero}));

    Value bufferPtr =
        rewriter.create<LLVM::NullOp>(UnknownLoc::get(context), int8Ptr);

    SmallVector<Value, 16> newArgs;
    if (args.size() >= 1) {
      SmallVector<Type> argTypes;
      for (auto arg : args) {
        Type newType;
        Value newArg;
        std::tie(newType, newArg) = promoteValue(rewriter, arg);
        argTypes.push_back(newType);
        newArgs.push_back(newArg);
      }

      Type structTy = LLVM::LLVMStructType::getLiteral(context, argTypes);
      auto allocated = rewriter.create<LLVM::AllocaOp>(UnknownLoc::get(context),
                                                       ptr_ty(structTy), one,
                                                       /*alignment=*/0);

      for (const auto &entry : llvm::enumerate(newArgs)) {
        auto index = rewriter.create<LLVM::ConstantOp>(
            UnknownLoc::get(context), i32_ty,
            rewriter.getI32IntegerAttr(entry.index()));
        auto fieldPtr = rewriter.create<LLVM::GEPOp>(
            UnknownLoc::get(context), ptr_ty(argTypes[entry.index()]),
            allocated, ArrayRef<Value>{zero, index});
        rewriter.create<LLVM::StoreOp>(UnknownLoc::get(context), entry.value(),
                                       fieldPtr);
      }
      bufferPtr = rewriter.create<LLVM::BitcastOp>(UnknownLoc::get(context),
                                                   int8Ptr, allocated);
    }

    SmallVector<Value> operands{stringStart, bufferPtr};
    rewriter.create<LLVM::CallOp>(UnknownLoc::get(context), funcOp, operands);
  }
};

struct MakeRangeOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::MakeRangeOp> {

  MakeRangeOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::MakeRangeOp>(converter,
                                                             benefit) {}

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto rankedTy = op.result().getType().dyn_cast<RankedTensorType>();
    auto shape = rankedTy.getShape();
    auto layout = rankedTy.getEncoding();

    auto elemTy = rankedTy.getElementType();
    assert(elemTy.isInteger(32));
    Value start = createIndexAttrConstant(rewriter, loc, elemTy, op.start());
    auto idxs = emitIndices(loc, rewriter, layout, shape);
    unsigned elems = idxs.size();
    SmallVector<Value> retVals(elems);
    // TODO: slice layout has more elements than expected.
    // Unexpected behavior for make range, but genereally ok when followed by
    // expand dims + broadcast. very weird behavior otherwise potentially.
    for (const auto multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = add(multiDim.value()[0], start);
    }
    SmallVector<Type> types(elems, elemTy);
    Type structTy = LLVM::LLVMStructType::getLiteral(getContext(), types);
    Value result = getStructFromElements(loc, retVals, rewriter, structTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct GetProgramIdOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::GetProgramIdOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::GetProgramIdOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    assert(op.axis() < 3);

    Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(
        loc, rewriter.getIndexType(), dims[op.axis()]);
    auto llvmIndexTy = getTypeConverter()->getIndexType();
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, TypeRange{llvmIndexTy}, ValueRange{blockId});
    return success();
  }

  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
};

struct GetNumProgramsOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    assert(op.axis() < 3);

    Value blockId = rewriter.create<::mlir::gpu::GridDimOp>(
        loc, rewriter.getIndexType(), dims[op.axis()]);
    auto llvmIndexTy = getTypeConverter()->getIndexType();
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, TypeRange{llvmIndexTy}, ValueRange{blockId});
    return success();
  }

  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
};

struct AddPtrOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AddPtrOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AddPtrOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType();
    auto resultTensorTy = resultTy.dyn_cast<RankedTensorType>();
    if (resultTensorTy) {
      unsigned elems = getElemsPerThread(resultTy);
      Type elemTy =
          getTypeConverter()->convertType(resultTensorTy.getElementType());
      SmallVector<Type> types(elems, elemTy);
      Type structTy = LLVM::LLVMStructType::getLiteral(getContext(), types);
      auto ptrs = getElementsFromStruct(loc, adaptor.ptr(), rewriter);
      auto offsets = getElementsFromStruct(loc, adaptor.offset(), rewriter);
      SmallVector<Value> resultVals(elems);
      for (unsigned i = 0; i < elems; ++i) {
        resultVals[i] = gep(elemTy, ptrs[i], offsets[i]);
      }
      Value view = getStructFromElements(loc, resultVals, rewriter, structTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(resultTy.isa<triton::PointerType>());
      Type llResultTy = getTypeConverter()->convertType(resultTy);
      Value result = gep(llResultTy, adaptor.ptr(), adaptor.offset());
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

struct AllocTensorOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AllocTensorOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AllocTensorOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AllocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getResult());
    auto resultTy = op.getType().dyn_cast<RankedTensorType>();
    auto llvmElemTy =
        getTypeConverter()->convertType(resultTy.getElementType());
    auto elemPtrTy = ptr_ty(llvmElemTy, 3);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto order = resultTy.getEncoding().cast<SharedEncodingAttr>().getOrder();
    // workaround for 3D tensors
    // TODO: We need to modify the pipeline pass to give a proper shared
    // encoding to 3D tensors
    SmallVector<unsigned> newOrder;
    if (resultTy.getShape().size() == 3)
      newOrder = {1 + order[0], 1 + order[1], 0};
    else
      newOrder = SmallVector<unsigned>(order.begin(), order.end());

    auto smemObj = SharedMemoryObject(smemBase, resultTy.getShape(), newOrder,
                                      loc, rewriter);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct ExtractSliceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<tensor::ExtractSliceOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      tensor::ExtractSliceOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = extract_slice %src[%offsets]
    Location loc = op->getLoc();
    auto srcTy = op.source().getType().dyn_cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    assert(srcLayout && "Unexpected resultLayout in ExtractSliceOpConversion");
    assert(op.hasUnitStride() &&
           "Only unit stride supported by ExtractSliceOpConversion");

    // newBase = base + offset
    // Triton support either static and dynamic offsets
    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, adaptor.source(), rewriter);
    SmallVector<Value, 4> opOffsetVals;
    SmallVector<Value, 4> offsetVals;
    auto mixedOffsets = op.getMixedOffsets();
    for (auto i = 0; i < mixedOffsets.size(); ++i) {
      if (op.isDynamicOffset(i))
        opOffsetVals.emplace_back(adaptor.offsets()[i]);
      else
        opOffsetVals.emplace_back(i32_val(op.getStaticOffset(i)));
      offsetVals.emplace_back(add(smemObj.offsets[i], opOffsetVals[i]));
    }
    // Compute the offset based on the original strides of the shared memory
    // object
    auto offset = dot(rewriter, loc, opOffsetVals, smemObj.strides);
    // newShape = rank_reduce(shape)
    // Triton only supports static tensor sizes
    SmallVector<Value, 4> strideVals;
    for (auto i = 0; i < op.static_sizes().size(); ++i) {
      if (op.getStaticSize(i) == 1) {
        offsetVals.erase(offsetVals.begin() + i);
      } else {
        strideVals.emplace_back(smemObj.strides[i]);
      }
    }

    // llvm::outs() << "extract slice\n";
    // llvm::outs() << strideVals[0] << " " << smemObj.strides[1] << "\n";
    // llvm::outs() << strideVals[1] << " " << smemObj.strides[2] << "\n";
    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto elemPtrTy = ptr_ty(llvmElemTy, 3);
    auto resTy = op.getType().dyn_cast<RankedTensorType>();
    smemObj = SharedMemoryObject(gep(elemPtrTy, smemObj.base, offset),
                                 strideVals, offsetVals);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct FpToFpOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::FpToFpOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::FpToFpOp>::ConvertTritonGPUOpToLLVMPattern;

  static SmallVector<Value>
  convertFp8x4ToFp16x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto ctx = rewriter.getContext();
    auto fp8x4VecTy = vec_ty(i8_ty, 4);
    Value fp8x4Vec = undef(fp8x4VecTy);
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v0, i32_val(0));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v1, i32_val(1));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v2, i32_val(2));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v3, i32_val(3));
    fp8x4Vec = bitcast(fp8x4Vec, i32_ty);

    PTXBuilder builder;
    auto *ptxAsm = "{                                      \n"
                   ".reg .b32 a<2>, b<2>;                  \n"
                   "prmt.b32 a0, 0, $2, 0x5040;            \n"
                   "prmt.b32 a1, 0, $2, 0x7060;            \n"
                   "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;  \n"
                   "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;  \n"
                   "shr.b32  b0, b0, 1;                    \n"
                   "shr.b32  b1, b1, 1;                    \n"
                   "lop3.b32 $0, b0, 0x80008000, a0, 0xf8; \n"
                   "lop3.b32 $1, b1, 0x80008000, a1, 0xf8; \n"
                   "}";
    auto &call = *builder.create(ptxAsm);

    auto *o0 = builder.newOperand("=r");
    auto *o1 = builder.newOperand("=r");
    auto *i = builder.newOperand(fp8x4Vec, "r");
    call({o0, o1, i}, /* onlyAttachMLIRArgs */ true);

    auto fp16x2VecTy = vec_ty(f16_ty, 2);
    auto fp16x2x2StructTy =
        struct_ty(SmallVector<Type>{fp16x2VecTy, fp16x2VecTy});
    auto fp16x2x2Struct =
        builder.launch(rewriter, loc, fp16x2x2StructTy, false);
    auto fp16x2Vec0 =
        extract_val(fp16x2VecTy, fp16x2x2Struct, rewriter.getI32ArrayAttr({0}));
    auto fp16x2Vec1 =
        extract_val(fp16x2VecTy, fp16x2x2Struct, rewriter.getI32ArrayAttr({1}));
    return {extract_element(f16_ty, fp16x2Vec0, i32_val(0)),
            extract_element(f16_ty, fp16x2Vec0, i32_val(1)),
            extract_element(f16_ty, fp16x2Vec1, i32_val(0)),
            extract_element(f16_ty, fp16x2Vec1, i32_val(1))};
  }

  static SmallVector<Value>
  convertFp16x4ToFp8x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto ctx = rewriter.getContext();
    auto fp16x2VecTy = vec_ty(f16_ty, 2);
    Value fp16x2Vec0 = undef(fp16x2VecTy);
    Value fp16x2Vec1 = undef(fp16x2VecTy);
    fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v0, i32_val(0));
    fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v1, i32_val(1));
    fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v2, i32_val(0));
    fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v3, i32_val(1));
    fp16x2Vec0 = bitcast(fp16x2Vec0, i32_ty);
    fp16x2Vec1 = bitcast(fp16x2Vec1, i32_ty);

    PTXBuilder builder;
    auto *ptxAsm = "{                                      \n"
                   ".reg .b32 a<2>, b<2>;                  \n"
                   "shl.b32 a0, $1, 1;                     \n"
                   "shl.b32 a1, $2, 1;                     \n"
                   "lop3.b32 a0, a0, 0x7fff7fff, 0, 0xc0;  \n"
                   "lop3.b32 a1, a1, 0x7fff7fff, 0, 0xc0;  \n"
                   "add.u32 a0, a0, 0x00800080;            \n"
                   "add.u32 a1, a1, 0x00800080;            \n"
                   "lop3.b32 b0, $1, 0x80008000, a0, 0xea; \n"
                   "lop3.b32 b1, $2, 0x80008000, a1, 0xea; \n"
                   "prmt.b32 $0, b0, b1, 0x7531;           \n"
                   "}";
    auto &call = *builder.create(ptxAsm);

    auto *o = builder.newOperand("=r");
    auto *i0 = builder.newOperand(fp16x2Vec0, "r");
    auto *i1 = builder.newOperand(fp16x2Vec1, "r");
    call({o, i0, i1}, /* onlyAttachMLIRArgs */ true);

    auto fp8x4VecTy = vec_ty(i8_ty, 4);
    auto fp8x4Vec = builder.launch(rewriter, loc, fp8x4VecTy, false);
    return {extract_element(i8_ty, fp8x4Vec, i32_val(0)),
            extract_element(i8_ty, fp8x4Vec, i32_val(1)),
            extract_element(i8_ty, fp8x4Vec, i32_val(2)),
            extract_element(i8_ty, fp8x4Vec, i32_val(3))};
  }

  static SmallVector<Value>
  convertFp8x4ToBf16x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto ctx = rewriter.getContext();
    auto fp8x4VecTy = vec_ty(i8_ty, 4);
    Value fp8x4Vec = undef(fp8x4VecTy);
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v0, i32_val(0));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v1, i32_val(1));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v2, i32_val(2));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v3, i32_val(3));
    fp8x4Vec = bitcast(fp8x4Vec, i32_ty);

    PTXBuilder builder;
    auto *ptxAsm = "{                                          \n"
                   ".reg .b32 a<2>, sign<2>, nosign<2>, b<2>;  \n"
                   "prmt.b32 a0, 0, $2, 0x5040;                \n"
                   "prmt.b32 a1, 0, $2, 0x7060;                \n"
                   "and.b32 sign0, a0, 0x80008000;             \n"
                   "and.b32 sign1, a1, 0x80008000;             \n"
                   "and.b32 nosign0, a0, 0x7fff7fff;           \n"
                   "and.b32 nosign1, a1, 0x7fff7fff;           \n"
                   "shr.b32 nosign0, nosign0, 4;               \n"
                   "shr.b32 nosign1, nosign1, 4;               \n"
                   "add.u32 nosign0, nosign0, 0x38003800;      \n"
                   "add.u32 nosign1, nosign1, 0x38003800;      \n"
                   "or.b32 $0, sign0, nosign0;                 \n"
                   "or.b32 $1, sign1, nosign1;                 \n"
                   "}";
    auto &call = *builder.create(ptxAsm);

    auto *o0 = builder.newOperand("=r");
    auto *o1 = builder.newOperand("=r");
    auto *i = builder.newOperand(fp8x4Vec, "r");
    call({o0, o1, i}, /* onlyAttachMLIRArgs */ true);

    auto bf16x2VecTy = vec_ty(bf16_ty, 2);
    auto bf16x2x2StructTy =
        struct_ty(SmallVector<Type>{bf16x2VecTy, bf16x2VecTy});
    auto bf16x2x2Struct =
        builder.launch(rewriter, loc, bf16x2x2StructTy, false);
    auto bf16x2Vec0 =
        extract_val(bf16x2VecTy, bf16x2x2Struct, rewriter.getI32ArrayAttr({0}));
    auto bf16x2Vec1 =
        extract_val(bf16x2VecTy, bf16x2x2Struct, rewriter.getI32ArrayAttr({1}));
    return {extract_element(bf16_ty, bf16x2Vec0, i32_val(0)),
            extract_element(bf16_ty, bf16x2Vec0, i32_val(1)),
            extract_element(bf16_ty, bf16x2Vec1, i32_val(0)),
            extract_element(bf16_ty, bf16x2Vec1, i32_val(1))};
  }

  static SmallVector<Value>
  convertBf16x4ToFp8x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto ctx = rewriter.getContext();
    auto bf16x2VecTy = vec_ty(bf16_ty, 2);
    Value bf16x2Vec0 = undef(bf16x2VecTy);
    Value bf16x2Vec1 = undef(bf16x2VecTy);
    bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v0, i32_val(0));
    bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v1, i32_val(1));
    bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v2, i32_val(0));
    bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v3, i32_val(1));
    bf16x2Vec0 = bitcast(bf16x2Vec0, i32_ty);
    bf16x2Vec1 = bitcast(bf16x2Vec1, i32_ty);

    PTXBuilder builder;
    auto *ptxAsm = "{                                            \n"
                   ".reg .u32 sign, sign<2>, nosign, nosign<2>;  \n"
                   ".reg .u32 fp8_min, fp8_max, rn_, zero;       \n"
                   "mov.u32 fp8_min, 0x38003800;                 \n"
                   "mov.u32 fp8_max, 0x3ff03ff0;                 \n"
                   "mov.u32 rn_, 0x80008;                        \n"
                   "mov.u32 zero, 0;                             \n"
                   "and.b32 sign0, $1, 0x80008000;               \n"
                   "and.b32 sign1, $2, 0x80008000;               \n"
                   "prmt.b32 sign, sign0, sign1, 0x7531;         \n"
                   "and.b32 nosign0, $1, 0x7fff7fff;             \n"
                   "and.b32 nosign1, $2, 0x7fff7fff;             \n"
                   ".reg .u32 nosign_0_<2>, nosign_1_<2>;        \n"
                   "and.b32 nosign_0_0, nosign0, 0xffff0000;     \n"
                   "max.u32 nosign_0_0, nosign_0_0, 0x38000000;  \n"
                   "min.u32 nosign_0_0, nosign_0_0, 0x3ff00000;  \n"
                   "and.b32 nosign_0_1, nosign0, 0x0000ffff;     \n"
                   "max.u32 nosign_0_1, nosign_0_1, 0x3800;      \n"
                   "min.u32 nosign_0_1, nosign_0_1, 0x3ff0;      \n"
                   "or.b32 nosign0, nosign_0_0, nosign_0_1;      \n"
                   "and.b32 nosign_1_0, nosign1, 0xffff0000;     \n"
                   "max.u32 nosign_1_0, nosign_1_0, 0x38000000;  \n"
                   "min.u32 nosign_1_0, nosign_1_0, 0x3ff00000;  \n"
                   "and.b32 nosign_1_1, nosign1, 0x0000ffff;     \n"
                   "max.u32 nosign_1_1, nosign_1_1, 0x3800;      \n"
                   "min.u32 nosign_1_1, nosign_1_1, 0x3ff0;      \n"
                   "or.b32 nosign1, nosign_1_0, nosign_1_1;      \n"
                   "add.u32 nosign0, nosign0, rn_;               \n"
                   "add.u32 nosign1, nosign1, rn_;               \n"
                   "sub.u32 nosign0, nosign0, 0x38003800;        \n"
                   "sub.u32 nosign1, nosign1, 0x38003800;        \n"
                   "shr.u32 nosign0, nosign0, 4;                 \n"
                   "shr.u32 nosign1, nosign1, 4;                 \n"
                   "prmt.b32 nosign, nosign0, nosign1, 0x6420;   \n"
                   "or.b32 $0, nosign, sign;                     \n"
                   "}";
    auto &call = *builder.create(ptxAsm);

    auto *o = builder.newOperand("=r");
    auto *i0 = builder.newOperand(bf16x2Vec0, "r");
    auto *i1 = builder.newOperand(bf16x2Vec1, "r");
    call({o, i0, i1}, /* onlyAttachMLIRArgs */ true);

    auto fp8x4VecTy = vec_ty(i8_ty, 4);
    auto fp8x4Vec = builder.launch(rewriter, loc, fp8x4VecTy, false);
    return {extract_element(i8_ty, fp8x4Vec, i32_val(0)),
            extract_element(i8_ty, fp8x4Vec, i32_val(1)),
            extract_element(i8_ty, fp8x4Vec, i32_val(2)),
            extract_element(i8_ty, fp8x4Vec, i32_val(3))};
  }

  static SmallVector<Value>
  convertFp8x4ToFp32x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto fp16Values = convertFp8x4ToFp16x4(loc, rewriter, v0, v1, v2, v3);
    return {rewriter.create<LLVM::FPExtOp>(loc, f32_ty, fp16Values[0]),
            rewriter.create<LLVM::FPExtOp>(loc, f32_ty, fp16Values[1]),
            rewriter.create<LLVM::FPExtOp>(loc, f32_ty, fp16Values[2]),
            rewriter.create<LLVM::FPExtOp>(loc, f32_ty, fp16Values[3])};
  }

  static SmallVector<Value>
  convertFp32x4ToFp8x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto c0 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v0);
    auto c1 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v1);
    auto c2 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v2);
    auto c3 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v3);
    return convertFp16x4ToFp8x4(loc, rewriter, c0, c1, c2, c3);
  }

  static SmallVector<Value>
  convertFp8x4ToFp64x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto fp16Values = convertFp8x4ToFp16x4(loc, rewriter, v0, v1, v2, v3);
    return {rewriter.create<LLVM::FPExtOp>(loc, f64_ty, fp16Values[0]),
            rewriter.create<LLVM::FPExtOp>(loc, f64_ty, fp16Values[1]),
            rewriter.create<LLVM::FPExtOp>(loc, f64_ty, fp16Values[2]),
            rewriter.create<LLVM::FPExtOp>(loc, f64_ty, fp16Values[3])};
  }

  static SmallVector<Value>
  convertFp64x4ToFp8x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto c0 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v0);
    auto c1 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v1);
    auto c2 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v2);
    auto c3 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v3);
    return convertFp16x4ToFp8x4(loc, rewriter, c0, c1, c2, c3);
  }

  LogicalResult
  matchAndRewrite(triton::FpToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTensorType = op.from().getType().cast<mlir::RankedTensorType>();
    auto dstTensorType = op.result().getType().cast<mlir::RankedTensorType>();
    auto srcEltType = srcTensorType.getElementType();
    auto dstEltType = dstTensorType.getElementType();
    assert(srcEltType.isa<triton::Float8Type>() ||
           dstEltType.isa<triton::Float8Type>());
    auto convertedDstTensorType =
        this->getTypeConverter()->convertType(dstTensorType);
    auto convertedDstEleType =
        this->getTypeConverter()->convertType(dstEltType);

    // Select convertor
    std::function<SmallVector<Value>(Location, ConversionPatternRewriter &,
                                     const Value &, const Value &,
                                     const Value &, const Value &)>
        convertor;
    if (srcEltType.isa<triton::Float8Type>() && dstEltType.isF16()) {
      convertor = convertFp8x4ToFp16x4;
    } else if (srcEltType.isF16() && dstEltType.isa<triton::Float8Type>()) {
      convertor = convertFp16x4ToFp8x4;
    } else if (srcEltType.isa<triton::Float8Type>() && dstEltType.isBF16()) {
      convertor = convertFp8x4ToBf16x4;
    } else if (srcEltType.isBF16() && dstEltType.isa<triton::Float8Type>()) {
      convertor = convertBf16x4ToFp8x4;
    } else if (srcEltType.isa<triton::Float8Type>() && dstEltType.isF32()) {
      convertor = convertFp8x4ToFp32x4;
    } else if (srcEltType.isF32() && dstEltType.isa<triton::Float8Type>()) {
      convertor = convertFp32x4ToFp8x4;
    } else if (srcEltType.isa<triton::Float8Type>() && dstEltType.isF64()) {
      convertor = convertFp8x4ToFp64x4;
    } else if (srcEltType.isF64() && dstEltType.isa<triton::Float8Type>()) {
      convertor = convertFp64x4ToFp8x4;
    } else {
      assert(false && "unsupported type casting");
    }

    // Vectorized casting
    auto loc = op->getLoc();
    auto elems = getElemsPerThread(dstTensorType);
    assert(elems % 4 == 0 &&
           "FP8 casting only support tensors with 4-aligned sizes");
    auto elements = getElementsFromStruct(loc, adaptor.from(), rewriter);
    SmallVector<Value> resultVals;
    for (size_t i = 0; i < elems; i += 4) {
      auto converted = convertor(loc, rewriter, elements[i], elements[i + 1],
                                 elements[i + 2], elements[i + 3]);
      resultVals.append(converted);
    }
    assert(resultVals.size() == elems);
    auto result = getStructFromElements(loc, resultVals, rewriter,
                                        convertedDstTensorType);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// A CRTP style of base class.
template <typename SourceOp, typename ConcreteT>
class ElementwiseOpConversionBase
    : public ConvertTritonGPUOpToLLVMPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ElementwiseOpConversionBase(LLVMTypeConverter &typeConverter,
                                       PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<SourceOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType();
    Location loc = op->getLoc();

    unsigned elems = getElemsPerThread(resultTy);
    auto resultElementTy = getElementTypeOrSelf(resultTy);
    Type elemTy = this->getTypeConverter()->convertType(resultElementTy);
    SmallVector<Type> types(elems, elemTy);
    Type structTy = this->getTypeConverter()->convertType(resultTy);

    auto *concreteThis = static_cast<const ConcreteT *>(this);
    auto operands = getOperands(rewriter, adaptor, elems, loc);
    SmallVector<Value> resultVals(elems);
    for (unsigned i = 0; i < elems; ++i) {
      resultVals[i] = concreteThis->createDestOp(op, adaptor, rewriter, elemTy,
                                                 operands[i], loc);
      if (!bool(resultVals[i]))
        return failure();
    }
    Value view = getStructFromElements(loc, resultVals, rewriter, structTy);
    rewriter.replaceOp(op, view);

    return success();
  }

protected:
  SmallVector<SmallVector<Value>>
  getOperands(ConversionPatternRewriter &rewriter, OpAdaptor adaptor,
              const unsigned elems, Location loc) const {
    SmallVector<SmallVector<Value>> operands(elems);
    for (auto operand : adaptor.getOperands()) {
      auto sub_operands = getElementsFromStruct(loc, operand, rewriter);
      for (size_t i = 0; i < elems; ++i) {
        operands[i].push_back(sub_operands[i]);
      }
    }
    return operands;
  }
};

template <typename SourceOp, typename DestOp>
struct ElementwiseOpConversion
    : public ElementwiseOpConversionBase<
          SourceOp, ElementwiseOpConversion<SourceOp, DestOp>> {
  using Base =
      ElementwiseOpConversionBase<SourceOp,
                                  ElementwiseOpConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  explicit ElementwiseOpConversion(LLVMTypeConverter &typeConverter,
                                   PatternBenefit benefit = 1)
      : ElementwiseOpConversionBase<SourceOp, ElementwiseOpConversion>(
            typeConverter, benefit) {}

  // An interface to support variant DestOp builder.
  DestOp createDestOp(SourceOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter, Type elemTy,
                      ValueRange operands, Location loc) const {
    return rewriter.create<DestOp>(loc, elemTy, operands,
                                   adaptor.getAttributes().getValue());
  }
};

//
// comparisons
//

struct CmpIOpConversion
    : public ElementwiseOpConversionBase<triton::gpu::CmpIOp,
                                         CmpIOpConversion> {
  using Base =
      ElementwiseOpConversionBase<triton::gpu::CmpIOp, CmpIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  LLVM::ICmpOp createDestOp(triton::gpu::CmpIOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter, Type elemTy,
                            ValueRange operands, Location loc) const {
    return rewriter.create<LLVM::ICmpOp>(
        loc, elemTy, ArithCmpIPredicteToLLVM(op.predicate()), operands[0],
        operands[1]);
  }

  static LLVM::ICmpPredicate
  ArithCmpIPredicteToLLVM(arith::CmpIPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__)                                                    \
  case arith::CmpIPredicate::item__:                                           \
    return LLVM::ICmpPredicate::item__

      __PRED_ENUM(eq);
      __PRED_ENUM(ne);
      __PRED_ENUM(sgt);
      __PRED_ENUM(sge);
      __PRED_ENUM(slt);
      __PRED_ENUM(sle);
      __PRED_ENUM(ugt);
      __PRED_ENUM(uge);
      __PRED_ENUM(ult);
      __PRED_ENUM(ule);

#undef __PRED_ENUM
    }
    return LLVM::ICmpPredicate::eq;
  }
};

struct CmpFOpConversion
    : public ElementwiseOpConversionBase<triton::gpu::CmpFOp,
                                         CmpFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<triton::gpu::CmpFOp, CmpFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  static LLVM::FCmpOp createDestOp(triton::gpu::CmpFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, ValueRange operands,
                                   Location loc) {
    return rewriter.create<LLVM::FCmpOp>(
        loc, elemTy, ArithCmpFPredicteToLLVM(op.predicate()), operands[0],
        operands[1]);
  }

  static LLVM::FCmpPredicate
  ArithCmpFPredicteToLLVM(arith::CmpFPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__, item1__)                                           \
  case arith::CmpFPredicate::item__:                                           \
    return LLVM::FCmpPredicate::item1__

      __PRED_ENUM(OEQ, oeq);
      __PRED_ENUM(ONE, one);
      __PRED_ENUM(OGT, ogt);
      __PRED_ENUM(OGE, oge);
      __PRED_ENUM(OLT, olt);
      __PRED_ENUM(OLE, ole);
      __PRED_ENUM(ORD, ord);
      __PRED_ENUM(UEQ, ueq);
      __PRED_ENUM(UGT, ugt);
      __PRED_ENUM(UGE, uge);
      __PRED_ENUM(ULT, ult);
      __PRED_ENUM(ULE, ule);
      __PRED_ENUM(UNE, une);
      __PRED_ENUM(UNO, uno);
      __PRED_ENUM(AlwaysTrue, _true);
      __PRED_ENUM(AlwaysFalse, _false);

#undef __PRED_ENUM
    }
    return LLVM::FCmpPredicate::_true;
  }
};

struct ConvertLayoutOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::ConvertLayoutOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = op.src();
    Value dst = op.result();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (srcLayout.isa<BlockedEncodingAttr>() &&
        dstLayout.isa<SharedEncodingAttr>()) {
      return lowerBlockedToShared(op, adaptor, rewriter);
    }
    if (srcLayout.isa<SharedEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerSharedToDotOperand(op, adaptor, rewriter);
    }
    if ((srcLayout.isa<BlockedEncodingAttr>() ||
         srcLayout.isa<MmaEncodingAttr>() ||
         srcLayout.isa<SliceEncodingAttr>()) &&
        (dstLayout.isa<BlockedEncodingAttr>() ||
         dstLayout.isa<MmaEncodingAttr>() ||
         dstLayout.isa<SliceEncodingAttr>())) {
      return lowerDistributedToDistributed(op, adaptor, rewriter);
    }
    if (srcLayout.isa<MmaEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerMmaToDotOperand(op, adaptor, rewriter);
    }
    // TODO: to be implemented
    llvm_unreachable("unsupported layout conversion");
    return failure();
  }

  static bool isMmaToDotShortcut(MmaEncodingAttr &mmaLayout,
                                 DotOperandEncodingAttr &dotOperandLayout) {
    // dot_op<opIdx=0, parent=#mma> = #mma
    // when #mma = MmaEncoding<version=2, warpsPerCTA=[..., 1]>
    return mmaLayout.getWarpsPerCTA()[1] == 1 &&
           dotOperandLayout.getOpIdx() == 0 &&
           dotOperandLayout.getParent() == mmaLayout;
  }

  static void storeBlockedToShared(Value src, Value llSrc,
                                   ArrayRef<Value> srcStrides,
                                   ArrayRef<Value> srcIndices, Value dst,
                                   Value smemBase, Type elemPtrTy, Location loc,
                                   ConversionPatternRewriter &rewriter) {
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto srcShape = srcTy.getShape();
    assert(srcShape.size() == 2 && "Unexpected rank of insertSlice");

    auto elemTy = srcTy.getElementType();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto srcBlockedLayout = srcTy.getEncoding().cast<BlockedEncodingAttr>();
    auto dstSharedLayout = dstTy.getEncoding().cast<SharedEncodingAttr>();
    auto inOrd = srcBlockedLayout.getOrder();
    auto outOrd = dstSharedLayout.getOrder();
    if (inOrd != outOrd)
      llvm_unreachable(
          "blocked -> shared with different order not yet implemented");
    unsigned inVec =
        inOrd == outOrd ? srcBlockedLayout.getSizePerThread()[inOrd[0]] : 1;
    unsigned outVec = dstSharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    unsigned perPhase = dstSharedLayout.getPerPhase();
    unsigned maxPhase = dstSharedLayout.getMaxPhase();
    unsigned numElems = getElemsPerThread(srcTy);
    auto inVals = getElementsFromStruct(loc, llSrc, rewriter);
    auto srcAccumSizeInThreads =
        product<unsigned>(srcBlockedLayout.getSizePerThread());
    auto wordTy = vec_ty(elemTy, minVec);

    // TODO: [goostavz] We should make a cache for the calculation of
    // emitBaseIndexForBlockedLayout in case backend compiler not being able to
    // optimize that
    SmallVector<unsigned> srcShapePerCTA = getShapePerCTA(srcBlockedLayout);
    SmallVector<unsigned> reps{ceil<unsigned>(srcShape[0], srcShapePerCTA[0]),
                               ceil<unsigned>(srcShape[1], srcShapePerCTA[1])};

    // Visit each input value in the order they are placed in inVals
    //
    // Please note that the order was not awaring of blockLayout.getOrder(),
    // thus the adjacent elems may not belong to a same word. This could be
    // improved if we update the elements order by emitIndicesForBlockedLayout()
    SmallVector<unsigned> wordsInEachRep(2);
    wordsInEachRep[0] = inOrd[0] == 0
                            ? srcBlockedLayout.getSizePerThread()[0] / minVec
                            : srcBlockedLayout.getSizePerThread()[0];
    wordsInEachRep[1] = inOrd[0] == 0
                            ? srcBlockedLayout.getSizePerThread()[1]
                            : srcBlockedLayout.getSizePerThread()[1] / minVec;
    Value outVecVal = i32_val(outVec);
    Value minVecVal = i32_val(minVec);
    auto numWordsEachRep = product<unsigned>(wordsInEachRep);
    SmallVector<Value> wordVecs(numWordsEachRep);
    for (unsigned i = 0; i < numElems; ++i) {
      if (i % srcAccumSizeInThreads == 0) {
        // start of a replication
        for (unsigned w = 0; w < numWordsEachRep; ++w) {
          wordVecs[w] = undef(wordTy);
        }
      }
      unsigned linearIdxInNanoTile = i % srcAccumSizeInThreads;
      auto multiDimIdxInNanoTile = getMultiDimIndex<unsigned>(
          linearIdxInNanoTile, srcBlockedLayout.getSizePerThread(), inOrd);
      unsigned pos = multiDimIdxInNanoTile[inOrd[0]] % minVec;
      multiDimIdxInNanoTile[inOrd[0]] /= minVec;
      auto wordVecIdx = getLinearIndex<unsigned>(multiDimIdxInNanoTile,
                                                 wordsInEachRep, inOrd);
      wordVecs[wordVecIdx] =
          insert_element(wordTy, wordVecs[wordVecIdx], inVals[i], i32_val(pos));

      if (i % srcAccumSizeInThreads == srcAccumSizeInThreads - 1) {
        // end of replication, store the vectors into shared memory
        unsigned linearRepIdx = i / srcAccumSizeInThreads;
        auto multiDimRepIdx =
            getMultiDimIndex<unsigned>(linearRepIdx, reps, inOrd);
        for (unsigned linearWordIdx = 0; linearWordIdx < numWordsEachRep;
             ++linearWordIdx) {
          // step 1: recover the multidim_index from the index of
          // input_elements
          auto multiDimWordIdx =
              getMultiDimIndex<unsigned>(linearWordIdx, wordsInEachRep, inOrd);
          SmallVector<Value> multiDimIdx(2);
          auto wordOffset0 = multiDimRepIdx[0] * srcShapePerCTA[0] +
                             multiDimWordIdx[0] * (inOrd[0] == 0 ? minVec : 1);
          auto wordOffset1 = multiDimRepIdx[1] * srcShapePerCTA[1] +
                             multiDimWordIdx[1] * (inOrd[0] == 1 ? minVec : 1);
          multiDimIdx[0] = add(srcIndices[0], i32_val(wordOffset0));
          multiDimIdx[1] = add(srcIndices[1], i32_val(wordOffset1));

          // step 2: do swizzling
          Value remained = urem(multiDimIdx[outOrd[0]], outVecVal);
          multiDimIdx[outOrd[0]] = udiv(multiDimIdx[outOrd[0]], outVecVal);
          Value off_1 = mul(multiDimIdx[outOrd[1]], srcStrides[outOrd[1]]);
          Value phaseId = udiv(multiDimIdx[outOrd[1]], i32_val(perPhase));
          phaseId = urem(phaseId, i32_val(maxPhase));
          Value off_0 = xor_(multiDimIdx[outOrd[0]], phaseId);
          off_0 = mul(off_0, outVecVal);
          remained = udiv(remained, minVecVal);
          off_0 = add(off_0, mul(remained, minVecVal));
          Value offset = add(off_1, off_0);

          // step 3: store
          Value smemAddr = gep(elemPtrTy, smemBase, offset);
          smemAddr = bitcast(smemAddr, ptr_ty(wordTy, 3));
          store(wordVecs[linearWordIdx], smemAddr);
        }
      }
    }
  }

private:
  SmallVector<Value> getMultiDimOffset(Attribute layout, Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       unsigned elemId, ArrayRef<int64_t> shape,
                                       ArrayRef<unsigned> multiDimCTAInRepId,
                                       ArrayRef<unsigned> shapePerCTA) const {
    unsigned rank = shape.size();
    if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
      auto multiDimOffsetFirstElem =
          emitBaseIndexForBlockedLayout(loc, rewriter, blockedLayout, shape);
      SmallVector<Value> multiDimOffset(rank);
      SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
          elemId, getSizePerThread(layout), getOrder(layout));
      for (unsigned d = 0; d < rank; ++d) {
        multiDimOffset[d] = add(multiDimOffsetFirstElem[d],
                                idx_val(multiDimCTAInRepId[d] * shapePerCTA[d] +
                                        multiDimElemId[d]));
      }
      return multiDimOffset;
    }
    if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
      unsigned dim = sliceLayout.getDim();
      auto multiDimOffsetParent =
          getMultiDimOffset(sliceLayout.getParent(), loc, rewriter, elemId,
                            sliceLayout.paddedShape(shape),
                            sliceLayout.paddedShape(multiDimCTAInRepId),
                            sliceLayout.paddedShape(shapePerCTA));
      SmallVector<Value> multiDimOffset(rank);
      for (unsigned d = 0; d < rank + 1; ++d) {
        if (d == dim)
          continue;
        unsigned slicedD = d < dim ? d : (d - 1);
        multiDimOffset[slicedD] = multiDimOffsetParent[d];
      }
      return multiDimOffset;
    }
    if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
      SmallVector<Value> mmaColIdx(4);
      SmallVector<Value> mmaRowIdx(2);
      Value threadId = getThreadId(rewriter, loc);
      Value warpSize = idx_val(32);
      Value laneId = urem(threadId, warpSize);
      Value warpId = udiv(threadId, warpSize);
      // TODO: fix the bug in MMAEncodingAttr document
      SmallVector<Value> multiDimWarpId(2);
      multiDimWarpId[0] = urem(warpId, idx_val(mmaLayout.getWarpsPerCTA()[0]));
      multiDimWarpId[1] = udiv(warpId, idx_val(mmaLayout.getWarpsPerCTA()[0]));
      Value _1 = idx_val(1);
      Value _2 = idx_val(2);
      Value _4 = idx_val(4);
      Value _8 = idx_val(8);
      Value _16 = idx_val(16);
      if (mmaLayout.isAmpere()) {
        multiDimWarpId[0] = urem(multiDimWarpId[0], idx_val(shape[0] / 16));
        multiDimWarpId[1] = urem(multiDimWarpId[1], idx_val(shape[1] / 8));
        Value mmaGrpId = udiv(laneId, _4);
        Value mmaGrpIdP8 = add(mmaGrpId, _8);
        Value mmaThreadIdInGrp = urem(laneId, _4);
        Value mmaThreadIdInGrpM2 = mul(mmaThreadIdInGrp, _2);
        Value mmaThreadIdInGrpM2P1 = add(mmaThreadIdInGrpM2, _1);
        Value rowWarpOffset = mul(multiDimWarpId[0], _16);
        mmaRowIdx[0] = add(mmaGrpId, rowWarpOffset);
        mmaRowIdx[1] = add(mmaGrpIdP8, rowWarpOffset);
        Value colWarpOffset = mul(multiDimWarpId[1], _8);
        mmaColIdx[0] = add(mmaThreadIdInGrpM2, colWarpOffset);
        mmaColIdx[1] = add(mmaThreadIdInGrpM2P1, colWarpOffset);
      } else if (mmaLayout.isVolta()) {
        multiDimWarpId[0] = urem(multiDimWarpId[0], idx_val(shape[0] / 16));
        multiDimWarpId[1] = urem(multiDimWarpId[1], idx_val(shape[1] / 16));
        Value laneIdDiv16 = udiv(laneId, _16);
        Value laneIdRem16 = urem(laneId, _16);
        Value laneIdRem2 = urem(laneId, _2);
        Value laneIdRem16Div8 = udiv(laneIdRem16, _8);
        Value laneIdRem16Div4 = udiv(laneIdRem16, _4);
        Value laneIdRem16Div4Rem2 = urem(laneIdRem16Div4, _2);
        Value laneIdRem4Div2 = udiv(urem(laneId, _4), _2);
        Value rowWarpOffset = mul(multiDimWarpId[0], _16);
        Value colWarpOffset = mul(multiDimWarpId[1], _16);
        mmaRowIdx[0] =
            add(add(mul(laneIdDiv16, _8), mul(laneIdRem16Div4Rem2, _4)),
                laneIdRem2);
        mmaRowIdx[0] = add(mmaRowIdx[0], rowWarpOffset);
        mmaRowIdx[1] = add(mmaRowIdx[0], _2);
        mmaColIdx[0] = add(mul(laneIdRem16Div8, _4), mul(laneIdRem4Div2, _2));
        mmaColIdx[0] = add(mmaColIdx[0], colWarpOffset);
        mmaColIdx[1] = add(mmaColIdx[0], _1);
        mmaColIdx[2] = add(mmaColIdx[0], _8);
        mmaColIdx[3] = add(mmaColIdx[0], idx_val(9));
      } else {
        llvm_unreachable("Unexpected MMALayout version");
      }

      assert(rank == 2);
      SmallVector<Value> multiDimOffset(rank);
      if (mmaLayout.isAmpere()) {
        multiDimOffset[0] = elemId < 2 ? mmaRowIdx[0] : mmaRowIdx[1];
        multiDimOffset[1] = elemId % 2 == 0 ? mmaColIdx[0] : mmaColIdx[1];
        multiDimOffset[0] = add(
            multiDimOffset[0], idx_val(multiDimCTAInRepId[0] * shapePerCTA[0]));
        multiDimOffset[1] = add(
            multiDimOffset[1], idx_val(multiDimCTAInRepId[1] * shapePerCTA[1]));
      } else if (mmaLayout.isVolta()) {
        // the order of elements in a thread:
        //   c0, c1, ...  c4, c5
        //   c2, c3, ...  c6, c7
        if (elemId < 2) {
          multiDimOffset[0] = mmaRowIdx[0];
          multiDimOffset[1] = mmaColIdx[elemId % 2];
        } else if (elemId >= 2 && elemId < 4) {
          multiDimOffset[0] = mmaRowIdx[1];
          multiDimOffset[1] = mmaColIdx[elemId % 2];
        } else if (elemId >= 4 && elemId < 6) {
          multiDimOffset[0] = mmaRowIdx[0];
          multiDimOffset[1] = mmaColIdx[elemId % 2 + 2];
        } else if (elemId >= 6) {
          multiDimOffset[0] = mmaRowIdx[1];
          multiDimOffset[1] = mmaColIdx[elemId % 2 + 2];
        }
        multiDimOffset[0] = add(
            multiDimOffset[0], idx_val(multiDimCTAInRepId[0] * shapePerCTA[0]));
        multiDimOffset[1] = add(
            multiDimOffset[1], idx_val(multiDimCTAInRepId[1] * shapePerCTA[1]));
      } else {
        llvm_unreachable("Unexpected MMALayout version");
      }
      return multiDimOffset;
    }
    llvm_unreachable("unexpected layout in getMultiDimOffset");
  }

  // shared memory rd/st for blocked or mma layout with data padding
  void processReplica(Location loc, ConversionPatternRewriter &rewriter,
                      bool stNotRd, RankedTensorType type,
                      ArrayRef<unsigned> numCTAsEachRep,
                      ArrayRef<unsigned> multiDimRepId, unsigned vec,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> outOrd, SmallVector<Value> &vals,
                      Value smemBase) const;

  // blocked/mma -> blocked/mma.
  // Data padding in shared memory to avoid bank conflict.
  LogicalResult
  lowerDistributedToDistributed(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const;

  // blocked -> shared.
  // Swizzling in shared memory to avoid bank conflict. Normally used for
  // A/B operands of dots.
  LogicalResult lowerBlockedToShared(triton::gpu::ConvertLayoutOp op,
                                     OpAdaptor adaptor,
                                     ConversionPatternRewriter &rewriter) const;

  // shared -> mma_operand
  LogicalResult
  lowerSharedToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const;

  // mma -> dot_operand
  LogicalResult lowerMmaToDotOperand(triton::gpu::ConvertLayoutOp op,
                                     OpAdaptor adaptor,
                                     ConversionPatternRewriter &rewriter) const;

  // shared -> dot_operand if the result layout is mma
  Value lowerSharedToDotOperandMMA(
      triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, const MmaEncodingAttr &mmaLayout,
      const DotOperandEncodingAttr &dotOperandLayout, bool isOuter) const;
};

void ConvertLayoutOpConversion::processReplica(
    Location loc, ConversionPatternRewriter &rewriter, bool stNotRd,
    RankedTensorType type, ArrayRef<unsigned> numCTAsEachRep,
    ArrayRef<unsigned> multiDimRepId, unsigned vec,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> outOrd,
    SmallVector<Value> &vals, Value smemBase) const {
  auto accumNumCTAsEachRep = product<unsigned>(numCTAsEachRep);
  auto layout = type.getEncoding();
  auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>();
  auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>();
  auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>();
  auto rank = type.getRank();
  auto sizePerThread = getSizePerThread(layout);
  auto accumSizePerThread = product<unsigned>(sizePerThread);
  SmallVector<unsigned> numCTAs(rank);
  auto shapePerCTA = getShapePerCTA(layout);
  auto order = getOrder(layout);
  for (unsigned d = 0; d < rank; ++d) {
    numCTAs[d] = ceil<unsigned>(type.getShape()[d], shapePerCTA[d]);
  }
  auto elemTy = type.getElementType();
  bool isInt1 = elemTy.isInteger(1);
  bool isPtr = elemTy.isa<triton::PointerType>();
  auto llvmElemTyOrig = getTypeConverter()->convertType(elemTy);
  if (isInt1)
    elemTy = IntegerType::get(elemTy.getContext(), 8);
  else if (isPtr)
    elemTy = IntegerType::get(elemTy.getContext(), 64);

  auto llvmElemTy = getTypeConverter()->convertType(elemTy);

  for (unsigned ctaId = 0; ctaId < accumNumCTAsEachRep; ++ctaId) {
    auto multiDimCTAInRepId =
        getMultiDimIndex<unsigned>(ctaId, numCTAsEachRep, order);
    SmallVector<unsigned> multiDimCTAId(rank);
    for (const auto &it : llvm::enumerate(multiDimCTAInRepId)) {
      auto d = it.index();
      multiDimCTAId[d] = multiDimRepId[d] * numCTAsEachRep[d] + it.value();
    }

    auto linearCTAId = getLinearIndex<unsigned>(multiDimCTAId, numCTAs, order);
    // TODO: This is actually redundant index calculation, we should
    //       consider of caching the index calculation result in case
    //       of performance issue observed.
    for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
      SmallVector<Value> multiDimOffset =
          getMultiDimOffset(layout, loc, rewriter, elemId, type.getShape(),
                            multiDimCTAInRepId, shapePerCTA);
      Value offset =
          linearize(rewriter, loc, multiDimOffset, paddedRepShape, outOrd);

      auto elemPtrTy = ptr_ty(llvmElemTy, 3);
      Value ptr = gep(elemPtrTy, smemBase, offset);
      auto vecTy = vec_ty(llvmElemTy, vec);
      ptr = bitcast(ptr, ptr_ty(vecTy, 3));
      if (stNotRd) {
        Value valVec = undef(vecTy);
        for (unsigned v = 0; v < vec; ++v) {
          auto currVal = vals[elemId + linearCTAId * accumSizePerThread + v];
          if (isInt1)
            currVal = zext(llvmElemTy, currVal);
          else if (isPtr)
            currVal = ptrtoint(llvmElemTy, currVal);
          valVec = insert_element(vecTy, valVec, currVal, idx_val(v));
        }
        store(valVec, ptr);
      } else {
        Value valVec = load(ptr);
        for (unsigned v = 0; v < vec; ++v) {
          Value currVal = extract_element(llvmElemTy, valVec, idx_val(v));
          if (isInt1)
            currVal =
                icmp_ne(currVal, rewriter.create<LLVM::ConstantOp>(
                                     loc, i8_ty, rewriter.getI8IntegerAttr(0)));
          else if (isPtr)
            currVal = inttoptr(llvmElemTyOrig, currVal);
          vals[elemId + linearCTAId * accumSizePerThread + v] = currVal;
        }
      }
    }
  }
}

LogicalResult ConvertLayoutOpConversion::lowerDistributedToDistributed(
    triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value src = op.src();
  Value dst = op.result();
  auto srcTy = src.getType().cast<RankedTensorType>();
  auto dstTy = dst.getType().cast<RankedTensorType>();
  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();
  auto llvmElemTy = getTypeConverter()->convertType(dstTy.getElementType());
  Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
  auto elemPtrTy = ptr_ty(llvmElemTy, 3);
  smemBase = bitcast(smemBase, elemPtrTy);
  auto shape = dstTy.getShape();
  unsigned rank = dstTy.getRank();
  SmallVector<unsigned> numReplicates(rank);
  SmallVector<unsigned> inNumCTAsEachRep(rank);
  SmallVector<unsigned> outNumCTAsEachRep(rank);
  SmallVector<unsigned> inNumCTAs(rank);
  SmallVector<unsigned> outNumCTAs(rank);
  auto srcShapePerCTA = getShapePerCTA(srcLayout);
  auto dstShapePerCTA = getShapePerCTA(dstLayout);
  for (unsigned d = 0; d < rank; ++d) {
    unsigned inPerCTA = std::min<unsigned>(shape[d], srcShapePerCTA[d]);
    unsigned outPerCTA = std::min<unsigned>(shape[d], dstShapePerCTA[d]);
    unsigned maxPerCTA = std::max(inPerCTA, outPerCTA);
    numReplicates[d] = ceil<unsigned>(shape[d], maxPerCTA);
    inNumCTAsEachRep[d] = maxPerCTA / inPerCTA;
    outNumCTAsEachRep[d] = maxPerCTA / outPerCTA;
    assert(maxPerCTA % inPerCTA == 0 && maxPerCTA % outPerCTA == 0);
    inNumCTAs[d] = ceil<unsigned>(shape[d], inPerCTA);
    outNumCTAs[d] = ceil<unsigned>(shape[d], outPerCTA);
  }
  // Potentially we need to store for multiple CTAs in this replication
  auto accumNumReplicates = product<unsigned>(numReplicates);
  // unsigned elems = getElemsPerThread(srcTy);
  auto vals = getElementsFromStruct(loc, adaptor.src(), rewriter);
  unsigned inVec = 0;
  unsigned outVec = 0;
  auto paddedRepShape = getScratchConfigForCvtLayout(op, inVec, outVec);

  unsigned outElems = getElemsPerThread(dstTy);
  auto outOrd = getOrder(dstLayout);
  SmallVector<Value> outVals(outElems);

  for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
    auto multiDimRepId =
        getMultiDimIndex<unsigned>(repId, numReplicates, outOrd);
    if (repId != 0)
      barrier();
    if (srcLayout.isa<BlockedEncodingAttr>() ||
        srcLayout.isa<SliceEncodingAttr>() ||
        srcLayout.isa<MmaEncodingAttr>()) {
      processReplica(loc, rewriter, /*stNotRd*/ true, srcTy, inNumCTAsEachRep,
                     multiDimRepId, inVec, paddedRepShape, outOrd, vals,
                     smemBase);
    } else {
      assert(0 && "ConvertLayout with input layout not implemented");
      return failure();
    }
    barrier();
    if (dstLayout.isa<BlockedEncodingAttr>() ||
        dstLayout.isa<SliceEncodingAttr>() ||
        dstLayout.isa<MmaEncodingAttr>()) {
      processReplica(loc, rewriter, /*stNotRd*/ false, dstTy, outNumCTAsEachRep,
                     multiDimRepId, outVec, paddedRepShape, outOrd, outVals,
                     smemBase);
    } else {
      assert(0 && "ConvertLayout with output layout not implemented");
      return failure();
    }
  }

  SmallVector<Type> types(outElems, llvmElemTy);
  auto *ctx = llvmElemTy.getContext();
  Type structTy = struct_ty(types);
  Value result = getStructFromElements(loc, outVals, rewriter, structTy);
  rewriter.replaceOp(op, result);

  return success();
}

LogicalResult ConvertLayoutOpConversion::lowerBlockedToShared(
    triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value src = op.src();
  Value dst = op.result();
  auto srcTy = src.getType().cast<RankedTensorType>();
  auto srcShape = srcTy.getShape();
  auto dstTy = dst.getType().cast<RankedTensorType>();
  auto dstShape = dstTy.getShape();
  assert(srcShape.size() == 2 &&
         "Unexpected rank of ConvertLayout(blocked->shared)");
  auto srcBlockedLayout = srcTy.getEncoding().cast<BlockedEncodingAttr>();
  auto dstSharedLayout = dstTy.getEncoding().cast<SharedEncodingAttr>();
  auto inOrd = srcBlockedLayout.getOrder();
  auto outOrd = dstSharedLayout.getOrder();
  Value smemBase = getSharedMemoryBase(loc, rewriter, dst);
  auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
  auto elemPtrTy = ptr_ty(getTypeConverter()->convertType(elemTy), 3);
  smemBase = bitcast(smemBase, elemPtrTy);

  auto srcStrides = getStridesFromShapeAndOrder(srcShape, inOrd, loc, rewriter);
  auto srcIndices =
      emitBaseIndexForBlockedLayout(loc, rewriter, srcBlockedLayout, srcShape);
  storeBlockedToShared(src, adaptor.src(), srcStrides, srcIndices, dst,
                       smemBase, elemPtrTy, loc, rewriter);

  auto smemObj = SharedMemoryObject(smemBase, dstShape, outOrd, loc, rewriter);
  auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
  rewriter.replaceOp(op, retVal);
  return success();
}

LogicalResult ConvertLayoutOpConversion::lowerMmaToDotOperand(
    triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto srcTy = op.src().getType().cast<RankedTensorType>();
  auto dstTy = op.result().getType().cast<RankedTensorType>();
  auto srcLayout = srcTy.getEncoding();
  auto dstLayout = dstTy.getEncoding();
  auto srcMmaLayout = srcLayout.cast<MmaEncodingAttr>();
  auto dstDotLayout = dstLayout.cast<DotOperandEncodingAttr>();
  if (isMmaToDotShortcut(srcMmaLayout, dstDotLayout)) {
    // get source values
    auto vals = getElementsFromStruct(loc, adaptor.src(), rewriter);
    unsigned elems = getElemsPerThread(srcTy);
    Type elemTy = this->getTypeConverter()->convertType(srcTy.getElementType());
    // for the destination type, we need to pack values together
    // so they can be consumed by tensor core operations
    unsigned vecSize =
        std::max<unsigned>(32 / elemTy.getIntOrFloatBitWidth(), 1);
    Type vecTy = vec_ty(elemTy, vecSize);
    SmallVector<Type> types(elems / vecSize, vecTy);
    SmallVector<Value> vecVals;
    for (unsigned i = 0; i < elems; i += vecSize) {
      Value packed = rewriter.create<LLVM::UndefOp>(loc, vecTy);
      for (unsigned j = 0; j < vecSize; j++)
        packed = insert_element(vecTy, packed, vals[i + j], i32_val(j));
      vecVals.push_back(packed);
    }

    // This needs to be ordered the same way that
    // ldmatrix.x4 would order it
    // TODO: this needs to be refactor so we don't
    // implicitly depends on how emitOffsetsForMMAV2
    // is implemented
    SmallVector<Value> reorderedVals;
    for (unsigned i = 0; i < vecVals.size(); i += 4) {
      reorderedVals.push_back(vecVals[i]);
      reorderedVals.push_back(vecVals[i + 2]);
      reorderedVals.push_back(vecVals[i + 1]);
      reorderedVals.push_back(vecVals[i + 3]);
    }

    // return composeValuesToDotOperandLayoutStruct(ha, numRepM, numRepK);

    Type structTy = LLVM::LLVMStructType::getLiteral(this->getContext(), types);
    Value view = getStructFromElements(loc, reorderedVals, rewriter, structTy);
    rewriter.replaceOp(op, view);
    return success();
  }
  return failure();
}

struct InsertSliceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<tensor::InsertSliceOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      tensor::InsertSliceOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = insert_slice %src into %dst[%offsets]
    Location loc = op->getLoc();
    Value dst = op.dest();
    Value src = op.source();
    Value res = op.result();
    assert(allocation->getBufferId(res) == Allocation::InvalidBufferId &&
           "Only support in-place insert_slice for now");

    auto srcTy = src.getType().dyn_cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding().dyn_cast<BlockedEncodingAttr>();
    auto srcShape = srcTy.getShape();
    assert(srcLayout && "Unexpected srcLayout in InsertSliceOpConversion");

    auto dstTy = dst.getType().dyn_cast<RankedTensorType>();
    auto dstLayout = dstTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    auto llDst = adaptor.dest();
    assert(dstLayout && "Unexpected dstLayout in InsertSliceOpConversion");
    assert(op.hasUnitStride() &&
           "Only unit stride supported by InsertSliceOpConversion");

    // newBase = base + offset
    // Triton support either static and dynamic offsets
    auto smemObj = getSharedMemoryObjectFromStruct(loc, llDst, rewriter);
    SmallVector<Value, 4> offsets;
    SmallVector<Value, 4> srcStrides;
    auto mixedOffsets = op.getMixedOffsets();
    for (auto i = 0; i < mixedOffsets.size(); ++i) {
      if (op.isDynamicOffset(i)) {
        offsets.emplace_back(adaptor.offsets()[i]);
      } else {
        offsets.emplace_back(i32_val(op.getStaticOffset(i)));
      }
      // Like insert_slice_async, we only support slice from one dimension,
      // which has a slice size of 1
      if (op.getStaticSize(i) != 1) {
        srcStrides.emplace_back(smemObj.strides[i]);
      }
    }

    // Compute the offset based on the original strides of the shared memory
    // object
    auto offset = dot(rewriter, loc, offsets, smemObj.strides);
    auto llvmElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto elemPtrTy = ptr_ty(llvmElemTy, 3);
    auto smemBase = gep(elemPtrTy, smemObj.base, offset);

    auto llSrc = adaptor.source();
    auto srcIndices =
        emitBaseIndexForBlockedLayout(loc, rewriter, srcLayout, srcShape);
    ConvertLayoutOpConversion::storeBlockedToShared(src, llSrc, srcStrides,
                                                    srcIndices, dst, smemBase,
                                                    elemPtrTy, loc, rewriter);
    // Barrier is not necessary.
    // The membar pass knows that it writes to shared memory and will handle it
    // properly.
    rewriter.replaceOp(op, llDst);
    return success();
  }
};

/// ====================== dot codegen begin ==========================

struct DotOpConversion : public ConvertTritonGPUOpToLLVMPattern<triton::DotOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::DotOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // D = A * B + C
    Value A = op.a();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShape = A.getType().cast<RankedTensorType>().getShape();
    size_t reduceAxis = 1;
    unsigned K = AShape[reduceAxis];
    bool isOuter = K == 1;

    bool isMMA = D.getType()
                     .cast<RankedTensorType>()
                     .getEncoding()
                     .isa<MmaEncodingAttr>();
    MmaEncodingAttr mmaLayout;
    if (isMMA)
      mmaLayout = D.getType()
                      .cast<RankedTensorType>()
                      .getEncoding()
                      .cast<MmaEncodingAttr>();

    bool isHMMA = isDotHMMA(op);
    if (!isOuter && isMMA && isHMMA) {
      if (mmaLayout.isVolta())
        return convertMMA884(op, adaptor, rewriter);
      if (mmaLayout.isAmpere())
        return convertMMA16816(op, adaptor, rewriter);

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotOp to LLVM.");
    }

    if (op.getType().cast<RankedTensorType>().getElementType().isF32() &&
        A.getType().cast<RankedTensorType>().getElementType().isF32() &&
        !op.allowTF32())
      return convertFMADot(op, adaptor, rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }

  // Tell whether a DotOp support HMMA.
  // This is port from the master branch, the original logic is retained.
  static bool isDotHMMA(DotOp op) {
    auto a = op.a();
    auto b = op.b();
    auto c = op.c();
    auto d = op.getResult();
    auto aTensorTy = a.getType().cast<RankedTensorType>();
    auto bTensorTy = b.getType().cast<RankedTensorType>();
    auto cTensorTy = c.getType().cast<RankedTensorType>();
    auto dTensorTy = d.getType().cast<RankedTensorType>();

    if (!dTensorTy.getEncoding().isa<MmaEncodingAttr>())
      return false;

    auto mmaLayout = dTensorTy.getEncoding().cast<MmaEncodingAttr>();
    auto aElemTy = aTensorTy.getElementType();
    auto bElemTy = bTensorTy.getElementType();

    assert((mmaLayout.isVolta() || mmaLayout.isAmpere()) &&
           "Unexpected MMA layout version found");
    // Refer to mma section for the data type supported by Volta and Hopper
    // Tensor Core in
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f16
    return (aElemTy.isF16() && bElemTy.isF16()) ||
           (aElemTy.isBF16() && bElemTy.isBF16()) ||
           (aElemTy.isF32() && bElemTy.isF32() && op.allowTF32() &&
            mmaLayout.getVersionMajor() >= 2) ||
           (aElemTy.isInteger(8) && bElemTy.isInteger(8) &&
            mmaLayout.getVersionMajor() >= 2);
  }

  // Tell whether a DotOp support HMMA by the operand type(either $a or $b).
  // We cannot get both the operand types(in TypeConverter), here we assume the
  // types of both the operands are identical here.
  // TODO[Superjomn]: Find a better way to implement it.
  static bool isDotHMMA(TensorType operand, int mmaVersion) {
    auto elemTy = operand.getElementType();
    return elemTy.isF16() || elemTy.isBF16() ||
           (elemTy.isF32() && mmaVersion >= 2) ||
           (elemTy.isInteger(8) && mmaVersion >= 2);
  }

private:
  // Convert to mma.m16n8k16
  LogicalResult convertMMA16816(triton::DotOp a, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const;
  /// Convert to mma.m8n8k4
  LogicalResult convertMMA884(triton::DotOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const;

  LogicalResult convertFMADot(triton::DotOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const;
};

Value ConvertLayoutOpConversion::lowerSharedToDotOperandMMA(
    triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter, const MmaEncodingAttr &mmaLayout,
    const DotOperandEncodingAttr &dotOperandLayout, bool isOuter) const {
  auto loc = op.getLoc();
  Value src = op.src();
  Value dst = op.result();
  auto dstTensorTy = dst.getType().cast<RankedTensorType>();
  bool isHMMA =
      DotOpConversion::isDotHMMA(dstTensorTy, mmaLayout.getVersionMajor());

  auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.src(), rewriter);
  Value res;

  if (!isOuter && mmaLayout.isAmpere() && isHMMA) { // tensor core v2
    MMA16816ConversionHelper mmaHelper(src.getType(), mmaLayout,
                                       getThreadId(rewriter, loc), rewriter,
                                       getTypeConverter(), op.getLoc());

    if (dotOperandLayout.getOpIdx() == 0) {
      // operand $a
      res = mmaHelper.loadA(src, smemObj);
    } else if (dotOperandLayout.getOpIdx() == 1) {
      // operand $b
      res = mmaHelper.loadB(src, smemObj);
    }
  } else if (!isOuter && mmaLayout.isVolta() && isHMMA) { // tensor core v1
    DotOpMmaV1ConversionHelper helper(mmaLayout);
    bool isMMAv1Row =
        dotOperandLayout.getIsMMAv1Row().cast<BoolAttr>().getValue();
    auto srcSharedLayout = src.getType()
                               .cast<RankedTensorType>()
                               .getEncoding()
                               .cast<SharedEncodingAttr>();

    // Can only convert [1, 0] to row or [0, 1] to col for now
    if ((srcSharedLayout.getOrder()[0] == 1 && !isMMAv1Row) ||
        (srcSharedLayout.getOrder()[0] == 0 && isMMAv1Row)) {
      llvm::errs() << "Unsupported Shared -> DotOperand[MMAv1] conversion\n";
      return Value();
    }

    if (dotOperandLayout.getOpIdx() == 0) { // operand $a
      // TODO[Superjomn]: transA is not available here.
      bool transA = false;
      res = helper.loadA(src, transA, smemObj, getThreadId(rewriter, loc), loc,
                         rewriter);
    } else if (dotOperandLayout.getOpIdx() == 1) { // operand $b
      // TODO[Superjomn]: transB is not available here.
      bool transB = false;
      res = helper.loadB(src, transB, smemObj, getThreadId(rewriter, loc), loc,
                         rewriter);
    }
  } else {
    assert(false && "Unsupported mma layout found");
  }
  return res;
}

LogicalResult ConvertLayoutOpConversion::lowerSharedToDotOperand(
    triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value src = op.src();
  Value dst = op.result();
  auto dstTensorTy = dst.getType().cast<RankedTensorType>();
  auto srcTensorTy = src.getType().cast<RankedTensorType>();
  auto dotOperandLayout =
      dstTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
  auto sharedLayout = srcTensorTy.getEncoding().cast<SharedEncodingAttr>();

  bool isOuter{};
  int K{};
  if (dotOperandLayout.getOpIdx() == 0) // $a
    K = dstTensorTy.getShape()[sharedLayout.getOrder()[0]];
  else // $b
    K = dstTensorTy.getShape()[sharedLayout.getOrder()[1]];
  isOuter = K == 1;

  Value res;
  if (auto mmaLayout =
          dotOperandLayout.getParent().dyn_cast_or_null<MmaEncodingAttr>()) {
    res = lowerSharedToDotOperandMMA(op, adaptor, rewriter, mmaLayout,
                                     dotOperandLayout, isOuter);
  } else if (auto blockedLayout =
                 dotOperandLayout.getParent()
                     .dyn_cast_or_null<BlockedEncodingAttr>()) {
    auto dotOpLayout = dstTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    DotOpFMAConversionHelper helper(blockedLayout);
    auto thread = getThreadId(rewriter, loc);
    if (dotOpLayout.getOpIdx() == 0) { // $a
      res = helper.loadA(src, adaptor.src(), blockedLayout, thread, loc,
                         rewriter);
    } else { // $b
      res = helper.loadB(src, adaptor.src(), blockedLayout, thread, loc,
                         rewriter);
    }
  } else {
    assert(false && "Unsupported dot operand layout found");
  }

  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult
DotOpConversion::convertMMA16816(triton::DotOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto mmaLayout = op.getResult()
                       .getType()
                       .cast<RankedTensorType>()
                       .getEncoding()
                       .cast<MmaEncodingAttr>();

  Value A = op.a();
  Value B = op.b();
  Value C = op.c();

  MMA16816ConversionHelper mmaHelper(A.getType(), mmaLayout,
                                     getThreadId(rewriter, loc), rewriter,
                                     getTypeConverter(), loc);

  auto ATensorTy = A.getType().cast<RankedTensorType>();
  auto BTensorTy = B.getType().cast<RankedTensorType>();

  assert(ATensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
         BTensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
         "Both $a and %b should be DotOperand layout.");

  Value loadedA, loadedB, loadedC;
  loadedA = adaptor.a();
  loadedB = adaptor.b();
  loadedC = mmaHelper.loadC(op.c(), adaptor.c());

  return mmaHelper.convertDot(A, B, C, op.d(), loadedA, loadedB, loadedC, op,
                              adaptor);
}

// Simply port the old code here to avoid large difference and make debugging
// and profiling easier.
LogicalResult
DotOpConversion::convertMMA884(triton::DotOp op, DotOpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  auto *ctx = op.getContext();
  auto loc = op.getLoc();

  Value A = op.a();
  Value B = op.b();
  Value D = op.getResult();
  auto mmaLayout = D.getType()
                       .cast<RankedTensorType>()
                       .getEncoding()
                       .cast<MmaEncodingAttr>();
  auto ALayout = A.getType()
                     .cast<RankedTensorType>()
                     .getEncoding()
                     .cast<DotOperandEncodingAttr>();
  auto BLayout = B.getType()
                     .cast<RankedTensorType>()
                     .getEncoding()
                     .cast<DotOperandEncodingAttr>();

  auto ATensorTy = A.getType().cast<RankedTensorType>();
  auto BTensorTy = B.getType().cast<RankedTensorType>();
  auto DTensorTy = D.getType().cast<RankedTensorType>();
  auto AShape = ATensorTy.getShape();
  auto BShape = BTensorTy.getShape();
  auto DShape = DTensorTy.getShape();
  auto wpt = mmaLayout.getWarpsPerCTA();

  bool isARow = ALayout.getIsMMAv1Row().cast<BoolAttr>().getValue();
  bool isBRow = BLayout.getIsMMAv1Row().cast<BoolAttr>().getValue();

  DotOpMmaV1ConversionHelper helper(mmaLayout);

  unsigned numM = helper.getNumM(AShape, isARow);
  unsigned numN = helper.getNumN(BShape, isBRow);
  unsigned NK = AShape[1];

  auto has = helper.extractLoadedOperand(adaptor.a(), NK, rewriter);
  auto hbs = helper.extractLoadedOperand(adaptor.b(), NK, rewriter);

  // Initialize accumulators with external values, the acc holds the accumulator
  // value that is shared between the MMA instructions inside a DotOp, we can
  // call the order of the values the accumulator-internal order.
  SmallVector<Value> acc = getElementsFromStruct(loc, adaptor.c(), rewriter);
  size_t resSize = acc.size();

  // The resVals holds the final result of the DotOp.
  // NOTE The current order of resVals is different from acc, we call it the
  // accumulator-external order. and
  SmallVector<Value> resVals(resSize);

  auto getIdx = [&](int m, int n) {
    std::vector<size_t> idx{{
        (m * 2 + 0) + (n * 4 + 0) * numM, // row0
        (m * 2 + 0) + (n * 4 + 1) * numM,
        (m * 2 + 1) + (n * 4 + 0) * numM, // row1
        (m * 2 + 1) + (n * 4 + 1) * numM,
        (m * 2 + 0) + (n * 4 + 2) * numM, // row2
        (m * 2 + 0) + (n * 4 + 3) * numM,
        (m * 2 + 1) + (n * 4 + 2) * numM, // row3
        (m * 2 + 1) + (n * 4 + 3) * numM,
    }};
    return idx;
  };

  { // convert the acc's value from accumuator-external order to
    // accumulator-internal order.
    SmallVector<Value> accInit(acc.size());

    for (unsigned m = 0; m < numM / 2; ++m)
      for (unsigned n = 0; n < numN / 2; ++n) {
        auto idx = getIdx(m, n);
        for (unsigned i = 0; i < 8; ++i)
          accInit[idx[i]] = acc[(m * numN / 2 + n) * 8 + i];
      }

    acc = accInit;
  }

  auto callMMA = [&](unsigned m, unsigned n, unsigned k) {
    auto ha = has.at({m, k});
    auto hb = hbs.at({n, k});

    PTXBuilder builder;
    auto idx = getIdx(m, n);

    auto *resOprs = builder.newListOperand(8, "=f");
    auto *AOprs = builder.newListOperand({
        {ha.first, "r"},
        {ha.second, "r"},
    });

    auto *BOprs = builder.newListOperand({
        {hb.first, "r"},
        {hb.second, "r"},
    });
    auto *COprs = builder.newListOperand();
    for (int i = 0; i < 8; ++i)
      COprs->listAppend(builder.newOperand(acc[idx[i]], std::to_string(i)));

    auto mma = builder.create("mma.sync.aligned.m8n8k4")
                   ->o(isARow ? "row" : "col")
                   .o(isBRow ? "row" : "col")
                   .o("f32.f16.f16.f32");

    mma(resOprs, AOprs, BOprs, COprs);

    Value res = builder.launch(rewriter, loc, helper.getMmaRetType(ATensorTy));

    auto getIntAttr = [&](int v) {
      return ArrayAttr::get(ctx, {IntegerAttr::get(i32_ty, v)});
    };

    for (unsigned i = 0; i < 8; i++) {
      Value elem = extract_val(f32_ty, res, getIntAttr(i));
      acc[idx[i]] = elem;
      resVals[(m * numN / 2 + n) * 8 + i] = elem;
    }
  };

  for (unsigned k = 0; k < NK; k += 4)
    for (unsigned m = 0; m < numM / 2; ++m)
      for (unsigned n = 0; n < numN / 2; ++n) {
        callMMA(m, n, k);
      }

  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(resSize, type::f32Ty(ctx)));
  Value res = getStructFromElements(loc, resVals, rewriter, structTy);
  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult
DotOpConversion::convertFMADot(triton::DotOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();
  auto threadId = getThreadId(rewriter, loc);

  using ValueTable = std::map<std::pair<int, int>, Value>;

  auto A = op.a();
  auto B = op.b();
  auto C = op.c();
  auto D = op.getResult();

  auto aTensorTy = A.getType().cast<RankedTensorType>();
  auto bTensorTy = B.getType().cast<RankedTensorType>();
  auto cTensorTy = C.getType().cast<RankedTensorType>();
  auto dTensorTy = D.getType().cast<RankedTensorType>();

  auto aShape = aTensorTy.getShape();
  auto bShape = bTensorTy.getShape();
  auto cShape = cTensorTy.getShape();

  BlockedEncodingAttr dLayout =
      dTensorTy.getEncoding().cast<BlockedEncodingAttr>();
  auto order = dLayout.getOrder();
  auto cc = getElementsFromStruct(loc, adaptor.c(), rewriter);

  DotOpFMAConversionHelper helper(dLayout);
  auto aDotOpLayout = aTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
  auto bDotOpLayout = bTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
  auto aLayout = aDotOpLayout.getParent().cast<BlockedEncodingAttr>();
  auto bLayout = bDotOpLayout.getParent().cast<BlockedEncodingAttr>();

  Value llA = adaptor.a();
  Value llB = adaptor.b();

  auto sizePerThread = getSizePerThread(dLayout);
  auto shapePerCTA = getShapePerCTA(dLayout);

  int K = aShape[1];
  int M = aShape[0];
  int N = bShape[1];

  int mShapePerCTA =
      order[0] == 1 ? shapePerCTA[order[1]] : shapePerCTA[order[0]];
  int mSizePerThread =
      order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
  int nShapePerCTA =
      order[0] == 0 ? shapePerCTA[order[1]] : shapePerCTA[order[0]];
  int nSizePerThread =
      order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];

  auto has = helper.getValueTableFromStruct(llA, K, M, mShapePerCTA,
                                            mSizePerThread, rewriter, loc);
  auto hbs = helper.getValueTableFromStruct(llB, K, N, nShapePerCTA,
                                            nSizePerThread, rewriter, loc);

  SmallVector<Value> ret = cc;
  for (unsigned k = 0; k < K; k++) {
    int z = 0;
    for (unsigned m = 0; m < M; m += mShapePerCTA)
      for (unsigned n = 0; n < N; n += nShapePerCTA)
        for (unsigned mm = 0; mm < mSizePerThread; ++mm)
          for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
            ret[z] = rewriter.create<LLVM::FMulAddOp>(loc, has[{m + mm, k}],
                                                      hbs[{n + nn, k}], ret[z]);
            ++z;
          }
  }

  auto res = getStructFromElements(
      loc, ret, rewriter,
      struct_ty(SmallVector<Type>(ret.size(), ret[0].getType())));
  rewriter.replaceOp(op, res);

  return success();
}

/// ====================== mma codegen end ============================

/// ====================== trans codegen begin ============================

struct TransOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::TransOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::TransOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto srcSmemObj =
        getSharedMemoryObjectFromStruct(loc, adaptor.src(), rewriter);
    SmallVector<Value> dstStrides = {srcSmemObj.strides[1],
                                     srcSmemObj.strides[0]};
    SmallVector<Value> dstOffsets = {srcSmemObj.offsets[1],
                                     srcSmemObj.offsets[0]};
    auto dstSmemObj =
        SharedMemoryObject(srcSmemObj.base, dstStrides, dstOffsets);
    auto retVal = getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

/// ====================== trans codegen end ============================

Value convertSplatLikeOpWithMmaLayout(const MmaEncodingAttr &layout,
                                      Type resType, Type elemType,
                                      Value constVal,
                                      TypeConverter *typeConverter,
                                      ConversionPatternRewriter &rewriter,
                                      Location loc) {
  auto tensorTy = resType.cast<RankedTensorType>();
  auto shape = tensorTy.getShape();
  if (layout.isAmpere()) {
    auto [repM, repN] = DotOpMmaV2ConversionHelper::getRepMN(tensorTy);
    size_t fcSize = 4 * repM * repN;

    auto structTy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), SmallVector<Type>(fcSize, elemType));
    return getStructFromElements(loc, SmallVector<Value>(fcSize, constVal),
                                 rewriter, structTy);
  }
  if (layout.isVolta()) {
    DotOpMmaV1ConversionHelper helper(layout);
    int repM = helper.getRepM(shape[0]);
    int repN = helper.getRepN(shape[1]);
    // According to mma layout of v1, each thread process 8 elements.
    int elems = 8 * repM * repN;

    auto structTy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), SmallVector<Type>(elems, elemType));
    return getStructFromElements(loc, SmallVector<Value>(elems, constVal),
                                 rewriter, structTy);
  }

  assert(false && "Unsupported mma layout found");
  return {};
}

class TritonGPUToLLVMTypeConverter : public LLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonGPUToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions &option,
                               const DataLayoutAnalysis *analysis = nullptr)
      : LLVMTypeConverter(ctx, option, analysis) {
    addConversion([&](triton::PointerType type) -> llvm::Optional<Type> {
      return convertTritonPointerType(type);
    });
    addConversion([&](RankedTensorType type) -> llvm::Optional<Type> {
      return convertTritonTensorType(type);
    });
    // Internally store float8 as int8
    addConversion([&](triton::Float8Type type) -> llvm::Optional<Type> {
      return IntegerType::get(type.getContext(), 8);
    });
  }

  Type convertTritonPointerType(triton::PointerType type) {
    // Recursively translate pointee type
    return LLVM::LLVMPointerType::get(convertType(type.getPointeeType()),
                                      type.getAddressSpace());
  }

  llvm::Optional<Type> convertTritonTensorType(RankedTensorType type) {
    auto ctx = type.getContext();
    Attribute layout = type.getEncoding();
    SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());

    if (layout &&
        (layout.isa<BlockedEncodingAttr>() || layout.isa<SliceEncodingAttr>() ||
         layout.isa<MmaEncodingAttr>())) {
      unsigned numElementsPerThread = getElemsPerThread(type);
      SmallVector<Type, 4> types(numElementsPerThread,
                                 convertType(type.getElementType()));
      return LLVM::LLVMStructType::getLiteral(ctx, types);
    } else if (auto shared_layout =
                   layout.dyn_cast_or_null<SharedEncodingAttr>()) {
      SmallVector<Type, 4> types;
      // base ptr
      auto ptrType =
          LLVM::LLVMPointerType::get(convertType(type.getElementType()), 3);
      types.push_back(ptrType);
      // shape dims
      auto rank = type.getRank();
      // offsets + strides
      for (auto i = 0; i < rank * 2; i++) {
        types.push_back(IntegerType::get(ctx, 32));
      }
      return LLVM::LLVMStructType::getLiteral(ctx, types);
    } else if (auto dotOpLayout =
                   layout.dyn_cast_or_null<DotOperandEncodingAttr>()) {
      if (dotOpLayout.getParent()
              .isa<BlockedEncodingAttr>()) { // for parent is blocked layout
        int numElemsPerThread =
            DotOpFMAConversionHelper::getNumElemsPerThread(shape, dotOpLayout);

        return LLVM::LLVMStructType::getLiteral(
            ctx, SmallVector<Type>(numElemsPerThread, type::f32Ty(ctx)));
      } else { // for parent is MMA layout
        auto mmaLayout = dotOpLayout.getParent().cast<MmaEncodingAttr>();
        auto wpt = mmaLayout.getWarpsPerCTA();
        Type elemTy = convertType(type.getElementType());
        if (mmaLayout.isAmpere()) {
          const llvm::DenseMap<int, Type> targetTyMap = {
              {32, elemTy},
              {16, vec_ty(elemTy, 2)},
              {8, vec_ty(elemTy, 4)},
          };
          Type targetTy;
          if (targetTyMap.count(elemTy.getIntOrFloatBitWidth())) {
            targetTy = targetTyMap.lookup(elemTy.getIntOrFloatBitWidth());
          } else {
            assert(false && "Unsupported element type");
          }
          if (dotOpLayout.getOpIdx() == 0) { // $a
            int elems =
                MMA16816ConversionHelper::getANumElemsPerThread(type, wpt[0]);
            return LLVM::LLVMStructType::getLiteral(
                ctx, SmallVector<Type>(elems, targetTy));
          }
          if (dotOpLayout.getOpIdx() == 1) { // $b
            int elems =
                MMA16816ConversionHelper::getBNumElemsPerThread(type, wpt[1]);
            return struct_ty(SmallVector<Type>(elems, targetTy));
          }
        }

        if (mmaLayout.isVolta()) {
          DotOpMmaV1ConversionHelper helper(mmaLayout);

          // TODO[Superjomn]: Both transA and transB are not available here.
          bool trans = false;
          // TODO[Superjomn]: The order of A and B are not available here.
          SmallVector<unsigned> order({1, 0});
          if (trans) {
            std::swap(shape[0], shape[1]);
            std::swap(order[0], order[1]);
          }

          if (dotOpLayout.getOpIdx() == 0) { // $a
            int elems = helper.numElemsPerThreadA(shape, order);
            Type x2Ty = vec_ty(elemTy, 2);
            return struct_ty(SmallVector<Type>(elems, x2Ty));
          }
          if (dotOpLayout.getOpIdx() == 1) { // $b
            int elems = helper.numElemsPerThreadB(shape, order);
            Type x2Ty = vec_ty(elemTy, 2);
            return struct_ty(SmallVector<Type>(elems, x2Ty));
          }
        }
      }

      llvm::errs() << "Unexpected dot operand layout detected in "
                      "TritonToLLVMTypeConverter";
      return llvm::None;
    }

    return llvm::None;
  }
};

struct AsyncWaitOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PTXBuilder ptxBuilder;
    auto &asyncWaitOp = *ptxBuilder.create<>("cp.async.wait_group");
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    asyncWaitOp(ptxBuilder.newConstantOperand(num));

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto voidTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, voidTy);

    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct InsertSliceAsyncOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::InsertSliceAsyncOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::InsertSliceAsyncOp>::ConvertTritonGPUOpToLLVMPattern;

  InsertSliceAsyncOpConversion(LLVMTypeConverter &converter,
                               const Allocation *allocation, Value smem,
                               AxisInfoAnalysis &axisAnalysisPass,
                               PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::gpu::InsertSliceAsyncOp>(
            converter, allocation, smem, benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::InsertSliceAsyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // insert_slice_async %src, %dst, %index, %mask, %other
    auto loc = op.getLoc();
    Value src = op.src();
    Value dst = op.dst();
    Value res = op.result();
    Value mask = op.mask();
    Value other = op.other();
    assert(allocation->getBufferId(res) == Allocation::InvalidBufferId &&
           "Only support in-place insert_slice_async for now");

    auto srcTy = src.getType().cast<RankedTensorType>();
    auto resTy = dst.getType().cast<RankedTensorType>();
    auto resElemTy = getTypeConverter()->convertType(resTy.getElementType());
    auto srcBlockedLayout = srcTy.getEncoding().cast<BlockedEncodingAttr>();
    auto resSharedLayout = resTy.getEncoding().cast<SharedEncodingAttr>();
    auto srcShape = srcTy.getShape();
    assert(srcShape.size() == 2 &&
           "insert_slice_async: Unexpected rank of %src");

    Value llDst = adaptor.dst();
    Value llSrc = adaptor.src();
    Value llMask = adaptor.mask();
    Value llOther = adaptor.other();
    Value llIndex = adaptor.index();

    // %src
    auto srcElems = getLLVMElems(src, llSrc, rewriter, loc);

    // %dst
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto dstShape = dstTy.getShape();
    auto smemObj = getSharedMemoryObjectFromStruct(loc, llDst, rewriter);
    auto axis = op->getAttrOfType<IntegerAttr>("axis").getInt();
    SmallVector<Value, 4> offsetVals;
    SmallVector<Value, 4> srcStrides;
    for (auto i = 0; i < dstShape.size(); ++i) {
      if (i == axis) {
        offsetVals.emplace_back(llIndex);
      } else {
        offsetVals.emplace_back(i32_val(0));
        srcStrides.emplace_back(smemObj.strides[i]);
      }
    }
    // Compute the offset based on the original dimensions of the shared
    // memory object
    auto dstOffset = dot(rewriter, loc, offsetVals, smemObj.strides);
    auto dstPtrTy =
        ptr_ty(getTypeConverter()->convertType(resTy.getElementType()), 3);
    Value dstPtrBase = gep(dstPtrTy, smemObj.base, dstOffset);

    // %mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = getLLVMElems(mask, llMask, rewriter, loc);
      assert(srcElems.size() == maskElems.size());
    }

    // %other
    SmallVector<Value> otherElems;
    if (llOther) {
      // FIXME(Keren): always assume other is 0 for now
      // It's not necessary for now because the pipeline pass will skip
      // generating insert_slice_async if the load op has any "other" tensor.
      // assert(false && "insert_slice_async: Other value not supported yet");
      otherElems = getLLVMElems(other, llOther, rewriter, loc);
      assert(srcElems.size() == otherElems.size());
    }

    unsigned inVec = getVectorSize(src);
    unsigned outVec = resSharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    unsigned numElems = getElemsPerThread(srcTy);
    unsigned perPhase = resSharedLayout.getPerPhase();
    unsigned maxPhase = resSharedLayout.getMaxPhase();
    auto sizePerThread = srcBlockedLayout.getSizePerThread();
    auto threadsPerCTA = getThreadsPerCTA(srcBlockedLayout);
    auto inOrder = srcBlockedLayout.getOrder();

    // If perPhase * maxPhase > threadsPerCTA, we will have elements
    // that share the same tile indices. The index calculation will
    // be cached.
    auto numSwizzleRows = std::max<unsigned>(
        (perPhase * maxPhase) / threadsPerCTA[inOrder[1]], 1);
    // A sharedLayout encoding has a "vec" parameter.
    // On the column dimension, if inVec > outVec, it means we have to divide
    // single vector read into multiple ones
    auto numVecCols = std::max<unsigned>(inVec / outVec, 1);

    auto srcIndices = emitIndices(loc, rewriter, srcBlockedLayout, srcShape);
    //  <<tileVecIdxRow, tileVecIdxCol>, TileOffset>
    DenseMap<std::pair<unsigned, unsigned>, Value> tileOffsetMap;
    for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
      // minVec = 2, inVec = 4, outVec = 2
      //   baseOffsetCol = 0   baseOffsetCol = 0
      //   tileVecIdxCol = 0   tileVecIdxCol = 1
      //                -/\-   -/\-
      //               [|x x| |x x| x x x x x]
      //               [|x x| |x x| x x x x x]
      // baseOffsetRow [|x x| |x x| x x x x x]
      //               [|x x| |x x| x x x x x]
      auto vecIdx = elemIdx / minVec;
      auto vecIdxCol = vecIdx % (sizePerThread[inOrder[0]] / minVec);
      auto vecIdxRow = vecIdx / (sizePerThread[inOrder[0]] / minVec);
      auto baseOffsetCol =
          vecIdxCol / numVecCols * numVecCols * threadsPerCTA[inOrder[0]];
      auto baseOffsetRow = vecIdxRow / numSwizzleRows * numSwizzleRows *
                           threadsPerCTA[inOrder[1]];
      auto tileVecIdxCol = vecIdxCol % numVecCols;
      auto tileVecIdxRow = vecIdxRow % numSwizzleRows;

      if (!tileOffsetMap.count({tileVecIdxRow, tileVecIdxCol})) {
        // Swizzling
        // Since the swizzling index is related to outVec, and we know minVec
        // already, inVec doesn't matter
        //
        // (Numbers represent row indices)
        // Example1:
        // outVec = 2, inVec = 2, minVec = 2
        // outVec = 2, inVec = 4, minVec = 2
        //     | [1 2] [3 4] [5 6] ... |
        //     | [3 4] [1 2] [7 8] ... |
        //     | [5 6] [7 8] [1 2] ... |
        // Example2:
        // outVec = 4, inVec = 2, minVec = 2
        //     | [1 2 3 4] [5 6 7 8] [9 10 11 12] ... |
        //     | [5 6 7 8] [1 2 3 4] [13 14 15 16] ... |
        //     | [9 10 11 12] [13 14 15 16] [1 2 3 4] ... |
        auto srcIdx = srcIndices[tileVecIdxRow * sizePerThread[inOrder[0]]];
        Value phase = urem(udiv(srcIdx[inOrder[1]], i32_val(perPhase)),
                           i32_val(maxPhase));
        // srcShape and smemObj.shape maybe different if smemObj is a
        // slice of the original shared memory object.
        // So we need to use the original shape to compute the offset
        Value rowOffset = mul(srcIdx[inOrder[1]], srcStrides[inOrder[1]]);
        Value colOffset =
            add(srcIdx[inOrder[0]], i32_val(tileVecIdxCol * minVec));
        Value swizzleIdx = udiv(colOffset, i32_val(outVec));
        Value swizzleColOffset =
            add(mul(xor_(swizzleIdx, phase), i32_val(outVec)),
                urem(colOffset, i32_val(outVec)));
        Value tileOffset = add(rowOffset, swizzleColOffset);
        tileOffsetMap[{tileVecIdxRow, tileVecIdxCol}] =
            gep(dstPtrTy, dstPtrBase, tileOffset);
      }

      // 16 * 8 = 128bits
      auto maxBitWidth =
          std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
      auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
      auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
      auto numWords = vecBitWidth / bitWidth;
      auto numWordElems = bitWidth / resElemTy.getIntOrFloatBitWidth();

      // Tune CG and CA here.
      auto byteWidth = bitWidth / 8;
      CacheModifier srcCacheModifier =
          byteWidth == 16 ? CacheModifier::CG : CacheModifier::CA;
      assert(byteWidth == 16 || byteWidth == 8 || byteWidth == 4);
      auto resByteWidth = resElemTy.getIntOrFloatBitWidth() / 8;

      Value tileOffset = tileOffsetMap[{tileVecIdxRow, tileVecIdxCol}];
      Value baseOffset =
          add(mul(i32_val(baseOffsetRow), srcStrides[inOrder[1]]),
              i32_val(baseOffsetCol));
      Value basePtr = gep(dstPtrTy, tileOffset, baseOffset);
      for (size_t wordIdx = 0; wordIdx < numWords; ++wordIdx) {
        PTXBuilder ptxBuilder;
        auto wordElemIdx = wordIdx * numWordElems;
        auto &copyAsyncOp =
            *ptxBuilder.create<PTXCpAsyncLoadInstr>(srcCacheModifier);
        auto *dstOperand =
            ptxBuilder.newAddrOperand(basePtr, "r", wordElemIdx * resByteWidth);
        auto *srcOperand =
            ptxBuilder.newAddrOperand(srcElems[elemIdx + wordElemIdx], "l");
        auto *copySize = ptxBuilder.newConstantOperand(byteWidth);
        auto *srcSize = copySize;
        if (op.mask()) {
          // We don't use predicate in this case, setting src-size to 0
          // if there's any mask. cp.async will automatically fill the
          // remaining slots with 0 if cp-size > src-size.
          // XXX(Keren): Always assume other = 0 for now.
          auto selectOp = select(maskElems[elemIdx + wordElemIdx],
                                 i32_val(byteWidth), i32_val(0));
          srcSize = ptxBuilder.newOperand(selectOp, "r");
        }
        copyAsyncOp(dstOperand, srcOperand, copySize, srcSize);
        ptxBuilder.launch(rewriter, loc, void_ty(getContext()));
      }
    }

    PTXBuilder ptxBuilder;
    ptxBuilder.create<>("cp.async.commit_group")->operator()();
    ptxBuilder.launch(rewriter, loc, void_ty(getContext()));
    rewriter.replaceOp(op, llDst);
    return success();
  }
};

struct ExtElemwiseOpConversion
    : public ElementwiseOpConversionBase<triton::ExtElemwiseOp,
                                         ExtElemwiseOpConversion> {
  using Base = ElementwiseOpConversionBase<triton::ExtElemwiseOp,
                                           ExtElemwiseOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(triton::ExtElemwiseOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    StringRef funcName = op.symbol();
    if (funcName.empty())
      llvm::errs() << "ExtElemwiseOpConversion";

    Type funcType = getFunctionType(elemTy, operands);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetFuncOp(rewriter, op, funcName, funcType);
    return rewriter.create<LLVM::CallOp>(loc, funcOp, operands).getResult(0);
  }

private:
  Type getFunctionType(Type resultType, ValueRange operands) const {
    SmallVector<Type> operandTypes(operands.getTypes());
    return LLVM::LLVMFunctionType::get(resultType, operandTypes);
  }

  LLVM::LLVMFuncOp appendOrGetFuncOp(ConversionPatternRewriter &rewriter,
                                     triton::ExtElemwiseOp op,
                                     StringRef funcName, Type funcType) const {
    using LLVM::LLVMFuncOp;

    auto funcAttr = StringAttr::get(op->getContext(), funcName);
    Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
    if (funcOp)
      return cast<LLVMFuncOp>(*funcOp);

    mlir::OpBuilder b(op->getParentOfType<LLVMFuncOp>());
    auto ret = b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
    ret.getOperation()->setAttr(
        "libname", StringAttr::get(op->getContext(), op.libname()));
    ret.getOperation()->setAttr(
        "libpath", StringAttr::get(op->getContext(), op.libpath()));
    return ret;
  }
};

struct FDivOpConversion
    : ElementwiseOpConversionBase<mlir::arith::DivFOp, FDivOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::DivFOp, FDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::DivFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {

    PTXBuilder ptxBuilder;
    auto &fdiv = *ptxBuilder.create<PTXInstr>("div");
    unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
    if (32 == bitwidth) {
      fdiv.o("full").o("f32");
    } else if (64 == bitwidth) {
      fdiv.o("rn").o("f64");
    } else {
      assert(0 && bitwidth && "not supported");
    }

    auto res = ptxBuilder.newOperand(bitwidth == 32 ? "=r" : "=l");
    auto lhs = ptxBuilder.newOperand(operands[0], bitwidth == 32 ? "r" : "l");
    auto rhs = ptxBuilder.newOperand(operands[1], bitwidth == 32 ? "r" : "l");
    fdiv(res, lhs, rhs);

    Value ret = ptxBuilder.launch(rewriter, loc, elemTy, false);
    return ret;
  }
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<mlir::math::ExpOp, ExpOpConversionApprox> {
  using Base =
      ElementwiseOpConversionBase<mlir::math::ExpOp, ExpOpConversionApprox>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::math::ExpOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    // For FP64 input, call __nv_expf for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() == 64)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = fmul(f32_ty, operands[0], f32_val(log2e));

    PTXBuilder ptxBuilder;
    auto &exp2 = ptxBuilder.create<PTXInstr>("ex2")->o("approx").o("f32");
    auto output = ptxBuilder.newOperand("=f");
    auto input = ptxBuilder.newOperand(prod, "f");
    exp2(output, input);
    return ptxBuilder.launch(rewriter, loc, f32_ty, false);
  }
};
/// ====================== atomic_cas codegen begin ==========================
struct AtomicCASOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AtomicCASOp>::ConvertTritonGPUOpToLLVMPattern;

  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const Allocation *allocation, Value smem,
                        AxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicCASOp>(
            converter, allocation, smem, benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Value ptr = op.ptr();

    Value llPtr = adaptor.ptr();
    Value llCmp = adaptor.cmp();
    Value llVal = adaptor.val();

    auto ptrElements = getElementsFromStruct(loc, llPtr, rewriter);
    auto cmpElements = getElementsFromStruct(loc, llCmp, rewriter);
    auto valElements = getElementsFromStruct(loc, llVal, rewriter);

    auto valueTy = op.getResult().getType().dyn_cast<RankedTensorType>();
    Type valueElemTy =
        valueTy ? getTypeConverter()->convertType(valueTy.getElementType())
                : op.getResult().getType();
    auto tid = tid_val();
    Value pred = icmp_eq(tid, i32_val(0));
    PTXBuilder ptxBuilderMemfence;
    auto memfenc = ptxBuilderMemfence.create<PTXInstr>("membar")->o("gl");
    memfenc();
    auto ASMReturnTy = void_ty(ctx);
    ptxBuilderMemfence.launch(rewriter, loc, ASMReturnTy);

    Value atomPtr = getSharedMemoryBase(loc, rewriter, op.getOperation());
    atomPtr = bitcast(atomPtr, ptr_ty(valueElemTy, 3));

    Value casPtr = ptrElements[0];
    Value casCmp = cmpElements[0];
    Value casVal = valElements[0];

    PTXBuilder ptxBuilderAtomicCAS;
    auto *dstOpr = ptxBuilderAtomicCAS.newOperand("=r");
    auto *ptrOpr = ptxBuilderAtomicCAS.newAddrOperand(casPtr, "l");
    auto *cmpOpr = ptxBuilderAtomicCAS.newOperand(casCmp, "r");
    auto *valOpr = ptxBuilderAtomicCAS.newOperand(casVal, "r");
    auto &atom = *ptxBuilderAtomicCAS.create<PTXInstr>("atom");
    atom.global().o("cas").o("b32");
    atom(dstOpr, ptrOpr, cmpOpr, valOpr).predicate(pred);
    auto old = ptxBuilderAtomicCAS.launch(rewriter, loc, valueElemTy);
    barrier();

    PTXBuilder ptxBuilderStore;
    auto *dstOprStore = ptxBuilderStore.newAddrOperand(atomPtr, "l");
    auto *valOprStore = ptxBuilderStore.newOperand(old, "r");
    auto &st = *ptxBuilderStore.create<PTXInstr>("st");
    st.shared().o("b32");
    st(dstOprStore, valOprStore).predicate(pred);
    ptxBuilderStore.launch(rewriter, loc, ASMReturnTy);
    ptxBuilderMemfence.launch(rewriter, loc, ASMReturnTy);
    barrier();
    Value ret = load(atomPtr);
    barrier();
    rewriter.replaceOp(op, {ret});
    return success();
  }
};
/// ====================== atomic_cas codegen end ==========================

/// ====================== atomic_rmw codegen begin ==========================
struct AtomicRMWOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AtomicRMWOp>::ConvertTritonGPUOpToLLVMPattern;

  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const Allocation *allocation, Value smem,
                        AxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicRMWOp>(
            converter, allocation, smem, benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto atomicRmwAttr = op.atomic_rmw_op();
    Value ptr = op.ptr();
    Value val = op.val();

    Value llPtr = adaptor.ptr();
    Value llVal = adaptor.val();
    Value llMask = adaptor.mask();

    auto valElements = getElementsFromStruct(loc, llVal, rewriter);
    auto ptrElements = getElementsFromStruct(loc, llPtr, rewriter);
    auto maskElements = getElementsFromStruct(loc, llMask, rewriter);

    auto valueTy = op.getResult().getType().dyn_cast<RankedTensorType>();
    Type valueElemTy =
        valueTy ? getTypeConverter()->convertType(valueTy.getElementType())
                : op.getResult().getType();
    const size_t valueElemNbits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getElemsPerThread(val.getType());
    // vec = 1 for scalar
    auto vec = getVectorSize(ptr);
    Value mask = int_val(1, 1);
    auto tid = tid_val();
    // tensor
    if (valueTy) {
      auto valTy = val.getType().cast<RankedTensorType>();
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
      // mask
      auto shape = valueTy.getShape();
      auto numElements = product(shape);
      mask = and_(mask, icmp_slt(mul(tid, i32_val(elemsPerThread)),
                                 i32_val(numElements)));
    }

    auto vecTy = vec_ty(valueElemTy, vec);
    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value rmwVal = undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        rmwVal = insert_element(vecTy, rmwVal, valElements[i + ii], iiVal);
      }

      Value rmwPtr = ptrElements[i];
      Value rmwMask = maskElements[i];
      rmwMask = and_(rmwMask, mask);
      std::string sTy;
      PTXBuilder ptxBuilderAtomicRMW;
      std::string tyId = valueElemNbits * vec == 64
                             ? "l"
                             : (valueElemNbits * vec == 32 ? "r" : "h");
      auto *dstOpr = ptxBuilderAtomicRMW.newOperand("=" + tyId);
      auto *ptrOpr = ptxBuilderAtomicRMW.newAddrOperand(rmwPtr, "l");
      auto *valOpr = ptxBuilderAtomicRMW.newOperand(rmwVal, tyId);

      auto &atom = ptxBuilderAtomicRMW.create<>("atom")->global().o("gpu");
      auto rmwOp = stringifyRMWOp(atomicRmwAttr).str();
      auto sBits = std::to_string(valueElemNbits);
      switch (atomicRmwAttr) {
      case RMWOp::AND:
        sTy = "b" + sBits;
        break;
      case RMWOp::OR:
        sTy = "b" + sBits;
        break;
      case RMWOp::XOR:
        sTy = "b" + sBits;
        break;
      case RMWOp::ADD:
        sTy = "s" + sBits;
        break;
      case RMWOp::FADD:
        rmwOp = "add";
        rmwOp += (valueElemNbits == 16 ? ".noftz" : "");
        sTy = "f" + sBits;
        sTy += (vec == 2 && valueElemNbits == 16) ? "x2" : "";
        break;
      case RMWOp::MAX:
        sTy = "s" + sBits;
        break;
      case RMWOp::MIN:
        sTy = "s" + sBits;
        break;
      case RMWOp::UMAX:
        rmwOp = "max";
        sTy = "u" + sBits;
        break;
      case RMWOp::UMIN:
        rmwOp = "min";
        sTy = "u" + sBits;
        break;
      case RMWOp::XCHG:
        sTy = "b" + sBits;
        break;
      default:
        return failure();
      }
      atom.o(rmwOp).o(sTy);
      if (valueTy) {
        atom(dstOpr, ptrOpr, valOpr).predicate(rmwMask);
        auto retType = vec == 1 ? valueElemTy : vecTy;
        auto ret = ptxBuilderAtomicRMW.launch(rewriter, loc, retType);
        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? ret : extract_element(valueElemTy, ret, idx_val(ii));
        }
      } else {
        PTXBuilder ptxBuilderMemfence;
        auto memfenc = ptxBuilderMemfence.create<PTXInstr>("membar")->o("gl");
        memfenc();
        auto ASMReturnTy = void_ty(ctx);
        ptxBuilderMemfence.launch(rewriter, loc, ASMReturnTy);
        rmwMask = and_(rmwMask, icmp_eq(tid, i32_val(0)));
        atom(dstOpr, ptrOpr, valOpr).predicate(rmwMask);
        auto old = ptxBuilderAtomicRMW.launch(rewriter, loc, valueElemTy);
        Value atomPtr = getSharedMemoryBase(loc, rewriter, op.getOperation());
        atomPtr = bitcast(atomPtr, ptr_ty(valueElemTy, 3));
        store(old, atomPtr);
        barrier();
        Value ret = load(atomPtr);
        barrier();
        rewriter.replaceOp(op, {ret});
      }
    }
    if (valueTy) {
      Type structTy = getTypeConverter()->convertType(valueTy);
      Value resultStruct =
          getStructFromElements(loc, resultVals, rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
};
/// ====================== atomic_rmw codegen end ==========================

void populateTritonToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns, int numWarps,
                                  AxisInfoAnalysis &axisInfoAnalysis,
                                  const Allocation *allocation, Value smem,
                                  PatternBenefit benefit = 1) {
  patterns.add<AddPtrOpConversion>(typeConverter, benefit);
  patterns.add<AllocTensorOpConversion>(typeConverter, allocation, smem,
                                        benefit);
  patterns.add<ArithConstantSplatOpConversion>(typeConverter, benefit);
  patterns.add<AsyncWaitOpConversion>(typeConverter, benefit);

#define POPULATE_TERNARY_OP(SRC_OP, DST_OP)                                    \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(typeConverter, benefit);
  POPULATE_TERNARY_OP(triton::gpu::SelectOp, LLVM::SelectOp);
#undef POPULATE_TERNARY_OP

#define POPULATE_BINARY_OP(SRC_OP, DST_OP)                                     \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(typeConverter, benefit);

  POPULATE_BINARY_OP(arith::SubIOp, LLVM::SubOp) // -
  POPULATE_BINARY_OP(arith::SubFOp, LLVM::FSubOp)
  POPULATE_BINARY_OP(arith::AddIOp, LLVM::AddOp) // +
  POPULATE_BINARY_OP(arith::AddFOp, LLVM::FAddOp)
  POPULATE_BINARY_OP(arith::MulIOp, LLVM::MulOp) // *
  POPULATE_BINARY_OP(arith::MulFOp, LLVM::FMulOp)
  POPULATE_BINARY_OP(arith::DivFOp, LLVM::FDivOp) // /
  POPULATE_BINARY_OP(arith::DivSIOp, LLVM::SDivOp)
  POPULATE_BINARY_OP(arith::DivUIOp, LLVM::UDivOp)
  POPULATE_BINARY_OP(arith::RemFOp, LLVM::FRemOp) // %
  POPULATE_BINARY_OP(arith::RemSIOp, LLVM::SRemOp)
  POPULATE_BINARY_OP(arith::RemUIOp, LLVM::URemOp)
  POPULATE_BINARY_OP(arith::AndIOp, LLVM::AndOp)   // &
  POPULATE_BINARY_OP(arith::OrIOp, LLVM::OrOp)     // |
  POPULATE_BINARY_OP(arith::XOrIOp, LLVM::XOrOp)   // ^
  POPULATE_BINARY_OP(arith::ShLIOp, LLVM::ShlOp)   // <<
  POPULATE_BINARY_OP(arith::ShRSIOp, LLVM::AShrOp) // >>
  POPULATE_BINARY_OP(arith::ShRUIOp, LLVM::LShrOp) // >>
#undef POPULATE_BINARY_OP

  patterns.add<CmpIOpConversion>(typeConverter, benefit);
  patterns.add<CmpFOpConversion>(typeConverter, benefit);

  // ExpOpConversionApprox will try using ex2.approx if the input type is FP32.
  // For FP64 input type, ExpOpConversionApprox will return failure and
  // ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will call
  // __nv_expf for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, benefit);

#define POPULATE_UNARY_OP(SRC_OP, DST_OP)                                      \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(typeConverter, benefit);

  POPULATE_UNARY_OP(arith::TruncIOp, LLVM::TruncOp)
  POPULATE_UNARY_OP(arith::TruncFOp, LLVM::FPTruncOp)
  POPULATE_UNARY_OP(arith::ExtSIOp, LLVM::SExtOp)
  POPULATE_UNARY_OP(arith::ExtUIOp, LLVM::ZExtOp)
  POPULATE_UNARY_OP(arith::FPToUIOp, LLVM::FPToUIOp)
  POPULATE_UNARY_OP(arith::FPToSIOp, LLVM::FPToSIOp)
  POPULATE_UNARY_OP(arith::UIToFPOp, LLVM::UIToFPOp)
  POPULATE_UNARY_OP(arith::SIToFPOp, LLVM::SIToFPOp)
  POPULATE_UNARY_OP(arith::ExtFOp, LLVM::FPExtOp)
  POPULATE_UNARY_OP(math::LogOp, math::LogOp)
  POPULATE_UNARY_OP(math::CosOp, math::CosOp)
  POPULATE_UNARY_OP(math::SinOp, math::SinOp)
  POPULATE_UNARY_OP(math::SqrtOp, math::SqrtOp)
  POPULATE_UNARY_OP(math::ExpOp, math::ExpOp)
  POPULATE_UNARY_OP(triton::BitcastOp, LLVM::BitcastOp)
  POPULATE_UNARY_OP(triton::IntToPtrOp, LLVM::IntToPtrOp)
  POPULATE_UNARY_OP(triton::PtrToIntOp, LLVM::PtrToIntOp)
#undef POPULATE_UNARY_OP

  patterns.add<FpToFpOpConversion>(typeConverter, benefit);

  patterns.add<FDivOpConversion>(typeConverter, benefit);

  patterns.add<ExtElemwiseOpConversion>(typeConverter, benefit);

  patterns.add<BroadcastOpConversion>(typeConverter, benefit);
  patterns.add<ReduceOpConversion>(typeConverter, allocation, smem, benefit);
  patterns.add<ConvertLayoutOpConversion>(typeConverter, allocation, smem,
                                          benefit);
  patterns.add<AtomicCASOpConversion>(typeConverter, allocation, smem,
                                      axisInfoAnalysis, benefit);
  patterns.add<AtomicRMWOpConversion>(typeConverter, allocation, smem,
                                      axisInfoAnalysis, benefit);
  patterns.add<ExtractSliceOpConversion>(typeConverter, allocation, smem,
                                         benefit);
  patterns.add<GetProgramIdOpConversion>(typeConverter, benefit);
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<InsertSliceOpConversion>(typeConverter, allocation, smem,
                                        benefit);
  patterns.add<InsertSliceAsyncOpConversion>(typeConverter, allocation, smem,
                                             axisInfoAnalysis, benefit);
  patterns.add<LoadOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<MakeRangeOpConversion>(typeConverter, benefit);
  patterns.add<ReturnOpConversion>(typeConverter, benefit);
  patterns.add<SplatOpConversion>(typeConverter, benefit);
  patterns.add<StoreOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ViewLikeOpConversion<triton::ViewOp>>(typeConverter, benefit);
  patterns.add<ViewLikeOpConversion<triton::ExpandDimsOp>>(typeConverter,
                                                           benefit);
  patterns.add<DotOpConversion>(typeConverter, allocation, smem, benefit);
  patterns.add<TransOpConversion>(typeConverter, benefit);
  patterns.add<CatOpConversion>(typeConverter, benefit);
  patterns.add<PrintfOpConversion>(typeConverter, benefit);
}

class ConvertTritonGPUToLLVM
    : public ConvertTritonGPUToLLVMBase<ConvertTritonGPUToLLVM> {

private:
  void decomposeMmaToDotOperand(ModuleOp mod, int numWarps) {
    // replace `mma -> dot_op` with `mma -> blocked -> dot_op`
    // unless certain conditions are met
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcMma =
          srcType.getEncoding().dyn_cast<triton::gpu::MmaEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcMma && dstDotOp &&
          !ConvertLayoutOpConversion::isMmaToDotShortcut(srcMma, dstDotOp)) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), srcType.getShape(), getSizePerThread(srcMma),
                getOrder(srcMma), numWarps));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  void decomposeBlockedToDotOperand(ModuleOp mod) {
    // replace `blocked -> dot_op` with `blocked -> shared -> dot_op`
    // because the codegen doesn't handle `blocked -> dot_op` directly
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcBlocked && dstDotOp) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::SharedEncodingAttr::get(
                mod.getContext(), dstDotOp, srcType.getShape(),
                getOrder(srcBlocked), srcType.getElementType()));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  void decomposeInsertSliceAsyncOp(ModuleOp mod) {
    AxisInfoAnalysis axisInfoAnalysis(mod.getContext());
    axisInfoAnalysis.run(mod);
    // TODO(Keren): This is a hacky knob that may cause performance regression
    // when decomposition has been performed. We should remove this knob once we
    // have thorough analysis on async wait. Currently, we decompose
    // `insert_slice_async` into `load` and `insert_slice` without knowing which
    // `async_wait` is responsible for the `insert_slice_async`. To guarantee
    // correctness, we blindly set the `async_wait` to wait for all async ops.
    //
    // There are two options to improve this:
    // 1. We can perform a dataflow analysis to find the `async_wait` that is
    // responsible for the `insert_slice_async` in the backend.
    // 2. We can modify the pipeline to perform the decomposition before the
    // `async_wait` is inserted. However, it is also risky because we don't know
    // the correct vectorized shape yet in the pipeline pass. Making the
    // pipeline pass aware of the vectorization could introduce additional
    // dependencies on the AxisInfoAnalysis and the Coalesce analysis.
    bool decomposed = false;
    // insert_slice_async %src, %dst, %idx, %mask, %other
    // =>
    // %tmp = load %src, %mask, %other
    // %res = insert_slice %tmp into %dst[%idx]
    mod.walk([&](triton::gpu::InsertSliceAsyncOp insertSliceAsyncOp) -> void {
      OpBuilder builder(insertSliceAsyncOp);

      // Get the vectorized load size
      auto src = insertSliceAsyncOp.src();
      auto dst = insertSliceAsyncOp.dst();
      auto srcTy = src.getType().cast<RankedTensorType>();
      auto dstTy = dst.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcTy.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto resSharedLayout =
          dstTy.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
      auto resElemTy = dstTy.getElementType();
      unsigned inVec = axisInfoAnalysis.getPtrVectorSize(src);
      unsigned outVec = resSharedLayout.getVec();
      unsigned minVec = std::min(outVec, inVec);
      auto maxBitWidth =
          std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
      auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
      auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
      auto byteWidth = bitWidth / 8;

      // If the load byte width is not eligible or the current compute
      // capability does not support async copy, then we do decompose
      if (triton::gpu::InsertSliceAsyncOp::getEligibleLoadByteWidth(
              computeCapability)
              .contains(byteWidth))
        return;

      // load
      auto tmpTy =
          RankedTensorType::get(srcTy.getShape(), resElemTy, srcBlocked);
      auto loadOp = builder.create<triton::LoadOp>(
          insertSliceAsyncOp.getLoc(), tmpTy, insertSliceAsyncOp.src(),
          insertSliceAsyncOp.mask(), insertSliceAsyncOp.other(),
          insertSliceAsyncOp.cache(), insertSliceAsyncOp.evict(),
          insertSliceAsyncOp.isVolatile());

      // insert_slice
      auto axis = insertSliceAsyncOp.axis();
      auto intAttr = [&](int64_t v) { return builder.getI64IntegerAttr(v); };
      auto offsets = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(0));
      auto sizes = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(1));
      auto strides = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(1));
      offsets[axis] = insertSliceAsyncOp.index();
      for (size_t i = 0; i < dstTy.getRank(); i++) {
        if (i != axis)
          sizes[i] = intAttr(dstTy.getShape()[i]);
      }
      auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
          insertSliceAsyncOp.getLoc(), loadOp, insertSliceAsyncOp.dst(),
          offsets, sizes, strides);
      // Replace
      insertSliceAsyncOp.replaceAllUsesWith(insertSliceOp.getResult());
      insertSliceAsyncOp.erase();
      decomposed = true;
    });

    mod.walk([&](triton::gpu::AsyncWaitOp asyncWaitOp) -> void {
      if (!triton::gpu::AsyncWaitOp::isSupported(computeCapability)) {
        // async wait is supported in Ampere and later
        asyncWaitOp.erase();
      } else if (decomposed) {
        // Wait for all previous async ops
        OpBuilder builder(asyncWaitOp);
        auto newAsyncWaitOp =
            builder.create<triton::gpu::AsyncWaitOp>(asyncWaitOp.getLoc(), 0);
        asyncWaitOp.erase();
      }
    });
  }

public:
  explicit ConvertTritonGPUToLLVM(int computeCapability)
      : computeCapability(computeCapability) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    // TODO: need confirm
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMFunctionConversionTarget funcTarget(*context, typeConverter);
    TritonLLVMConversionTarget target(*context, typeConverter);

    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    // step 1: Decompose unoptimized layout conversions to use shared memory
    // step 2: Decompose insert_slice_async to use load + insert_slice for
    // pre-Ampere architectures or unsupported vectorized load sizes
    // step 3: Allocate shared memories and insert barriers
    // step 4: Convert SCF to CFG
    // step 5: Convert FuncOp to LLVMFuncOp via partial conversion
    // step 6: Convert the rest of ops via partial
    // conversion The reason for putting step 1 before step 2 is that the membar
    // analysis currently only supports SCF but not CFG. The reason for a
    // separation between 1/4 is that, step 3 is out of the scope of Dialect
    // Conversion, thus we need to make sure the smem is not revised during the
    // conversion of step 4.
    decomposeMmaToDotOperand(mod, numWarps);

    decomposeBlockedToDotOperand(mod);

    decomposeInsertSliceAsyncOp(mod);

    Allocation allocation(mod);
    MembarAnalysis membarPass(&allocation);
    membarPass.run();

    RewritePatternSet scf_patterns(context);
    mlir::populateLoopToStdConversionPatterns(scf_patterns);
    mlir::ConversionTarget scf_target(*context);
    scf_target.addIllegalOp<scf::ForOp, scf::IfOp, scf::ParallelOp,
                            scf::WhileOp, scf::ExecuteRegionOp>();
    scf_target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(
            applyPartialConversion(mod, scf_target, std::move(scf_patterns))))
      return signalPassFailure();

    RewritePatternSet func_patterns(context);
    func_patterns.add<FuncOpConversion>(typeConverter, numWarps, 1 /*benefit*/);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(func_patterns))))
      return signalPassFailure();

    auto axisAnalysis = runAxisAnalysis(mod);
    initSharedMemory(allocation.getSharedMemorySize(), typeConverter);
    mod->setAttr("triton_gpu.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32),
                                        allocation.getSharedMemorySize()));

    // We set a higher benefit here to ensure triton's patterns runs before
    // arith patterns for some encoding not supported by the community
    // patterns.
    RewritePatternSet patterns(context);
    populateTritonToLLVMPatterns(typeConverter, patterns, numWarps,
                                 *axisAnalysis, &allocation, smem,
                                 10 /*benefit*/);

    // Add arith/math's patterns to help convert scalar expression to LLVM.
    mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter,
                                                            patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);

    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }

protected:
  std::unique_ptr<AxisInfoAnalysis> runAxisAnalysis(ModuleOp module) {
    auto axisAnalysisPass =
        std::make_unique<AxisInfoAnalysis>(module->getContext());
    axisAnalysisPass->run(module);

    return axisAnalysisPass;
  }

  void initSharedMemory(size_t size,
                        TritonGPUToLLVMTypeConverter &typeConverter);

  Value smem;

  int computeCapability{};
};

void ConvertTritonGPUToLLVM::initSharedMemory(
    size_t size, TritonGPUToLLVMTypeConverter &typeConverter) {
  ModuleOp mod = getOperation();
  OpBuilder b(mod.getBodyRegion());
  auto loc = mod.getLoc();
  auto elemTy = typeConverter.convertType(b.getIntegerType(8));
  // Set array size 0 and external linkage indicates that we use dynamic
  // shared allocation to allow a larger shared memory size for each kernel.
  auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
  auto global = b.create<LLVM::GlobalOp>(
      loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
      "global_smem", /*value=*/Attribute(),
      /*alignment=*/0, mlir::gpu::GPUDialect::getWorkgroupAddressSpace());
  SmallVector<LLVM::LLVMFuncOp> funcs;
  mod.walk([&](LLVM::LLVMFuncOp func) { funcs.push_back(func); });
  assert(funcs.size() == 1 &&
         "Inliner pass is expected before TritonGPUToLLVM");
  b.setInsertionPointToStart(&funcs[0].getBody().front());
  smem = b.create<LLVM::AddressOfOp>(loc, global);
  auto ptrTy =
      LLVM::LLVMPointerType::get(typeConverter.convertType(b.getI8Type()), 3);
  smem = b.create<LLVM::BitcastOp>(loc, ptrTy, smem);
}

} // namespace

namespace mlir {

namespace LLVM {

void vprintf(StringRef msg, ValueRange args,
             ConversionPatternRewriter &rewriter) {
  PrintfOpConversion::llPrintf(msg, args, rewriter);
}

void vprintf_array(Value thread, ArrayRef<Value> arr, std::string info,
                   std::string elem_repr, ConversionPatternRewriter &builder) {
  std::string fmt = info + " t-%d ";
  std::vector<Value> new_arr({thread});
  for (int i = 0; i < arr.size(); ++i) {
    fmt += elem_repr + ((i == arr.size() - 1) ? "" : ", ");
    new_arr.push_back(arr[i]);
  }

  vprintf(fmt, new_arr, builder);
}

} // namespace LLVM

TritonLLVMConversionTarget::TritonLLVMConversionTarget(
    MLIRContext &ctx, mlir::LLVMTypeConverter &typeConverter)
    : ConversionTarget(ctx) {
  addLegalDialect<LLVM::LLVMDialect>();
  addLegalDialect<NVVM::NVVMDialect>();
  // addIllegalDialect<triton::TritonDialect>();
  // addIllegalDialect<triton::gpu::TritonGPUDialect>();
  addIllegalDialect<mlir::gpu::GPUDialect>();
  addIllegalDialect<mlir::StandardOpsDialect>();
  addLegalOp<mlir::UnrealizedConversionCastOp>();
}

TritonLLVMFunctionConversionTarget::TritonLLVMFunctionConversionTarget(
    MLIRContext &ctx, mlir::LLVMTypeConverter &typeConverter)
    : ConversionTarget(ctx) {
  addLegalDialect<LLVM::LLVMDialect>();
  // addLegalDialect<NVVM::NVVMDialect>();
  addIllegalOp<mlir::FuncOp>();
  addLegalOp<mlir::UnrealizedConversionCastOp>();
}

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int computeCapability) {
  return std::make_unique<::ConvertTritonGPUToLLVM>(computeCapability);
}

} // namespace triton
} // namespace mlir
