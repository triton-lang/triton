#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVM.h"
#include "../PassDetail.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
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

namespace {

// Create a 32-bit integer constant.
Value createConstantI32(Location loc, PatternRewriter &rewriter, int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<LLVM::ConstantOp>(loc, i32ty,
                                           IntegerAttr::get(i32ty, v));
}

// Create a index type constant.
Value createIndexConstant(OpBuilder &builder, Location loc,

                          TypeConverter *converter, int64_t value) {
  Type ty = converter->convertType(builder.getIndexType());
  return builder.create<LLVM::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));
}

// Create an integer constant of \param width bits.
Value createLLVMIntegerConstant(OpBuilder &builder, Location loc, short width,
                                int64_t value) {
  Type ty = builder.getIntegerType(width);
  return builder.create<LLVM::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));
}

} // namespace

// Shortcuts for some commonly used LLVM ops to keep code simple and intuitive
#define udiv(...) rewriter.create<LLVM::UDivOp>(loc, __VA_ARGS__)
#define urem(...) rewriter.create<LLVM::URemOp>(loc, __VA_ARGS__)
#define add(...) rewriter.create<LLVM::AddOp>(loc, __VA_ARGS__)
#define mul(...) rewriter.create<LLVM::MulOp>(loc, __VA_ARGS__)
#define xor_(...) rewriter.create<LLVM::XOrOp>(loc, __VA_ARGS__)
#define bitcast(...) rewriter.create<LLVM::BitcastOp>(loc, __VA_ARGS__)
#define gep(...) rewriter.create<LLVM::GEPOp>(loc, __VA_ARGS__)
#define ptr_ty(...) LLVM::LLVMPointerType::get(__VA_ARGS__)
#define insert_val(...) rewriter.create<LLVM::InsertValueOp>(loc, __VA_ARGS__)
#define extract_val(...) rewriter.create<LLVM::ExtractValueOp>(loc, __VA_ARGS__)
#define insert_element(...)                                                    \
  rewriter.create<LLVM::InsertElementOp>(loc, __VA_ARGS__)
#define extract_element(...)                                                   \
  rewriter.create<LLVM::ExtractElementOp>(loc, __VA_ARGS__)
#define load(...) rewriter.create<LLVM::LoadOp>(loc, __VA_ARGS__)
#define store(val, ptr) rewriter.create<LLVM::StoreOp>(loc, val, ptr)
#define select(...) rewriter.create<LLVM::SelectOp>(loc, __VA_ARGS__)
#define address_of(...) rewriter.create<LLVM::AddressOfOp>(loc, __VA_ARGS__)
#define barrier rewriter.create<mlir::gpu::BarrierOp>(loc)
#define undef(...) rewriter.create<LLVM::UndefOp>(loc, __VA_ARGS__)
#define i32_ty rewriter.getIntegerType(32)
#define vec_ty(type, num) VectorType::get(num, type)
#define void_ty LLVM::LLVMVoidType::get(ctx)
#define struct_ty(...) LLVM::LLVMStructType::getLiteral(__VA_ARGS__)

// Creator for constant
#define i32_val(...) LLVM::createConstantI32(loc, rewriter, __VA_ARGS__)
#define int_val(width, val)                                                    \
  LLVM::createLLVMIntegerConstant(rewriter, loc, width, val)
#define idx_val(...)                                                           \
  LLVM::createIndexConstant(rewriter, loc, this->getTypeConverter(),           \
                            __VA_ARGS__)

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
static void filterFuncAttributes(ArrayRef<NamedAttribute> attrs,
                                 bool filterArgAttrs,
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
static auto wrapAsStructAttrs(OpBuilder &b, ArrayAttr attrs) {
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
static constexpr StringRef kEmitIfaceAttrName = "llvm.emit_c_interface";
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
    Location loc = op->getLoc();
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

Value getStructFromElements(Location loc, ValueRange resultVals,
                            ConversionPatternRewriter &rewriter,
                            Type structType) {
  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structType);
  for (auto v : llvm::enumerate(resultVals)) {
    llvmStruct = insert_val(structType, llvmStruct, v.value(),
                            rewriter.getI64ArrayAttr(v.index()));
  }
  return llvmStruct;
}

template <typename T>
static SmallVector<T> getMultiDimIndex(T linearIndex, ArrayRef<T> shape) {
  // shape: {a, b, c, d}  ->  accMul: {b*c*d, c*d, d, 1}
  size_t rank = shape.size();
  T accMul = product(shape.drop_front());
  T linearRemain = linearIndex;
  SmallVector<T> multiDimIndex(rank);
  for (size_t i = 0; i < rank; ++i) {
    multiDimIndex[i] = linearRemain / accMul;
    linearRemain = linearRemain % accMul;
    if (i != (rank - 1)) {
      accMul = accMul / shape[i + 1];
    }
  }
  return multiDimIndex;
}

template <typename T>
static T getLinearIndex(ArrayRef<T> multiDimIndex, ArrayRef<T> shape) {
  assert(multiDimIndex.size() == shape.size());
  // shape: {a, b, c, d}  ->  accMul: {b*c*d, c*d, d, 1}
  size_t rank = shape.size();
  T accMul = product(shape.drop_front());
  T linearIndex = 0;
  for (size_t i = 0; i < rank; ++i) {
    linearIndex += multiDimIndex[i] * accMul;
    if (i != (rank - 1)) {
      accMul = accMul / shape[i + 1];
    }
  }
  return linearIndex;
}

struct ConvertTritonGPUOpToLLVMPatternBase {
  static SmallVector<Value>
  getElementsFromStruct(Location loc, Value llvmStruct,
                        ConversionPatternRewriter &rewriter) {
    ArrayRef<Type> types =
        llvmStruct.getType().cast<LLVM::LLVMStructType>().getBody();
    SmallVector<Value> results(types.size());
    for (unsigned i = 0; i < types.size(); ++i) {
      Type type = types[i];
      results[i] = extract_val(type, llvmStruct, rewriter.getI64ArrayAttr(i));
    }
    return results;
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
      for (auto &&en : llvm::enumerate(llvm::reverse(shape.drop_front()))) {
        Value dimSize = idx_val(en.value());
        multiDim[rank - 1 - en.index()] = urem(remained, dimSize);
        remained = udiv(remained, dimSize);
      }
      multiDim[0] = remained;
    }
    return multiDim;
  }

  Value linearize(ConversionPatternRewriter &rewriter, Location loc,
                  ArrayRef<Value> multiDim, ArrayRef<unsigned> shape) const {
    int rank = multiDim.size();
    Value linear = idx_val(0);
    if (rank > 0) {
      linear = multiDim.front();
      for (auto [dim, shape] :
           llvm::zip(multiDim.drop_front(), shape.drop_front())) {
        Value dimSize = idx_val(shape);
        linear = add(mul(linear, dimSize), dim);
      }
    }
    return linear;
  }

  // Get an index-base for each dimension for a \param blocked_layout.
  SmallVector<Value>
  emitBaseIndexForBlockedLayout(Location loc,
                                ConversionPatternRewriter &rewriter,
                                const BlockedEncodingAttr &blocked_layout,
                                ArrayRef<int64_t> shape) const {
    auto llvmIndexTy = this->getTypeConverter()->getIndexType();
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

  SmallVector<SmallVector<Value>> emitIndices(Location loc,
                                              ConversionPatternRewriter &b,
                                              const Attribute &layout,
                                              ArrayRef<int64_t> shape) const {
    if (auto blocked = layout.dyn_cast<BlockedEncodingAttr>()) {
      return emitIndicesForBlockedLayout(loc, b, blocked, shape);
    } else if (auto slice = layout.dyn_cast<SliceEncodingAttr>()) {
      return emitIndicesForSliceLayout(loc, b, slice, shape);
    } else {
      assert(0 && "emitIndices for layouts other than blocked & slice not "
                  "implemented yet");
      return {};
    }
  }

  SmallVector<SmallVector<Value>>
  emitIndicesForSliceLayout(Location loc, ConversionPatternRewriter &rewriter,
                            const SliceEncodingAttr &sliceLayout,
                            ArrayRef<int64_t> shape) const {
    auto parent = sliceLayout.getParent();
    unsigned dim = sliceLayout.getDim();
    size_t rank = shape.size();
    if (auto blockedParent = parent.dyn_cast<BlockedEncodingAttr>()) {
      SmallVector<int64_t> paddedShape(rank + 1);
      for (unsigned d = 0; d < rank + 1; ++d) {
        if (d < dim)
          paddedShape[d] = shape[d];
        else if (d == dim)
          paddedShape[d] = 1;
        else
          paddedShape[d] = shape[d - 1];
      }
      auto paddedIndices = emitIndicesForBlockedLayout(
          loc, rewriter, blockedParent, paddedShape);
      unsigned numIndices = paddedIndices.size();
      SmallVector<SmallVector<Value>> resultIndices(numIndices);
      for (unsigned i = 0; i < numIndices; ++i)
        for (unsigned d = 0; d < rank + 1; ++d)
          if (d != dim)
            resultIndices[i].push_back(paddedIndices[i][d]);

      return resultIndices;

    } else if (auto sliceParent = parent.dyn_cast<SliceEncodingAttr>()) {
      assert(0 && "emitIndicesForSliceLayout with parent of sliceLayout"
                  "is not implemented yet");
      return {};

    } else {
      assert(0 && "emitIndicesForSliceLayout with parent other than blocked & "
                  "slice not implemented yet");
      return {};
    }
  }

  // Emit indices calculation within each ConversionPattern, and returns a
  // [elemsPerThread X rank] index matrix.
  // TODO: [goostavz] Double confirm the redundant indices calculations will
  //       be eliminated in the consequent MLIR/LLVM optimization. We might
  //       implement a indiceCache if necessary.
  SmallVector<SmallVector<Value>>
  emitIndicesForBlockedLayout(Location loc, ConversionPatternRewriter &rewriter,
                              const BlockedEncodingAttr &blockedLayout,
                              ArrayRef<int64_t> shape) const {
    auto llvmIndexTy = this->getTypeConverter()->getIndexType();
    auto sizePerThread = blockedLayout.getSizePerThread();
    auto threadsPerWarp = blockedLayout.getThreadsPerWarp();
    auto warpsPerCTA = blockedLayout.getWarpsPerCTA();
    unsigned rank = shape.size();
    SmallVector<unsigned> shapePerCTA = getShapePerCTA(blockedLayout);
    SmallVector<unsigned> tilesPerDim(rank);
    for (unsigned k = 0; k < rank; ++k)
      tilesPerDim[k] = ceil<unsigned>(shape[k], shapePerCTA[k]);

    // step 1, delinearize threadId to get the base index
    auto multiDimBase =
        emitBaseIndexForBlockedLayout(loc, rewriter, blockedLayout, shape);

    // step 2, get offset of each element
    unsigned elemsPerThread = blockedLayout.getElemsPerThread(shape);
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
    // step 3, add offset to base, and reorder the sequence of indices to
    // guarantee that elems in the same sizePerThread are adjacent in order
    SmallVector<SmallVector<Value>> multiDimIdx(elemsPerThread,
                                                SmallVector<Value>(rank));
    unsigned totalSizePerThread = product<unsigned>(sizePerThread);

    for (unsigned n = 0; n < elemsPerThread; ++n) {
      unsigned linearNanoTileId = n / totalSizePerThread;
      unsigned linearNanoTileElemId = n % totalSizePerThread;
      SmallVector<unsigned> multiDimNanoTileId =
          getMultiDimIndex<unsigned>(linearNanoTileId, tilesPerDim);
      SmallVector<unsigned> multiDimNanoTileElemId =
          getMultiDimIndex<unsigned>(linearNanoTileElemId, sizePerThread);
      for (unsigned k = 0; k < rank; ++k) {
        unsigned reorderedMultiDimId =
            multiDimNanoTileId[k] *
                (sizePerThread[k] * threadsPerWarp[k] * warpsPerCTA[k]) +
            multiDimNanoTileElemId[k];
        multiDimIdx[n][k] =
            add(multiDimBase[k], idx_val(offset[k][reorderedMultiDimId]));
      }
    }

    return multiDimIdx;
  }

  template <typename T>
  Value getSharedMemoryBase(Location loc, ConversionPatternRewriter &rewriter,
                            T value) const {
    auto ptrTy = LLVM::LLVMPointerType::get(
        this->getTypeConverter()->convertType(rewriter.getI8Type()), 3);
    auto bufferId = allocation->getBufferId(value);
    assert(bufferId != Allocation::InvalidBufferId && "BufferId not found");
    size_t offset = allocation->getOffset(bufferId);
    auto llvmIndexTy = this->getTypeConverter()->getIndexType();
    Value offVal = idx_val(offset);
    Value base = gep(ptrTy, smem, offVal);
    return base;
  }

protected:
  const Allocation *allocation;
  Value smem;
};

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
  auto layout = tensorTy.getEncoding();
  auto srcType = typeConverter->convertType(elemType);
  auto llSrc = bitcast(srcType, constVal);
  size_t elemsPerThread = getElemsPerThread(layout, tensorTy.getShape());
  llvm::SmallVector<Value, 4> elems(elemsPerThread, llSrc);
  llvm::SmallVector<Type, 4> elemTypes(elems.size(), srcType);
  auto structTy =
      LLVM::LLVMStructType::getLiteral(rewriter.getContext(), elemTypes);

  return getStructFromElements(loc, elems, rewriter, structTy);
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
  LoadStoreConversionBase(AxisInfoAnalysis &axisAnalysisPass)
      : AxisAnalysisPass(axisAnalysisPass) {}

  // Get corresponding LLVM element values of \param value.
  SmallVector<Value> getLLVMElems(Value value, Value llValue,
                                  const BlockedEncodingAttr &layout,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) const {
    if (!value)
      return {};

    auto shape = value.getType().cast<RankedTensorType>().getShape();
    // Here, we assume that all inputs should have a blockedLayout
    auto valueVals = getElementsFromStruct(loc, llValue, rewriter);
    return valueVals;
  }

  // Get the blocked layout.
  std::tuple<BlockedEncodingAttr, unsigned> getLayout(Value val) const {
    auto ty = val.getType().cast<RankedTensorType>();
    // Here, we assume that all inputs should have a blockedLayout
    auto layout = ty.getEncoding().dyn_cast<BlockedEncodingAttr>();
    assert(layout && "unexpected layout in getLayout");
    auto shape = ty.getShape();
    unsigned valueElems = layout.getElemsPerThread(shape);
    return {layout, valueElems};
  }

  unsigned getAlignment(Value val, const BlockedEncodingAttr &layout) const {
    auto axisInfo = getAxisInfo(val);
    auto order = layout.getOrder();
    unsigned maxMultiple = axisInfo->getDivisibility(order[0]);
    unsigned maxContig = axisInfo->getContiguity(order[0]);
    unsigned alignment = std::min(maxMultiple, maxContig);
    return alignment;
  }

  unsigned getVectorizeSize(Value ptr,
                            const BlockedEncodingAttr &layout) const {
    auto axisInfo = getAxisInfo(ptr);
    // Here order should be ordered by contiguous first, so the first element
    // should have the largest contiguous.
    auto order = layout.getOrder();
    unsigned align = getAlignment(ptr, layout);

    auto ty = ptr.getType().dyn_cast<RankedTensorType>();
    assert(ty);
    auto shape = ty.getShape();

    unsigned contigPerThread = layout.getSizePerThread()[order[0]];
    unsigned vec = std::min(align, contigPerThread);
    vec = std::min<unsigned>(shape[order[0]], vec);

    return vec;
  }

  unsigned getMaskAlignment(Value mask) const {
    auto maskOrder = mask.getType()
                         .cast<RankedTensorType>()
                         .getEncoding()
                         .cast<BlockedEncodingAttr>()
                         .getOrder();

    auto maskAxis = getAxisInfo(mask);
    return std::max<int>(maskAxis->getConstancy(maskOrder[0]), 1);
  }

  llvm::Optional<AxisInfo> getAxisInfo(Value val) const {
    if (auto it = AxisAnalysisPass.lookupLatticeElement(val)) {
      return it->getValue();
    }

    return llvm::Optional<AxisInfo>{};
  }

protected:
  AxisInfoAnalysis &AxisAnalysisPass;
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
    Value ptr = op.ptr();
    Value mask = op.mask();
    Value other = op.other();

    Value llPtr = adaptor.ptr();
    Value llMask = adaptor.mask();
    Value llOther = adaptor.other();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!valueTy)
      return failure();
    Type valueElemTy =
        getTypeConverter()->convertType(valueTy.getElementType());

    auto [layout, numElems] = getLayout(ptr);

    auto ptrElems = getLLVMElems(ptr, llPtr, layout, rewriter, loc);
    assert(ptrElems.size() == numElems);
    // Determine the vectorization size
    size_t vec = getVectorizeSize(ptr, layout);

    SmallVector<Value> maskElems;
    if (llMask) {
      unsigned maskAlignment = getMaskAlignment(mask);
      maskElems = getLLVMElems(mask, llMask, layout, rewriter, loc);
      assert(ptrElems.size() == maskElems.size());

      size_t maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    const size_t dtsize =
        std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNbits = dtsize * 8;

    const int numVecs = numElems / vec;

    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (valueElemTy.isa<IntegerType>() &&
        matchPattern(op.other(), m_Constant(&constAttr)) &&
        constAttr.isSplat()) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }

    auto otherElems = getLLVMElems(other, llOther, layout, rewriter, loc);

    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      // TODO: optimization when ptr is GEP with constant offset
      size_t in_off = 0;

      const int maxWordWidth = std::max<int>(32, valueElemNbits);
      const int totalWidth = valueElemNbits * vec;
      const int width = std::min(totalWidth, maxWordWidth);
      const int nWords = std::max(1, totalWidth / width);
      const int wordNElems = width / valueElemNbits;
      const int vecNElems = totalWidth / valueElemNbits;
      assert(wordNElems * nWords * numVecs == numElems);

      // TODO(Superjomn) Add cache policy fields to StoreOp.
      // TODO(Superjomn) Deal with cache policy here.
      const bool hasL2EvictPolicy = false;

      PTXBuilder ptxBuilder;
      auto &ld = *ptxBuilder.create<PTXIOInstr>("ld");

      Value pred = mask ? maskElems[vecStart] : int_val(1, 1);

      const std::string readConstraint =
          (width == 64) ? "l" : ((width == 32) ? "r" : "c");
      const std::string writeConstraint =
          (width == 64) ? "=l" : ((width == 32) ? "=r" : "=c");

      // prepare asm operands
      auto *dstsOpr = ptxBuilder.newListOperand();
      for (int wordIdx = 0; wordIdx < nWords; ++wordIdx) {
        auto *opr = ptxBuilder.newOperand(writeConstraint); // =r operations
        dstsOpr->listAppend(opr);
      }

      auto *addrOpr =
          ptxBuilder.newAddrOperand(ptrElems[vecStart], "l", in_off);

      // Define the instruction opcode
      ld.o("volatile", op.isVolatile())
          .global()
          .o("ca", op.cache() == triton::CacheModifier::CA)
          .o("cg", op.cache() == triton::CacheModifier::CG)
          .o("L1::evict_first",
             op.evict() == triton::EvictionPolicy::EVICT_FIRST)
          .o("L1::evict_last", op.evict() == triton::EvictionPolicy::EVICT_LAST)
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
          PTXInstr &mov = *ptxBuilder.create<>("mov");
          mov.o("u", width);

          size_t size = width / valueElemNbits;

          auto vecTy = LLVM::getFixedVectorType(valueElemTy, size);
          Value v = rewriter.create<LLVM::UndefOp>(loc, vecTy);
          for (size_t s = 0; s < size; ++s) {
            Value falseVal = otherElems[vecStart + ii * size + s];
            Value sVal = createIndexAttrConstant(
                rewriter, loc, this->getTypeConverter()->getIndexType(), s);
            v = insert_element(vecTy, v, falseVal, sVal);
          }
          v = bitcast(IntegerType::get(getContext(), width), v);

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
      auto asmDialectAttr = LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                                      LLVM::AsmDialect::AD_ATT);
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
        curr = bitcast(
            LLVM::getFixedVectorType(valueElemTy, width / valueElemNbits),
            curr);
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

    auto valueTy = value.getType().dyn_cast<RankedTensorType>();
    if (!valueTy)
      return failure();
    Type valueElemTy =
        getTypeConverter()->convertType(valueTy.getElementType());

    auto [layout, numElems] = getLayout(ptr);

    auto ptrElems = getLLVMElems(ptr, llPtr, layout, rewriter, loc);
    auto valueElems = getLLVMElems(value, llValue, layout, rewriter, loc);
    assert(ptrElems.size() == valueElems.size());

    // Determine the vectorization size
    size_t vec = getVectorizeSize(ptr, layout);
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = getLLVMElems(mask, llMask, layout, rewriter, loc);
      assert(valueElems.size() == maskElems.size());

      size_t maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    const size_t dtsize =
        std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNbits = dtsize * 8;

    const int numVecs = numElems / vec;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      // TODO: optimization when ptr is AddPtr with constant offset
      size_t in_off = 0;

      const int maxWordWidth = std::max<int>(32, valueElemNbits);
      const int totalWidth = valueElemNbits * vec;
      const int width = std::min(totalWidth, maxWordWidth);
      const int nWords = std::max(1, totalWidth / width);
      const int wordNElems = width / valueElemNbits;
      const int vecNElems = totalWidth / valueElemNbits;
      assert(wordNElems * nWords * numVecs == numElems);

      // TODO(Superjomn) Add cache policy fields to StoreOp.
      // TODO(Superjomn) Deal with cache policy here.
      const bool hasL2EvictPolicy = false;

      Type valArgTy = IntegerType::get(ctx, width);
      auto wordTy = vec_ty(valueElemTy, wordNElems);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      for (int wordIdx = 0; wordIdx < nWords; ++wordIdx) {
        // llWord is a width-len composition
        Value llWord = rewriter.create<LLVM::UndefOp>(loc, wordTy);
        // Insert each value element to the composition
        for (int elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
          const size_t elemOffset = vecStart + wordIdx * wordNElems + elemIdx;
          assert(elemOffset < valueElems.size());
          Value elem = valueElems[elemOffset];
          if (elem.getType().isInteger(1))
            elem = rewriter.create<LLVM::SExtOp>(loc, type::i8Ty(ctx), elem);
          elem = bitcast(valueElemTy, elem);

          Type u32Ty = typeConverter->convertType(type::u32Ty(ctx));
          llWord =
              insert_element(wordTy, llWord, elem,
                             rewriter.create<LLVM::ConstantOp>(
                                 loc, u32Ty, IntegerAttr::get(u32Ty, elemIdx)));
        }
        llWord = bitcast(valArgTy, llWord);
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
          ptxBuilder.create<PTXIOInstr>("st")->global().b(width).v(nWords);
      ptxStoreInstr(asmAddr, asmArgList).predicate(maskVal, "b");

      Type boolTy = getTypeConverter()->convertType(rewriter.getIntegerType(1));
      llvm::SmallVector<Type> argTys({boolTy, ptr.getType()});
      argTys.insert(argTys.end(), nWords, valArgTy);

      auto ASMReturnTy = LLVM::LLVMVoidType::get(ctx);

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
    auto srcLayout = srcTy.getEncoding().dyn_cast<BlockedEncodingAttr>();
    auto resultLayout = resultTy.getEncoding().dyn_cast<BlockedEncodingAttr>();
    assert(srcLayout && (srcLayout == resultLayout) &&
           "Unexpected layout of BroadcastOp");
    auto srcShape = srcTy.getShape();
    auto resultShape = resultTy.getShape();
    unsigned rank = srcTy.getRank();
    assert(rank == resultTy.getRank());

    SmallVector<int64_t, 4> srcLogicalShape(2 * rank);
    SmallVector<int64_t, 4> resultLogicalShape(2 * rank);
    SmallVector<unsigned, 2> broadcastDims;
    for (unsigned d = 0; d < rank; ++d) {
      unsigned resultShapePerCTA = resultLayout.getSizePerThread()[d] *
                                   resultLayout.getThreadsPerWarp()[d] *
                                   resultLayout.getWarpsPerCTA()[d];
      int64_t numCtas = ceil<unsigned>(resultShape[d], resultShapePerCTA);
      if (srcShape[d] != resultShape[d]) {
        assert(srcShape[d] == 1);
        broadcastDims.push_back(d);
        srcLogicalShape[d] = 1;
        srcLogicalShape[d + rank] =
            std::max(unsigned(1), srcLayout.getSizePerThread()[d]);
      } else {
        srcLogicalShape[d] = numCtas;
        srcLogicalShape[d + rank] = resultLayout.getSizePerThread()[d];
      }
      resultLogicalShape[d] = numCtas;
      resultLogicalShape[d + rank] = resultLayout.getSizePerThread()[d];
    }
    int64_t duplicates = 1;
    SmallVector<int64_t, 2> broadcastSizes(broadcastDims.size() * 2);
    for (auto it : llvm::enumerate(broadcastDims)) {
      // Incase there are multiple indices in the src that is actually
      // calculating the same element, srcLogicalShape may not need to be 1.
      // Such as the case when src of shape [256, 1], and with a blocked layout:
      // sizePerThread: [1, 4];  threadsPerWarp: [1, 32]; warpsPerCTA: [1, 2]
      int64_t d = resultLogicalShape[it.value()] / srcLogicalShape[it.value()];
      broadcastSizes[it.index()] = d;
      duplicates *= d;
      d = resultLogicalShape[it.value() + rank] /
          srcLogicalShape[it.value() + rank];
      broadcastSizes[it.index() + broadcastDims.size()] = d;
      duplicates *= d;
    }

    unsigned srcElems = srcLayout.getElemsPerThread(srcShape);
    auto elemTy = resultTy.getElementType();
    auto srcVals = getElementsFromStruct(loc, src, rewriter);
    unsigned resultElems = resultLayout.getElemsPerThread(resultShape);
    SmallVector<Value> resultVals(resultElems);
    for (unsigned i = 0; i < srcElems; ++i) {
      auto srcMultiDim = getMultiDimIndex<int64_t>(i, srcLogicalShape);
      for (int64_t j = 0; j < duplicates; ++j) {
        auto resultMultiDim = srcMultiDim;
        auto bcastMultiDim = getMultiDimIndex<int64_t>(j, broadcastSizes);
        for (auto bcastDim : llvm::enumerate(broadcastDims)) {
          resultMultiDim[bcastDim.value()] += bcastMultiDim[bcastDim.index()];
          resultMultiDim[bcastDim.value() + rank] +=
              bcastMultiDim[bcastDim.index() + broadcastDims.size()] *
              srcLogicalShape[bcastDim.index() + broadcastDims.size()];
        }
        auto resultLinearIndex =
            getLinearIndex<int64_t>(resultMultiDim, resultLogicalShape);
        resultVals[resultLinearIndex] = srcVals[i];
      }
    }
    auto llvmStructTy = getTypeConverter()->convertType(resultTy);
    Value resultStruct =
        getStructFromElements(loc, resultVals, rewriter, llvmStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

template <typename SourceOp>
struct ViewLikeOpConversion : public ConvertTritonGPUOpToLLVMPattern<SourceOp> {
  using OpAdaptor = typename SourceOp::Adaptor;
  explicit ViewLikeOpConversion(LLVMTypeConverter &typeConverter,
                                PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<SourceOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We cannot directly
    //   rewriter.replaceOp(op, adaptor.src());
    // due to MLIR's restrictions
    Location loc = op->getLoc();
    auto resultTy = op.getType().template cast<RankedTensorType>();
    auto resultShape = resultTy.getShape();
    unsigned elems = getElemsPerThread(resultTy.getEncoding(), resultShape);
    Type elemTy =
        this->getTypeConverter()->convertType(resultTy.getElementType());
    SmallVector<Type> types(elems, elemTy);
    Type structTy = LLVM::LLVMStructType::getLiteral(this->getContext(), types);
    auto vals = this->getElementsFromStruct(loc, adaptor.src(), rewriter);
    Value view = getStructFromElements(loc, vals, rewriter, structTy);
    rewriter.replaceOp(op, view);
    return success();
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
    for (auto multiDim : llvm::enumerate(idxs)) {
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
    Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(
        loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x);
    auto llvmIndexTy = getTypeConverter()->getIndexType();
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, TypeRange{llvmIndexTy}, ValueRange{blockId});
    return success();
  }
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
      auto resultLayout =
          resultTensorTy.getEncoding().dyn_cast<BlockedEncodingAttr>();
      assert(resultLayout && "Unexpected resultLayout in AddPtrOpConversion");
      auto resultShape = resultTensorTy.getShape();
      unsigned elems = resultLayout.getElemsPerThread(resultShape);
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
    auto elemPtrTy = LLVM::LLVMPointerType::get(llvmElemTy, 3);
    Value resultVal =
        rewriter.create<LLVM::BitcastOp>(loc, elemPtrTy, smemBase);
    rewriter.replaceOp(op, resultVal);
    return success();
  }
};

struct ExtractSliceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::ExtractSliceOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::ExtractSliceOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto srcTy = op.src().getType().dyn_cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    assert(srcLayout && "Unexpected resultLayout in ExtractSliceOpConversion");

    // axis > 0 will result in non-contiguous memory access if the result tensor
    // is an alias of the source tensor.
    auto axis = op->getAttrOfType<IntegerAttr>("axis").getInt();
    assert(axis == 0 && "extract_slice: Only axis=0 is supported for now");

    // Example:
    // %dst = extract_slice %src, %index {axis = 0}
    // src.shape = [11, 2, 3, 4, 1]
    // offset = %index * 2 * 3 * 4 * 1
    auto dstTy = op.getType().dyn_cast<RankedTensorType>();
    auto base = product<int64_t>(dstTy.getShape());
    auto baseVal = createIndexAttrConstant(
        rewriter, loc, getTypeConverter()->getIndexType(), base);
    Value offset = mul(adaptor.index(), baseVal);

    auto llvmElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto elemPtrTy = LLVM::LLVMPointerType::get(llvmElemTy, 3);
    Value resultVal = gep(elemPtrTy, adaptor.src(), offset);
    rewriter.replaceOp(op, resultVal);
    return success();
  }
};

// A CRTP style of base class.
template <typename SourceOp, typename DestOp, typename ConcreteT>
class BinaryOpConversionBase
    : public ConvertTritonGPUOpToLLVMPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit BinaryOpConversionBase(LLVMTypeConverter &typeConverter,
                                  PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<SourceOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType().template dyn_cast<RankedTensorType>();
    // ArithmeticToLLVM will handle the lowering of scalar ArithOps
    if (!resultTy)
      return failure();

    Location loc = op->getLoc();
    auto resultLayout =
        resultTy.getEncoding().template dyn_cast<BlockedEncodingAttr>();
    auto resultShape = resultTy.getShape();
    assert(resultLayout && "Unexpected resultLayout in BinaryOpConversion");
    unsigned elems = resultLayout.getElemsPerThread(resultShape);
    Type elemTy =
        this->getTypeConverter()->convertType(resultTy.getElementType());
    SmallVector<Type> types(elems, elemTy);
    Type structTy = LLVM::LLVMStructType::getLiteral(this->getContext(), types);

    auto *concreteThis = static_cast<const ConcreteT *>(this);
    auto lhss = this->getElementsFromStruct(loc, concreteThis->getLhs(adaptor),
                                            rewriter);
    auto rhss = this->getElementsFromStruct(loc, concreteThis->getRhs(adaptor),
                                            rewriter);
    SmallVector<Value> resultVals(elems);
    for (unsigned i = 0; i < elems; ++i) {
      resultVals[i] = concreteThis->createDestOp(op, rewriter, elemTy, lhss[i],
                                                 rhss[i], loc);
    }
    Value view = getStructFromElements(loc, resultVals, rewriter, structTy);
    rewriter.replaceOp(op, view);
    return success();
  }
};

template <typename SourceOp, typename DestOp>
struct BinaryOpConversion
    : public BinaryOpConversionBase<SourceOp, DestOp,
                                    BinaryOpConversion<SourceOp, DestOp>> {

  explicit BinaryOpConversion(LLVMTypeConverter &typeConverter,
                              PatternBenefit benefit = 1)
      : BinaryOpConversionBase<SourceOp, DestOp,
                               BinaryOpConversion<SourceOp, DestOp>>(
            typeConverter, benefit) {}

  using OpAdaptor = typename SourceOp::Adaptor;
  // An interface to support variant DestOp builder.
  DestOp createDestOp(SourceOp op, ConversionPatternRewriter &rewriter,
                      Type elemTy, Value lhs, Value rhs, Location loc) const {
    return rewriter.create<DestOp>(loc, elemTy, lhs, rhs);
  }

  // Get the left operand of the op.
  Value getLhs(OpAdaptor adaptor) const { return adaptor.getLhs(); }
  // Get the right operand of the op.
  Value getRhs(OpAdaptor adaptor) const { return adaptor.getRhs(); }
};

struct CmpIOpConversion
    : public BinaryOpConversionBase<triton::gpu::CmpIOp, LLVM::ICmpOp,
                                    CmpIOpConversion> {
  explicit CmpIOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit = 1)
      : BinaryOpConversionBase(typeConverter, benefit) {}

  // An interface to support variant DestOp builder.
  LLVM::ICmpOp createDestOp(triton::gpu::CmpIOp op,
                            ConversionPatternRewriter &rewriter, Type elemTy,
                            Value lhs, Value rhs, Location loc) const {
    return rewriter.create<LLVM::ICmpOp>(
        loc, elemTy, ArithCmpIPredicteToLLVM(op.predicate()), lhs, rhs);
  }

  // Get the left operand of the op.
  Value getLhs(OpAdaptor adaptor) const { return adaptor.lhs(); }
  // Get the right operand of the op.
  Value getRhs(OpAdaptor adaptor) const { return adaptor.rhs(); }

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
    : public BinaryOpConversionBase<triton::gpu::CmpFOp, LLVM::FCmpOp,
                                    CmpFOpConversion> {
  explicit CmpFOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit = 1)
      : BinaryOpConversionBase(typeConverter, benefit) {}

  // An interface to support variant DestOp builder.
  LLVM::FCmpOp createDestOp(triton::gpu::CmpFOp op,
                            ConversionPatternRewriter &rewriter, Type elemTy,
                            Value lhs, Value rhs, Location loc) const {
    return rewriter.create<LLVM::FCmpOp>(
        loc, elemTy, ArithCmpFPredicteToLLVM(op.predicate()), lhs, rhs);
  }

  // Get the left operand of the op.
  Value getLhs(OpAdaptor adaptor) const { return adaptor.lhs(); }
  // Get the right operand of the op.
  Value getRhs(OpAdaptor adaptor) const { return adaptor.rhs(); }

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
    if ((!srcLayout.isa<BlockedEncodingAttr>() &&
         !srcLayout.isa<MmaEncodingAttr>()) ||
        (!dstLayout.isa<BlockedEncodingAttr>() &&
         !dstLayout.isa<MmaEncodingAttr>())) {
      // TODO: to be implemented
      return failure();
    }

    return lowerDistributedToDistributed(op, adaptor, rewriter);
  }

private:
  template <typename T>
  SmallVector<T> reorder(ArrayRef<T> input, ArrayRef<unsigned> order) const {
    size_t rank = order.size();
    assert(input.size() == rank);
    SmallVector<T> result(rank);
    for (auto it : llvm::enumerate(order)) {
      result[rank - 1 - it.value()] = input[it.index()];
    }
    return result;
  };

  // shared memory rd/st for blocked or mma layout with data padding
  void processReplica(Location loc, ConversionPatternRewriter &rewriter,
                      bool stNotRd, RankedTensorType type,
                      ArrayRef<unsigned> numCTAsEachRep,
                      ArrayRef<unsigned> multiDimRepId, unsigned vec,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> outOrd, SmallVector<Value> &vals,
                      Value smemBase) const;

  // blocked/mma -> blocked/mma.
  // Data padding in shared memory to avoid bank confict.
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
};

void ConvertLayoutOpConversion::processReplica(
    Location loc, ConversionPatternRewriter &rewriter, bool stNotRd,
    RankedTensorType type, ArrayRef<unsigned> numCTAsEachRep,
    ArrayRef<unsigned> multiDimRepId, unsigned vec,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> outOrd,
    SmallVector<Value> &vals, Value smemBase) const {
  unsigned accumNumCTAsEachRep = product<unsigned>(numCTAsEachRep);
  auto layout = type.getEncoding();
  auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>();
  auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>();
  auto rank = type.getRank();
  auto sizePerThread = getSizePerThread(layout);
  auto accumSizePerThread = product<unsigned>(sizePerThread);
  auto llvmIndexTy = getTypeConverter()->getIndexType();
  SmallVector<unsigned> numCTAs(rank);
  auto shapePerCTA = getShapePerCTA(layout);
  for (unsigned d = 0; d < rank; ++d) {
    numCTAs[d] = ceil<unsigned>(type.getShape()[d], shapePerCTA[d]);
  }
  auto llvmElemTy = getTypeConverter()->convertType(type.getElementType());
  SmallVector<Value> multiDimOffsetFirstElem;
  SmallVector<Value> mmaColIdx(2);
  SmallVector<Value> mmaRowIdx(2);
  if (blockedLayout) {
    multiDimOffsetFirstElem = emitBaseIndexForBlockedLayout(
        loc, rewriter, blockedLayout, type.getShape());
  } else if (mmaLayout) {
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = idx_val(32);
    Value laneId = urem(threadId, warpSize);
    Value warpId = udiv(threadId, warpSize);
    // auto multiDimWarpId =
    //     delinearize(rewriter, loc, warpId, mmaLayout.getWarpsPerCTA());
    // TODO: double confirm if its document bug or DotConversion's Bug
    SmallVector<Value> multiDimWarpId(2);
    multiDimWarpId[0] = urem(warpId, idx_val(mmaLayout.getWarpsPerCTA()[0]));
    multiDimWarpId[1] = udiv(warpId, idx_val(mmaLayout.getWarpsPerCTA()[0]));
    Value four = idx_val(4);
    Value mmaGrpId = udiv(laneId, four);
    Value mmaGrpIdP8 = add(mmaGrpId, idx_val(8));
    Value mmaThreadIdInGrp = urem(laneId, four);
    Value mmaThreadIdInGrpM2 = mul(mmaThreadIdInGrp, idx_val(2));
    Value mmaThreadIdInGrpM2P1 = add(mmaThreadIdInGrpM2, idx_val(1));
    Value colWarpOffset = mul(multiDimWarpId[0], idx_val(16));
    mmaColIdx[0] = add(mmaGrpId, colWarpOffset);
    mmaColIdx[1] = add(mmaGrpIdP8, colWarpOffset);
    Value rowWarpOffset = mul(multiDimWarpId[1], idx_val(8));
    mmaRowIdx[0] = add(mmaThreadIdInGrpM2, rowWarpOffset);
    mmaRowIdx[1] = add(mmaThreadIdInGrpM2P1, rowWarpOffset);
  }
  for (unsigned ctaId = 0; ctaId < accumNumCTAsEachRep; ++ctaId) {
    auto multiDimCTAInRepId = getMultiDimIndex<unsigned>(ctaId, numCTAsEachRep);
    SmallVector<unsigned> multiDimCTAId(rank);
    for (auto it : llvm::enumerate(multiDimCTAInRepId)) {
      auto d = it.index();
      multiDimCTAId[d] = multiDimRepId[d] * numCTAsEachRep[d] + it.value();
    }

    unsigned linearCTAId = getLinearIndex<unsigned>(multiDimCTAId, numCTAs);
    // TODO: This is actually redundant index calculation, we should
    //       consider of caching the index calculation result in case
    //       of performance issue observed.
    for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
      SmallVector<Value> multiDimOffset(rank);
      if (blockedLayout) {
        SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
            elemId, blockedLayout.getSizePerThread());
        for (unsigned d = 0; d < rank; ++d) {
          multiDimOffset[d] =
              add(multiDimOffsetFirstElem[d],
                  idx_val(multiDimCTAInRepId[d] * shapePerCTA[d] +
                          multiDimElemId[d]));
        }
      } else if (mmaLayout) {
        assert(rank == 2);
        assert(mmaLayout.getVersion() == 2 &&
               "mmaLayout ver1 not implemented yet");
        multiDimOffset[0] = elemId < 2 ? mmaColIdx[0] : mmaColIdx[1];
        multiDimOffset[1] = elemId % 2 == 0 ? mmaRowIdx[0] : mmaRowIdx[1];
        multiDimOffset[0] = add(
            multiDimOffset[0], idx_val(multiDimCTAInRepId[0] * shapePerCTA[0]));
        multiDimOffset[1] = add(
            multiDimOffset[1], idx_val(multiDimCTAInRepId[1] * shapePerCTA[1]));
      } else {
        assert(0 && "unexpected layout in processReplica");
      }
      Value offset =
          linearize(rewriter, loc, reorder<Value>(multiDimOffset, outOrd),
                    reorder<unsigned>(paddedRepShape, outOrd));
      auto elemPtrTy = ptr_ty(llvmElemTy, 3);
      Value ptr = gep(elemPtrTy, smemBase, offset);
      auto vecTy = vec_ty(llvmElemTy, vec);
      ptr = bitcast(ptr_ty(vecTy, 3), ptr);
      if (stNotRd) {
        Value valVec = undef(vecTy);
        for (unsigned v = 0; v < vec; ++v) {
          valVec = insert_element(
              vecTy, valVec,
              vals[elemId + linearCTAId * accumSizePerThread + v], idx_val(v));
        }
        store(valVec, ptr);
      } else {
        Value valVec = load(ptr);
        for (unsigned v = 0; v < vec; ++v) {
          vals[elemId + linearCTAId * accumSizePerThread + v] =
              extract_element(llvmElemTy, valVec, idx_val(v));
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
  smemBase = bitcast(elemPtrTy, smemBase);
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
  unsigned accumNumReplicates = product<unsigned>(numReplicates);
  unsigned elems = getElemsPerThread(srcLayout, srcTy.getShape());
  auto vals = getElementsFromStruct(loc, adaptor.src(), rewriter);
  unsigned inVec = 0;
  unsigned outVec = 0;
  auto paddedRepShape = getScratchConfigForCvtLayout(op, inVec, outVec);

  unsigned outElems = getElemsPerThread(dstLayout, shape);
  auto outOrd = getOrder(dstLayout);
  SmallVector<Value> outVals(outElems);

  for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
    auto multiDimRepId = getMultiDimIndex<unsigned>(repId, numReplicates);
    barrier;
    if (srcLayout.isa<BlockedEncodingAttr>() ||
        srcLayout.isa<MmaEncodingAttr>()) {
      processReplica(loc, rewriter, /*stNotRd*/ true, srcTy, inNumCTAsEachRep,
                     multiDimRepId, inVec, paddedRepShape, outOrd, vals,
                     smemBase);
    } else {
      assert(0 && "ConvertLayout with input layout not implemented");
      return failure();
    }
    barrier;
    if (dstLayout.isa<BlockedEncodingAttr>() ||
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
  Type structTy = struct_ty(getContext(), types);
  Value result = getStructFromElements(loc, outVals, rewriter, structTy);
  rewriter.replaceOp(op, result);

  return success();
};

LogicalResult ConvertLayoutOpConversion::lowerBlockedToShared(
    triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value src = op.src();
  Value dst = op.result();
  auto srcTy = src.getType().cast<RankedTensorType>();
  auto dstTy = dst.getType().cast<RankedTensorType>();
  auto srcShape = srcTy.getShape();
  assert(srcShape.size() == 2 &&
         "Unexpected rank of ConvertLayout(blocked->shared)");
  auto srcBlockedLayout = srcTy.getEncoding().cast<BlockedEncodingAttr>();
  auto dstSharedLayout = dstTy.getEncoding().cast<SharedEncodingAttr>();
  auto inOrd = srcBlockedLayout.getOrder();
  auto outOrd = dstSharedLayout.getOrder();
  unsigned inVec =
      inOrd == outOrd ? srcBlockedLayout.getSizePerThread()[inOrd[0]] : 1;
  unsigned outVec = dstSharedLayout.getVec();
  unsigned minVec = std::min(outVec, inVec);
  unsigned perPhase = dstSharedLayout.getPerPhase();
  unsigned maxPhase = dstSharedLayout.getMaxPhase();
  unsigned numElems = getElemsPerThread(srcBlockedLayout, srcShape);
  auto inVals = getElementsFromStruct(loc, adaptor.src(), rewriter);
  unsigned srcAccumSizeInThreads =
      product<unsigned>(srcBlockedLayout.getSizePerThread());
  auto elemTy = srcTy.getElementType();
  auto wordTy = vec_ty(elemTy, minVec);

  // TODO: [goostavz] We should make a cache for the calculation of
  // emitBaseIndexForBlockedLayout in case backend compiler not being able to
  // optimize that
  SmallVector<Value> multiDimOffsetFirstElem =
      emitBaseIndexForBlockedLayout(loc, rewriter, srcBlockedLayout, srcShape);
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
  Value outVecVal = idx_val(outVec);
  Value minVecVal = idx_val(minVec);
  Value smemBase = getSharedMemoryBase(loc, rewriter, dst);
  auto elemPtrTy = ptr_ty(getTypeConverter()->convertType(elemTy), 3);
  smemBase = bitcast(elemPtrTy, smemBase);
  unsigned numWordsEachRep = product<unsigned>(wordsInEachRep);
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
        linearIdxInNanoTile, srcBlockedLayout.getSizePerThread());
    unsigned pos = multiDimIdxInNanoTile[inOrd[0]] % minVec;
    multiDimIdxInNanoTile[inOrd[0]] /= minVec;
    unsigned wordVecIdx =
        getLinearIndex<unsigned>(multiDimIdxInNanoTile, wordsInEachRep);
    wordVecs[wordVecIdx] =
        insert_element(wordTy, wordVecs[wordVecIdx], inVals[i], idx_val(pos));

    if (i % srcAccumSizeInThreads == srcAccumSizeInThreads - 1) {
      // end of replication, store the vectors into shared memory
      unsigned linearRepIdx = i / srcAccumSizeInThreads;
      auto multiDimRepIdx = getMultiDimIndex<unsigned>(linearRepIdx, reps);
      for (unsigned linearWordIdx = 0; linearWordIdx < numWordsEachRep;
           ++linearWordIdx) {
        // step 1: recover the multidim_index from the index of input_elements
        auto multiDimWordIdx =
            getMultiDimIndex<unsigned>(linearWordIdx, wordsInEachRep);
        SmallVector<Value> multiDimIdx(2);
        auto wordOffset0 = multiDimRepIdx[0] * srcShapePerCTA[0] +
                           multiDimWordIdx[0] * (inOrd[0] == 0 ? minVec : 1);
        auto wordOffset1 = multiDimRepIdx[1] * srcShapePerCTA[1] +
                           multiDimWordIdx[1] * (inOrd[0] == 1 ? minVec : 1);
        multiDimIdx[0] = add(multiDimOffsetFirstElem[0], idx_val(wordOffset0));
        multiDimIdx[1] = add(multiDimOffsetFirstElem[1], idx_val(wordOffset1));

        // step 2: do swizzling
        Value remained = urem(multiDimIdx[inOrd[0]], outVecVal);
        multiDimIdx[inOrd[0]] = udiv(multiDimIdx[inOrd[0]], outVecVal);
        Value off_1 = mul(multiDimIdx[inOrd[1]], idx_val(srcShape[inOrd[0]]));
        Value phaseId = udiv(multiDimIdx[inOrd[1]], idx_val(perPhase));
        phaseId = urem(phaseId, idx_val(maxPhase));
        Value off_0 = xor_(multiDimIdx[inOrd[0]], phaseId);
        off_0 = mul(off_0, outVecVal);
        remained = udiv(remained, minVecVal);
        off_0 = add(off_0, mul(remained, minVecVal));
        Value offset = add(off_1, off_0);

        // step 3: store
        Value smemAddr = gep(elemPtrTy, smemBase, offset);
        smemAddr = bitcast(ptr_ty(wordTy, 3), smemAddr);
        store(wordVecs[linearWordIdx], smemAddr);
      }
    }
  }
  // TODO: double confirm if the Barrier is necessary here
  barrier;
  rewriter.replaceOp(op, smemBase);
  return success();
}

/// ====================== dot codegen begin ==========================

// Data loader for mma.16816 instruction.
class MMA16816SmemLoader {
public:
  MMA16816SmemLoader(int wpt, ArrayRef<uint32_t> order, int kOrder,
                     ArrayRef<int64_t> tileShape, ArrayRef<int> instrShape,
                     ArrayRef<int> matShape, int perPhase, int maxPhase,
                     int elemBytes, ConversionPatternRewriter &rewriter,
                     TypeConverter *typeConverter, const Location &loc)
      : wpt(wpt), order(order.begin(), order.end()), kOrder(kOrder),
        tileShape(tileShape.begin(), tileShape.end()),
        instrShape(instrShape.begin(), instrShape.end()),
        matShape(matShape.begin(), matShape.end()), perPhase(perPhase),
        maxPhase(maxPhase), elemBytes(elemBytes), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(rewriter.getContext()) {
    cMatShape = matShape[order[0]];
    sMatShape = matShape[order[1]];

    cTileStride = tileShape[order[1]];
    sTileStride = tileShape[order[0]];

    // rule: k must be the fast-changing axis.
    needTrans = kOrder != order[0];
    canUseLdmatrix = elemBytes == 2 || (!needTrans); // b16

    if (canUseLdmatrix) {
      // Each CTA, the warps is arranged as [1xwpt] if not transposed,
      // otherwise [wptx1], and each warp will perform a mma.
      numPtr =
          tileShape[order[0]] / (needTrans ? wpt : 1) / instrShape[order[0]];
    } else {
      numPtr = tileShape[order[0]] / wpt / matShape[order[0]];
    }

    numPtr = std::max<int>(numPtr, 2);

    // Special rule for i8/u8, 4 ptrs for each matrix
    if (!canUseLdmatrix && elemBytes == 1)
      numPtr *= 4;

    int loadStrideInMat[2];
    loadStrideInMat[kOrder] =
        2; // instrShape[kOrder] / matShape[kOrder], always 2
    loadStrideInMat[kOrder ^ 1] =
        wpt * (instrShape[kOrder ^ 1] / matShape[kOrder ^ 1]);

    pLoadStrideInMat = loadStrideInMat[order[0]];
    sMatStride =
        loadStrideInMat[order[1]] / (instrShape[order[1]] / matShape[order[1]]);

    // Each matArr contains warpOffStride matrices.
    matArrStride = kOrder == 1 ? 1 : wpt;
    warpOffStride = instrShape[kOrder ^ 1] / matShape[kOrder ^ 1];
  }

  // lane = thread % 32
  // warpOff = (thread/32) % wpt(0)
  llvm::SmallVector<Value> computeOffsets(Value warpOff, Value lane) {
    if (canUseLdmatrix)
      return computeLdmatrixMatOffs(warpOff, lane);
    else if (elemBytes == 4 && needTrans)
      return computeB32MatOffs(warpOff, lane);
    else if (elemBytes == 1 && needTrans)
      return computeB8MatOffs(warpOff, lane);
    else
      llvm::report_fatal_error("Invalid smem load config");

    return {};
  }

  int getNumPtr() const { return numPtr; }

  // Compute the offset to the matrix this thread(indexed by warpOff and lane)
  // mapped to.
  SmallVector<Value> computeLdmatrixMatOffs(Value warpId, Value lane) {
    // 4x4 matrices
    Value c = urem(lane, i32_val(8));
    Value s = udiv(lane, i32_val(8)); // sub-warp-id

    // Decompose s => s_0, s_1, that is the coordinate in 2x2 matrices in a warp
    Value s0 = urem(s, i32_val(2));
    Value s1 = udiv(s, i32_val(2));

    // We use different orders for a and b for better performance.
    Value kMatArr = kOrder == 1 ? s1 : s0;
    Value nkMatArr = kOrder == 1 ? s0 : s1;

    // matrix coordinate inside a CTA, the matrix layout is [2x2wpt] for A and
    // [2wptx2] for B. e.g. Setting wpt=3, The data layout for A(kOrder=1) is
    //   |0 0 1 1 2 2| -> 0,1,2 are the warpids
    //   |0 0 1 1 2 2|
    //
    // for B(kOrder=0) is
    //   |0 0|  -> 0,1,2 are the warpids
    //   |1 1|
    //   |2 2|
    //   |0 0|
    //   |1 1|
    //   |2 2|
    // Note, for each warp, it handles a 2x2 matrices, that is the coordinate
    // address (s0,s1) annotates.

    Value matOff[2];
    matOff[kOrder ^ 1] = add(
        mul(warpId, i32_val(warpOffStride)),   // warp offset
        mul(nkMatArr, i32_val(matArrStride))); // matrix offset inside a warp
    matOff[kOrder] = kMatArr;

    // Physical offset (before swizzling)
    Value cMatOff = matOff[order[0]];
    Value sMatOff = matOff[order[1]];

    // row offset inside a matrix, each matrix has 8 rows.
    Value sOffInMat = c;

    SmallVector<Value> offs(numPtr);
    Value phase = urem(udiv(sOffInMat, i32_val(perPhase)), i32_val(maxPhase));
    Value sOff = add(sOffInMat, mul(sMatOff, i32_val(sMatShape)));
    for (int i = 0; i < numPtr; ++i) {
      Value cMatOffI = add(cMatOff, i32_val(i * pLoadStrideInMat));
      cMatOffI = xor_(cMatOffI, phase);
      offs[i] = add(mul(cMatOffI, i32_val(cMatShape)),
                    mul(sOff, i32_val(sTileStride)));
    }

    return offs;
  }

  // Compute 32-bit matrix offsets.
  SmallVector<Value> computeB32MatOffs(Value warpOff, Value lane) {
    assert(needTrans && "Only used in transpose mode.");
    // Load tf32 matrices with lds32
    Value cOffInMat = udiv(lane, i32_val(4));
    Value sOffInMat = urem(lane, i32_val(4));

    Value phase = urem(udiv(sOffInMat, i32_val(perPhase)), i32_val(maxPhase));
    SmallVector<Value> offs(numPtr);

    for (int mat = 0; mat < 4; ++mat) { // Load 4 mats each time
      int kMatArrInt = kOrder == 1 ? mat / 2 : mat % 2;
      int nkMatArrInt = kOrder == 1 ? mat % 2 : mat / 2;
      if (kMatArrInt > 0) // we don't need pointers for k
        continue;
      Value kMatArr = i32_val(kMatArrInt);
      Value nkMatArr = i32_val(nkMatArrInt);

      Value cMatOff = add(mul(warpOff, i32_val(warpOffStride)),
                          mul(nkMatArr, i32_val(matArrStride)));
      Value sMatOff = kMatArr;
      Value sOff = add(sOffInMat, mul(sMatOff, i32_val(sMatShape)));
      // FIXME: (kOrder == 1?) is really dirty hack
      for (int i = 0; i < numPtr / 2; ++i) {
        Value cMatOffI =
            add(cMatOff, i32_val(i * pLoadStrideInMat * (kOrder == 1 ? 1 : 2)));
        cMatOffI = xor_(cMatOffI, phase);
        Value cOff = add(cOffInMat, mul(cMatOffI, i32_val(cMatShape)));
        cOff = urem(cOff, i32_val(tileShape[order[0]]));
        sOff = urem(sOff, i32_val(tileShape[order[1]]));
        offs[2 * i + nkMatArrInt] = add(cOff, mul(sOff, i32_val(sTileStride)));
      }
    }
    return offs;
  }

  // compute 8-bit matrix offset.
  SmallVector<Value> computeB8MatOffs(Value warpOff, Value lane) {
    assert(needTrans && "Only used in transpose mode.");
    Value cOffInMat = udiv(lane, i32_val(4));
    Value sOffInMat =
        mul(urem(lane, i32_val(4)), i32_val(4)); // each thread load 4 cols

    SmallVector<Value> offs(numPtr);
    for (int mat = 0; mat < 4; ++mat) {
      int kMatArrInt = kOrder == 1 ? mat / 2 : mat % 2;
      int nkMatArrInt = kOrder == 1 ? mat % 2 : mat / 2;
      if (kMatArrInt > 0) // we don't need pointers for k
        continue;
      Value kMatArr = i32_val(kMatArrInt);
      Value nkMatArr = i32_val(nkMatArrInt);

      Value cMatOff = add(mul(warpOff, i32_val(warpOffStride)),
                          mul(nkMatArr, i32_val(matArrStride)));
      Value sMatOff = kMatArr;

      for (int loadx4Off = 0; loadx4Off < numPtr / 8; ++loadx4Off) {
        for (int elemOff = 0; elemOff < 4; ++elemOff) {
          int ptrOff = loadx4Off * 8 + nkMatArrInt * 4 + elemOff;
          Value cMatOffI = add(cMatOff, i32_val(loadx4Off * pLoadStrideInMat *
                                                (kOrder == 1 ? 1 : 2)));
          Value sOffInMatElem = add(sOffInMat, i32_val(elemOff));

          // disable swizzling ...

          Value cOff = add(cOffInMat, mul(cMatOffI, i32_val(cMatShape)));
          Value sOff = add(sOffInMatElem, mul(sMatOff, i32_val(sMatShape)));
          // To prevent out-of-bound access when tile is too small.
          cOff = urem(cOff, i32_val(tileShape[order[0]]));
          sOff = urem(sOff, i32_val(tileShape[order[1]]));
          offs[ptrOff] = add(cOff, mul(sOff, i32_val(sTileStride)));
        }
      }
    }
    return offs;
  }

  // Load 4 matrices and returns 4 vec<2> elements.
  std::tuple<Value, Value, Value, Value>
  loadX4(int mat0, int mat1, ArrayRef<Value> offs, ArrayRef<Value> ptrs,
         Type ldmatrixRetTy, Type shemPtrTy) const {
    assert(mat0 % 2 == 0 && mat1 % 2 == 0 &&
           "smem matrix load must be aligned");
    int matIdx[2] = {mat0, mat1};
    int k = matIdx[kOrder];

    int ptrIdx{-1};

    if (canUseLdmatrix)
      ptrIdx = matIdx[order[0]] / (instrShape[order[0]] / matShape[order[0]]);
    else if (elemBytes == 4 && needTrans)
      ptrIdx = matIdx[order[0]];
    else if (elemBytes == 1 && needTrans)
      ptrIdx = matIdx[order[0]] * 4;
    else
      llvm::report_fatal_error("unsupported mma type found");

    // The main difference with the original triton code is we removed the
    // prefetch-related logic here for the upstream optimizer phase should take
    // care with it, and that is transparent in dot conversion.
    auto getPtr = [&](int idx) { return ptrs[idx]; };

    Value ptr = getPtr(ptrIdx);

    Value resV4;
    if (canUseLdmatrix) {
      int sOffset =
          matIdx[order[1]] * sMatStride * sMatShape * sTileStride * elemBytes;
      PTXBuilder builder;

      // ldmatrix.m8n8.x4 returns 4x2xfp16(that is 4xb32) elements for a thread.
      auto resArgs = builder.newListOperand(4, "=r");
      auto addrArg = builder.newAddrOperand(ptr, "r", sOffset);

      auto ldmatrix = builder.create("ldmatrix.sync.aligned.m8n8.x4")
                          ->o("trans", needTrans /*predicate*/)
                          .o("shared.b16");
      ldmatrix(resArgs, addrArg);

      // The result type is 4xi32, each i32 is composed of 2xf16
      // elements(adjacent two columns in a row)
      Value resV4 = builder.launch(rewriter, loc, ldmatrixRetTy);

      auto getIntAttr = [&](int v) {
        return ArrayAttr::get(ctx, {IntegerAttr::get(i32_ty, v)});
      };

      Type fp16x2Ty = vec_ty(type::f16Ty(ctx), 2);

      return {extract_val(fp16x2Ty, resV4, getIntAttr(0)),
              extract_val(fp16x2Ty, resV4, getIntAttr(1)),
              extract_val(fp16x2Ty, resV4, getIntAttr(2)),
              extract_val(fp16x2Ty, resV4, getIntAttr(3))};
    } else if (elemBytes == 4 &&
               needTrans) { // Use lds.32 to load tf32 matrices
      Value ptr2 = getPtr(ptrIdx + 1);
      assert(sMatStride == 1);
      int sOffsetElem =
          matIdx[order[1]] * (sMatStride * sMatShape) * sTileStride;
      int sOffsetArrElem = 1 * (sMatStride * sMatShape) * sTileStride;

      Value elems[4];
      Type elemTy = type::f32Ty(ctx);
      if (kOrder == 1) {
        elems[0] = load(gep(elemTy, ptr, i32_val(sOffsetElem)));
        elems[1] = load(gep(elemTy, ptr2, i32_val(sOffsetElem)));
        elems[2] =
            load(gep(elemTy, ptr, i32_val(sOffsetElem + sOffsetArrElem)));
        elems[3] =
            load(gep(elemTy, ptr2, i32_val(sOffsetElem + sOffsetArrElem)));
      } else {
        elems[0] = load(gep(elemTy, ptr, i32_val(sOffsetElem)));
        elems[2] = load(gep(elemTy, ptr2, i32_val(sOffsetElem)));
        elems[1] =
            load(gep(elemTy, ptr, i32_val(sOffsetElem + sOffsetArrElem)));
        elems[3] =
            load(gep(elemTy, ptr2, i32_val(sOffsetElem + sOffsetArrElem)));
      }

      return {elems[0], elems[1], elems[2], elems[3]};
    } else if (elemBytes == 1 && needTrans) {
      std::array<std::array<Value, 4>, 2> ptrs;
      ptrs[0] = {
          getPtr(ptrIdx),
          getPtr(ptrIdx + 1),
          getPtr(ptrIdx + 2),
          getPtr(ptrIdx + 3),
      };

      ptrs[1] = {
          getPtr(ptrIdx + 4),
          getPtr(ptrIdx + 5),
          getPtr(ptrIdx + 6),
          getPtr(ptrIdx + 7),
      };

      assert(sMatStride == 1);
      int sOffsetElem =
          matIdx[order[1]] * (sMatStride * sMatShape) * sTileStride;
      int sOffsetArrElem = 1 * (sMatStride * sMatShape) * sTileStride;

      std::array<Value, 4> i8v4Elems;
      std::array<Value, 4> i32Elems;
      i8v4Elems.fill(
          rewriter.create<LLVM::UndefOp>(loc, vec_ty(type::i8Ty(ctx), 4)));

      Value i8Elems[4][4];
      Type elemTy = type::i8Ty(ctx);
      if (kOrder == 1) {
        Value offset = i32_val(sOffsetElem);

        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 4; ++j)
            i8Elems[i][j] = load(gep(elemTy, ptrs[i][j], offset));

        offset = i32_val(sOffsetElem + sOffsetArrElem);
        for (int i = 2; i < 4; ++i)
          for (int j = 0; j < 4; ++j)
            i8Elems[i][j] = load(gep(elemTy, ptrs[i - 2][j], offset));

        for (int m = 0; m < 4; ++m) {
          for (int e = 0; e < 4; ++e)
            i8v4Elems[m] = insert_element(i8v4Elems[m].getType(), i8v4Elems[m],
                                          i8Elems[m][e], i32_val(e));
          i32Elems[m] = bitcast(i32_ty, i8v4Elems[m]);
        }
      } else { // k first
        Value offset = i32_val(sOffsetElem);
        for (int j = 0; j < 4; ++j)
          i8Elems[0][j] = load(gep(elemTy, ptrs[0][j], offset));
        for (int j = 0; j < 4; ++j)
          i8Elems[2][j] = load(gep(elemTy, ptrs[1][j], offset));
        offset = i32_val(sOffsetElem + sOffsetArrElem);
        for (int j = 0; j < 4; ++j)
          i8Elems[1][j] = load(gep(elemTy, ptrs[0][j], offset));
        for (int j = 0; j < 4; ++j)
          i8Elems[3][j] = load(gep(elemTy, ptrs[1][j], offset));

        for (int m = 0; m < 4; ++m) {
          for (int e = 0; e < 4; ++e)
            i8v4Elems[m] = insert_element(i8v4Elems[m].getType(), i8v4Elems[m],
                                          i8Elems[m][e], i32_val(e));
          i32Elems[m] = bitcast(i32_ty, i8v4Elems[m]);
        }
      }

      return {i32Elems[0], i32Elems[1], i32Elems[2], i32Elems[3]};
    }

    assert(false && "Invalid smem load");
    return {Value{}, Value{}, Value{}, Value{}};
  }

private:
  int wpt;
  SmallVector<uint32_t> order;
  int kOrder;
  SmallVector<int64_t> tileShape;
  SmallVector<int> instrShape;
  SmallVector<int> matShape;
  int perPhase;
  int maxPhase;
  int elemBytes;
  ConversionPatternRewriter &rewriter;
  TypeConverter *typeConverter{};
  const Location &loc;
  MLIRContext *ctx{};

  int cMatShape;
  int sMatShape;

  int cTileStride;
  int sTileStride;

  bool needTrans;
  bool canUseLdmatrix;

  int numPtr;

  int pLoadStrideInMat;
  int sMatStride;

  int matArrStride;
  int warpOffStride;
};

bool isSplatLike(Value value) {
  if (auto constv = dyn_cast<arith::ConstantOp>(value.getDefiningOp()))
    if (auto attr = constv.getValue().dyn_cast<SplatElementsAttr>())
      return attr.isSplat();
  return false;
}

struct DotOpConversion : public ConvertTritonGPUOpToLLVMPattern<triton::DotOp> {
  enum class TensorCoreType : uint8_t {
    // floating-point tensor core instr
    FP32_FP16_FP16_FP32 = 0, // default
    FP32_BF16_BF16_FP32,
    FP32_TF32_TF32_FP32,
    // integer tensor core instr
    INT32_INT1_INT1_INT32, // Not implemented
    INT32_INT4_INT4_INT32, // Not implemented
    INT32_INT8_INT8_INT32, // Not implemented
    //
    NOT_APPLICABLE,
  };

  using ConvertTritonGPUOpToLLVMPattern<
      triton::DotOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value A = op.a();
    Value B = op.b();
    Value C = op.c();
    Value D = op.getResult();
    MLIRContext *ctx = op->getContext();
    bool allowTF32 = op.allowTF32();

    assert(isSplatLike(C) && "Currently only splat-like C is supported now");

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

    if (!isOuter && isMMA) {
      if (mmaLayout.getVersion() == 1)
        return convertMMA884(op, adaptor, rewriter);
      if (mmaLayout.getVersion() == 2)
        return convertMMA16816(op, adaptor, rewriter);
      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotOp to LLVM.");
    }

    if (op.getType().cast<RankedTensorType>().getElementType().isF32() &&
        A.getType().cast<RankedTensorType>().getElementType().isF32())
      return convertFMADot(op, adaptor, rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }

private:
  // Convert to mma.m16n8k16
  LogicalResult convertMMA16816(triton::DotOp a, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const;
  /// Convert to mma.m8n8k4
  LogicalResult convertMMA884(triton::DotOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    assert(false && "Not implemented yet.");
    return failure();
  }

  LogicalResult convertFMADot(triton::DotOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    assert(false && "Not implemented yet.");
    return failure();
  }
};

struct DotOpConversionHelper {
  using TensorCoreType = DotOpConversion::TensorCoreType;

  MmaEncodingAttr mmaLayout;
  MLIRContext *ctx{};

  explicit DotOpConversionHelper(MmaEncodingAttr mmaLayout)
      : mmaLayout(mmaLayout) {
    ctx = mmaLayout.getContext();
  }

  // Load SplatLike C which contains a constVal. It simply returns 4 fp32
  // constVal.
  SmallVector<Value> loadSplatLikeC(Value C, Location loc,
                                    ConversionPatternRewriter &rewriter) const {
    assert(isSplatLike(C));

    int numRes = getMmaInstrShape()[0] * getMmaInstrShape()[1] / 32;
    if (auto constv = llvm::dyn_cast<arith::ConstantOp>(C.getDefiningOp())) {
      if (auto attr = constv.getValue().dyn_cast<SplatElementsAttr>()) {
        Type elemType = attr.getElementType();
        if (elemType.isInteger(32)) {
          int v = attr.getSplatValue<int>();
          return SmallVector<Value>(numRes, i32_val(v));
        } else if (elemType.isInteger(8)) {
          int v = attr.getSplatValue<int8_t>();
          auto newv = rewriter.create<arith::ConstantOp>(
              loc, elemType, IntegerAttr::get(elemType, v));
          return SmallVector<Value>(numRes, newv);
        } else if (elemType.isF32()) {
          int v = attr.getSplatValue<float>();
          auto newv = rewriter.create<arith::ConstantOp>(
              loc, elemType, FloatAttr::get(elemType, v));
          return SmallVector<Value>(numRes, newv);
        }
      }
    }

    assert(false && "Not supported type.");
    return {};
  }

  void deduceMmaType(DotOp op) const { mmaType = getMmaType(op); }
  void deduceMmaType(Type operandTy) const {
    mmaType = getTensorCoreTypeFromOperand(operandTy);
  }

  Type getShemPtrTy() const {
    switch (mmaType) {
    case TensorCoreType::FP32_FP16_FP16_FP32:
      return ptr_ty(type::f16Ty(ctx), 3);
    case TensorCoreType::FP32_BF16_BF16_FP32:
      return ptr_ty(type::bf16Ty(ctx), 3);
    case TensorCoreType::FP32_TF32_TF32_FP32:
      return ptr_ty(type::f32Ty(ctx), 3);
    case TensorCoreType::INT32_INT8_INT8_INT32:
      return ptr_ty(type::i8Ty(ctx), 3);
    default:
      llvm::report_fatal_error("mma16816 data type not supported");
    }
    return Type{};
  }

  // The type of a matrix that loaded by either a ldmatrix or composed lds.
  Type getMatType() const {
    Type fp32Ty = type::f32Ty(ctx);
    Type fp16x2Ty = vec_ty(type::f16Ty(ctx), 2);
    Type bf16x2Ty = vec_ty(type::bf16Ty(ctx), 2);
    // floating point types
    Type fp16x2Pack4Ty =
        LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, fp16x2Ty));
    Type bf16x2Pack4Ty =
        LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, bf16x2Ty));
    Type fp32Pack4Ty =
        LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, fp32Ty));
    // integer types
    Type i8x4Ty = vec_ty(type::i8Ty(ctx), 4);
    Type i8x4Pack4Ty =
        LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, i8x4Ty));
    Type i32Pack4Ty = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(4, type::i32Ty(ctx)));

    switch (mmaType) {
    case TensorCoreType::FP32_FP16_FP16_FP32:
      return fp16x2Pack4Ty;
    case TensorCoreType::FP32_BF16_BF16_FP32:
      return bf16x2Pack4Ty;
    case TensorCoreType::FP32_TF32_TF32_FP32:
      return fp32Pack4Ty;
    case TensorCoreType::INT32_INT8_INT8_INT32:
      return i8x4Pack4Ty;
    default:
      llvm::report_fatal_error("Unsupported mma type found");
    }

    return Type{};
  }

  Type getLoadElemTy() {
    switch (mmaType) {
    case TensorCoreType::FP32_FP16_FP16_FP32:
      return vec_ty(type::f16Ty(ctx), 2);
    case TensorCoreType::FP32_BF16_BF16_FP32:
      return vec_ty(type::bf16Ty(ctx), 2);
    case TensorCoreType::FP32_TF32_TF32_FP32:
      return type::f32Ty(ctx);
    case TensorCoreType::INT32_INT8_INT8_INT32:
      return type::i32Ty(ctx);
    default:
      llvm::report_fatal_error("Unsupported mma type found");
    }

    return Type{};
  }

  Type getMmaRetType() const {
    Type fp32Ty = type::f32Ty(ctx);
    Type i32Ty = type::i32Ty(ctx);
    Type fp32x4Ty =
        LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, fp32Ty));
    Type i32x4Ty =
        LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, i32Ty));
    switch (mmaType) {
    case TensorCoreType::FP32_FP16_FP16_FP32:
      return fp32x4Ty;
    case TensorCoreType::FP32_BF16_BF16_FP32:
      return fp32x4Ty;
    case TensorCoreType::FP32_TF32_TF32_FP32:
      return fp32x4Ty;
    case TensorCoreType::INT32_INT8_INT8_INT32:
      return i32x4Ty;
    default:
      llvm::report_fatal_error("Unsupported mma type found");
    }

    return Type{};
  }

  ArrayRef<int> getMmaInstrShape() const {
    assert(mmaType != TensorCoreType::NOT_APPLICABLE &&
           "Unknown mma type found.");
    return mmaInstrShape.at(mmaType);
  }

  ArrayRef<int> getMmaMatShape() const {
    assert(mmaType != TensorCoreType::NOT_APPLICABLE &&
           "Unknown mma type found.");
    return mmaMatShape.at(mmaType);
  }

  // Deduce the TensorCoreType from either $a or $b's type. This method is not
  // safe, but we cannot get the DotOp in some getmaMatShape usage case.
  TensorCoreType getTensorCoreTypeFromOperand(Type operandTy) const {
    auto tensorTy = operandTy.cast<RankedTensorType>();
    auto elemTy = tensorTy.getElementType();
    if (elemTy.isF16())
      return TensorCoreType::FP32_FP16_FP16_FP32;
    if (elemTy.isF32())
      return TensorCoreType::FP32_TF32_TF32_FP32;
    if (elemTy.isBF16())
      return TensorCoreType::FP32_BF16_BF16_FP32;
    if (elemTy.isInteger(8))
      return TensorCoreType::INT32_INT8_INT8_INT32;
    return TensorCoreType::NOT_APPLICABLE;
  }

  int getVec() const {
    assert(mmaType != TensorCoreType::NOT_APPLICABLE &&
           "Unknown mma type found.");
    return mmaInstrVec.at(mmaType);
  }

  StringRef getMmaInstr() const {
    assert(mmaType != TensorCoreType::NOT_APPLICABLE &&
           "Unknown mma type found.");
    return mmaInstrPtx.at(mmaType);
  }

  static TensorCoreType getMmaType(triton::DotOp op) {
    Value A = op.a();
    Value B = op.b();
    auto aTy = A.getType().cast<RankedTensorType>();
    auto bTy = B.getType().cast<RankedTensorType>();
    // d = a*b + c
    auto dTy = op.d().getType().cast<RankedTensorType>();
    auto mmaLayout = dTy.getEncoding().cast<MmaEncodingAttr>();

    if (dTy.getElementType().isF32()) {
      if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
        return TensorCoreType::FP32_FP16_FP16_FP32;
      if (aTy.getElementType().isBF16() && bTy.getElementType().isBF16())
        return TensorCoreType::FP32_BF16_BF16_FP32;
      if (aTy.getElementType().isF32() && bTy.getElementType().isF32() &&
          op.allowTF32())
        return TensorCoreType::FP32_TF32_TF32_FP32;
    } else if (dTy.getElementType().isInteger(32)) {
      if (aTy.getElementType().isInteger(8) &&
          bTy.getElementType().isInteger(8))
        return TensorCoreType::INT32_INT8_INT8_INT32;
    }

    return TensorCoreType::NOT_APPLICABLE;
  }

private:
  mutable TensorCoreType mmaType{TensorCoreType::NOT_APPLICABLE};

  // Used on nvidia GPUs mma layout .version == 2
  // Refer to
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-storage
  // for more details.
  inline static const std::map<TensorCoreType, llvm::SmallVector<int>>
      mmaInstrShape = {
          {TensorCoreType::FP32_FP16_FP16_FP32, {16, 8, 16}},
          {TensorCoreType::FP32_BF16_BF16_FP32, {16, 8, 16}},
          {TensorCoreType::FP32_TF32_TF32_FP32, {16, 8, 8}},

          {TensorCoreType::INT32_INT1_INT1_INT32, {16, 8, 256}},
          {TensorCoreType::INT32_INT4_INT4_INT32, {16, 8, 64}},
          {TensorCoreType::INT32_INT8_INT8_INT32, {16, 8, 32}},
  };

  // shape of matrices loaded by ldmatrix (m-n-k, for mxk & kxn matrices)
  // Refer to
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix
  // for more details.
  inline static const std::map<TensorCoreType, llvm::SmallVector<int>>
      mmaMatShape = {
          {TensorCoreType::FP32_FP16_FP16_FP32, {8, 8, 8}},
          {TensorCoreType::FP32_BF16_BF16_FP32, {8, 8, 8}},
          {TensorCoreType::FP32_TF32_TF32_FP32, {8, 8, 4}},

          {TensorCoreType::INT32_INT1_INT1_INT32, {8, 8, 64}},
          {TensorCoreType::INT32_INT4_INT4_INT32, {8, 8, 32}},
          {TensorCoreType::INT32_INT8_INT8_INT32, {8, 8, 16}},
  };

  // Supported mma instruction in PTX.
  // Refer to
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma
  // for more details.
  inline static const std::map<TensorCoreType, std::string> mmaInstrPtx = {
      {TensorCoreType::FP32_FP16_FP16_FP32,
       "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"},
      {TensorCoreType::FP32_BF16_BF16_FP32,
       "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"},
      {TensorCoreType::FP32_TF32_TF32_FP32,
       "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"},

      {TensorCoreType::INT32_INT1_INT1_INT32,
       "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc"},
      {TensorCoreType::INT32_INT4_INT4_INT32,
       "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32"},
      {TensorCoreType::INT32_INT8_INT8_INT32,
       "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32"},
  };

  // vector length per ldmatrix (16*8/elelment_size_in_bits)
  inline static const std::map<TensorCoreType, uint8_t> mmaInstrVec = {
      {TensorCoreType::FP32_FP16_FP16_FP32, 8},
      {TensorCoreType::FP32_BF16_BF16_FP32, 8},
      {TensorCoreType::FP32_TF32_TF32_FP32, 4},

      {TensorCoreType::INT32_INT1_INT1_INT32, 128},
      {TensorCoreType::INT32_INT4_INT4_INT32, 32},
      {TensorCoreType::INT32_INT8_INT8_INT32, 16},
  };
};

// This class helps to adapt the existing DotOpConversion to the latest
// DotOpOperand layout design. It decouples the exising implementation to two
// parts:
// 1. loading the specific operand matrix(for $a, $b, $c) from smem
// 2. passing the loaded value and perform the mma codegen
struct MMA16816ConversionHelper {
  MmaEncodingAttr mmaLayout;
  ArrayRef<unsigned int> wpt;

  Value thread, lane, warp, warpMN, warpN, warpM;

  DotOpConversionHelper helper;
  ConversionPatternRewriter &rewriter;
  TypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

  MMA16816ConversionHelper(MmaEncodingAttr mmaLayout, Value thread,
                           ConversionPatternRewriter &rewriter,
                           TypeConverter *typeConverter, Location loc)
      : mmaLayout(mmaLayout), helper(mmaLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(mmaLayout.getContext()),
        thread(thread) {
    wpt = mmaLayout.getWarpsPerCTA();

    Value _32 = i32_val(32);
    lane = urem(thread, _32);
    warp = udiv(thread, _32);
    warpMN = udiv(warp, i32_val(wpt[0]));
    warpM = urem(warp, i32_val(wpt[0]));
    warpN = urem(warpMN, i32_val(wpt[1]));
  }

  // Get the mmaInstrShape from either $a or $b.
  std::tuple<int, int, int> getMmaInstrShape(Type operand) const {
    helper.deduceMmaType(operand);
    auto mmaInstrShape = helper.getMmaInstrShape();
    int mmaInstrM = mmaInstrShape[0];
    int mmaInstrN = mmaInstrShape[1];
    int mmaInstrK = mmaInstrShape[2];
    return std::make_tuple(mmaInstrM, mmaInstrN, mmaInstrK);
  }

  std::tuple<int, int, int> getMmaMatShape(Type operand) const {
    helper.deduceMmaType(operand);
    auto matShape = helper.getMmaMatShape();
    int matShapeM = matShape[0];
    int matShapeN = matShape[1];
    int matShapeK = matShape[2];
    return std::make_tuple(matShapeM, matShapeN, matShapeK);
  }

  // \param operand is either $a or $b's type.
  inline int getNumRepM(Type operand, int M) const {
    auto [mmaInstrM, mmaInstrN, mmaInstrK] = getMmaInstrShape(operand);
    return std::max<int>(M / (wpt[0] * mmaInstrM), 1);
  }

  // \param operand is either $a or $b's type.
  inline int getNumRepN(Type operand, int N) const {
    auto [mmaInstrM, mmaInstrN, mmaInstrK] = getMmaInstrShape(operand);
    return std::max<int>(N / (wpt[1] * mmaInstrN), 1);
  }

  // \param operand is either $a or $b's type.
  inline int getNumRepK(Type operand, int K) const {
    auto [mmaInstrM, mmaInstrN, mmaInstrK] = getMmaInstrShape(operand);
    return std::max<int>(K / mmaInstrK, 1);
  }

  // Loading $a from smem to registers, returns a LLVM::Struct.
  Value loadA(Value tensor, Value llTensor) const {
    auto aTensorTy = tensor.getType().cast<RankedTensorType>();
    auto shape = aTensorTy.getShape();

    ValueTable ha;
    std::function<void(int, int)> loadFn;
    auto [matShapeM, matShapeN, matShapeK] = getMmaMatShape(aTensorTy);
    auto [mmaInstrM, mmaInstrN, mmaInstrK] = getMmaInstrShape(aTensorTy);
    int numRepM = getNumRepM(aTensorTy, shape[0]);
    int numRepK = getNumRepK(aTensorTy, shape[1]);

    if (aTensorTy.getEncoding().isa<SharedEncodingAttr>()) {
      // load from smem
      loadFn = getLoadMatrixFn(
          tensor, llTensor, mmaLayout, mmaLayout.getWarpsPerCTA()[0] /*wpt*/,
          1 /*kOrder*/, {mmaInstrM, mmaInstrK} /*instrShpae*/,
          {matShapeM, matShapeK} /*matShape*/, warpM /*warpId*/, ha /*vals*/);
    } else if (aTensorTy.getEncoding().isa<BlockedEncodingAttr>()) {
      // load from registers, used in gemm fuse
      // TODO(Superjomn) Port the logic.
      assert(false && "Loading A from register is not supported yet.");
    } else {
      assert(false && "A's layout is not supported.");
    }

    // step1. Perform loading.
    for (unsigned m = 0; m < numRepM; ++m)
      for (unsigned k = 0; k < numRepK; ++k)
        loadFn(2 * m, 2 * k);

    // step2. Format the values to LLVM::Struct to passing to mma codegen.
    Value result = composeValuesToDotOperandLayoutStruct(ha, numRepM, numRepK);

    // TODO[Superjomn]: Replace the convert_layout op with the result once the
    // DotOperandEncodingAttr is ready.
    return result;
  }

  // Loading $b from smem to registers, returns a LLVM::Struct.
  Value loadB(Value tensor, Value llTensor) {
    ValueTable hb;
    auto tensorTy = tensor.getType().cast<RankedTensorType>();
    auto shape = tensorTy.getShape();
    auto [matShapeM, matShapeN, matShapeK] = getMmaMatShape(tensorTy);
    auto [mmaInstrM, mmaInstrN, mmaInstrK] = getMmaInstrShape(tensorTy);
    int numRepK = getNumRepK(tensorTy, shape[0]);
    int numRepN = getNumRepN(tensorTy, shape[1]);

    auto loadFn = getLoadMatrixFn(
        tensor, llTensor, mmaLayout, mmaLayout.getWarpsPerCTA()[1] /*wpt*/,
        0 /*kOrder*/, {mmaInstrK, mmaInstrN} /*instrShpae*/,
        {matShapeK, matShapeN} /*matShape*/, warpN /*warpId*/, hb /*vals*/);

    for (unsigned n = 0; n < std::max(numRepN / 2, 1); ++n) {
      for (unsigned k = 0; k < numRepK; ++k)
        loadFn(2 * n, 2 * k);
    }

    Value result = composeValuesToDotOperandLayoutStruct(
        hb, std::max(numRepN / 2, 1), numRepK);
    return result;
  }

  // Loading $c from smem(?) to registers, returns a Value.
  // NOTE Only SplatLike tensor is supported now.
  Value loadC(Value tensor) const {
    // Currently, we only support a SplatLike C. For the other cases, e.g., C in
    // shared layout or blocked layout, we will support them by expanding
    // convert_layout.
    auto hc = helper.loadSplatLikeC(tensor, loc, rewriter);
    assert(hc.size() == 4UL && "Only splat-like C is supported now");
    return hc[0];
  }

  // Conduct the Dot conversion.
  // \param a, \param b, \param c and \param d are DotOp operands.
  // \param loadedA, \param loadedB, \param loadedC, all of them are result of
  // loading.
  LogicalResult convertDot(Value a, Value b, Value c, Value d, Value loadedA,
                           Value loadedB, Value loadedC, DotOp op,
                           DotOpAdaptor adaptor) const {
    helper.deduceMmaType(op);

    auto aTensorTy = a.getType().cast<RankedTensorType>();
    auto bTensorTy = b.getType().cast<RankedTensorType>();
    auto cTensorTy = c.getType().cast<RankedTensorType>();
    auto dTensorTy = d.getType().cast<RankedTensorType>();

    auto aShape = aTensorTy.getShape();
    auto dShape = dTensorTy.getShape();

    int NK = aShape[1];
    // shape / shape_per_cta
    auto [matShapeM, matShapeN, matShapeK] = getMmaMatShape(aTensorTy);
    auto [mmaInstrM, mmaInstrN, mmaInstrK] = getMmaInstrShape(aTensorTy);
    int numRepM = getNumRepM(aTensorTy, dShape[0]);
    int numRepN = getNumRepN(aTensorTy, dShape[1]);
    int numRepK = getNumRepK(aTensorTy, aShape[1]);

    ValueTable ha =
        getValuesFromDotOperandLayoutStruct(loadedA, numRepM, numRepK);
    ValueTable hb = getValuesFromDotOperandLayoutStruct(
        loadedB, std::max(numRepN / 2, 1), numRepK);

    const int fcSize = 4 * numRepM * numRepN;
    SmallVector<Value> fc(fcSize, loadedC);

    auto callMma = [&](unsigned m, unsigned n, unsigned k) {
      unsigned colsPerThread = numRepN * 2;
      PTXBuilder builder;
      auto &mma = *builder.create(helper.getMmaInstr().str());
      auto retArgs = builder.newListOperand(4, "=r");
      auto aArgs = builder.newListOperand({
          {ha[{m, k}], "r"},
          {ha[{m + 1, k}], "r"},
          {ha[{m, k + 1}], "r"},
          {ha[{m + 1, k + 1}], "r"},
      });
      auto bArgs =
          builder.newListOperand({{hb[{n, k}], "r"}, {hb[{n, k + 1}], "r"}});
      auto cArgs = builder.newListOperand();
      for (int i = 0; i < 4; ++i) {
        cArgs->listAppend(builder.newOperand(fc[m * colsPerThread + 4 * n + i],
                                             std::to_string(i)));
        // reuse the output registers
      }
      mma(retArgs, aArgs, bArgs, cArgs);
      Value mmaOut = builder.launch(rewriter, loc, helper.getMmaRetType());

      auto getIntAttr = [&](int v) {
        return ArrayAttr::get(ctx, {IntegerAttr::get(i32_ty, v)});
      };

      for (int i = 0; i < 4; i++)
        fc[m * colsPerThread + 4 * n + i] =
            extract_val(type::f32Ty(ctx), mmaOut, getIntAttr(i));
    };

    for (unsigned k = 0; k < numRepK; ++k)
      for (unsigned m = 0; m < numRepM; ++m)
        for (unsigned n = 0; n < numRepN; ++n)
          callMma(2 * m, n, 2 * k);

    // replace with new packed result
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(fc.size(), type::f32Ty(ctx)));
    Value res = getStructFromElements(loc, fc, rewriter, structTy);
    rewriter.replaceOp(op, res);

    return success();
  }

private:
  std::function<void(int, int)>
  getLoadMatrixFn(Value tensor, Value llTensor, MmaEncodingAttr mmaLayout,
                  int wpt, int kOrder, ArrayRef<int> instrShape,
                  ArrayRef<int> matShape, Value warpId,
                  ValueTable &vals) const {
    auto tensorTy = tensor.getType().cast<RankedTensorType>();
    // We assumes that the input operand of Dot should be from shared layout.
    // TODO(Superjomn) Consider other layouts if needed later.
    auto sharedLayout = tensorTy.getEncoding().cast<SharedEncodingAttr>();
    const int perPhase = sharedLayout.getPerPhase();
    const int maxPhase = sharedLayout.getMaxPhase();
    const int elemBytes = tensorTy.getElementTypeBitWidth() / 8;
    auto order = sharedLayout.getOrder();

    bool needTrans = kOrder != order[0];

    // the original register_lds2, but discard the prefetch logic.
    auto ld2 = [](ValueTable &vals, int mn, int k, Value val) {
      vals[{mn, k}] = val;
    };

    // (a, b) is the coordinate.
    auto load = [=, &vals, &ld2](int a, int b) {
      MMA16816SmemLoader loader(wpt, sharedLayout.getOrder(), kOrder,
                                tensorTy.getShape() /*tileShape*/, instrShape,
                                matShape, perPhase, maxPhase, elemBytes,
                                rewriter, typeConverter, loc);
      SmallVector<Value> offs = loader.computeOffsets(warpId, lane);

      const int numPtrs = loader.getNumPtr();

      SmallVector<Value> ptrs(numPtrs);

      Type smemPtrTy = helper.getShemPtrTy();
      for (int i = 0; i < numPtrs; ++i) {
        ptrs[i] =
            bitcast(smemPtrTy, gep(smemPtrTy, llTensor, ValueRange({offs[i]})));
      }

      auto [ha0, ha1, ha2, ha3] = loader.loadX4(
          (kOrder == 1) ? a : b /*mat0*/, (kOrder == 1) ? b : a /*mat1*/, offs,
          ptrs, helper.getMatType(), helper.getShemPtrTy());
      if (!needTrans) {
        ld2(vals, a, b, ha0);
        ld2(vals, a + 1, b, ha1);
        ld2(vals, a, b + 1, ha2);
        ld2(vals, a + 1, b + 1, ha3);
      } else {
        ld2(vals, a, b, ha0);
        ld2(vals, a + 1, b, ha2);
        ld2(vals, a, b + 1, ha1);
        ld2(vals, a + 1, b + 1, ha3);
      }
    };

    return load;
  }

  // Compose a map of Values to a LLVM::Struct.
  // The layout is a list of Value with coordinate of (i,j), the order is as
  // the follows:
  // [
  //  (0,0), (0,1), (1,0), (1,1), # i=0, j=0
  //  (0,2), (0,3), (1,2), (1,3), # i=0, j=1
  //  (0,4), (0,5), (1,4), (1,5), # i=0, j=2
  //  ...
  //  (2,0), (2,1), (3,0), (3,1), # i=1, j=0
  //  (2,2), (2,3), (3,2), (3,3), # i=1, j=1
  //  (2,4), (2,5), (2,4), (2,5), # i=1, j=2
  //  ...
  // ]
  // i \in [0, n0) and j \in [0, n1)
  // There should be \param n0 * \param n1 elements in the output Struct.
  Value composeValuesToDotOperandLayoutStruct(const ValueTable &vals, int n0,
                                              int n1) const {
    std::vector<Value> elems;
    for (unsigned m = 0; m < n0; ++m)
      for (unsigned k = 0; k < n1; ++k) {
        elems.push_back(vals.at({2 * m, 2 * k}));
        elems.push_back(vals.at({2 * m, 2 * k + 1}));
        elems.push_back(vals.at({2 * m + 1, 2 * k}));
        elems.push_back(vals.at({2 * m + 1, 2 * k + 1}));
      }

    assert(!elems.empty());

    Type fp16Ty = type::f16Ty(ctx);
    Type fp16x2Ty = vec_ty(fp16Ty, 2);
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(elems.size(), fp16x2Ty));
    auto result = getStructFromElements(loc, elems, rewriter, structTy);
    return result;
  }

  ValueTable getValuesFromDotOperandLayoutStruct(Value value, int n0,
                                                 int n1) const {
    auto elems = ConvertTritonGPUOpToLLVMPatternBase::getElementsFromStruct(
        loc, value, rewriter);

    int offset{};
    ValueTable vals;
    for (int i = 0; i < n0; i++) {
      for (int j = 0; j < n1; j++) {
        vals[{2 * i, 2 * j}] = elems[offset++];
        vals[{2 * i, 2 * j + 1}] = elems[offset++];
        vals[{2 * i + 1, 2 * j}] = elems[offset++];
        vals[{2 * i + 1, 2 * j + 1}] = elems[offset++];
      }
    }
    return vals;
  }
};

LogicalResult ConvertLayoutOpConversion::lowerSharedToDotOperand(
    triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value src = op.src();
  Value dst = op.result();
  auto srcTensorTy = src.getType().cast<RankedTensorType>();
  auto dstTensorTy = dst.getType().cast<RankedTensorType>();

  auto sharedLayout = srcTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto dotOperandLayout =
      dstTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
  MmaEncodingAttr mmaLayout =
      dotOperandLayout.getParent().dyn_cast_or_null<MmaEncodingAttr>();
  assert(mmaLayout);

  MMA16816ConversionHelper mmaHelper(mmaLayout, getThreadId(rewriter, loc),
                                     rewriter, getTypeConverter(), op.getLoc());

  Value res;
  if (dotOperandLayout.getOpIdx() == 0) {
    // operand $a
    res = mmaHelper.loadA(src, adaptor.src());
  } else if (dotOperandLayout.getOpIdx() == 1) {
    // operand $b
    res = mmaHelper.loadB(src, adaptor.src());
  } else if (dotOperandLayout.getOpIdx() == 2) {
    // operand $c
    res = mmaHelper.loadC(src);
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
  MMA16816ConversionHelper mmaHelper(mmaLayout, getThreadId(rewriter, loc),
                                     rewriter, getTypeConverter(), loc);

  Value A = op.a();
  Value B = op.b();
  Value C = op.c();
  auto ATensorTy = A.getType().cast<RankedTensorType>();
  auto BTensorTy = B.getType().cast<RankedTensorType>();

  Value loadedA, loadedB, loadedC;
  // We support two kinds of operand layouts: 1. both $a, $b are dot_operand
  // layout, 2. both of them are shared layout.
  if (ATensorTy.getEncoding().isa<DotOperandEncodingAttr>()) {
    assert(BTensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
           "Both $a and %b should be DotOperand layout.");
    loadedA = adaptor.a();
    loadedB = adaptor.b();
  } else {
    loadedA = mmaHelper.loadA(op.a(), adaptor.a());
    loadedB = mmaHelper.loadB(op.b(), adaptor.b());
  }

  // TODO[Superjomn]: Process C as a mma layout.
  // Currently, C is simply treated as a Splat Op, and the data layout is not
  // mattered.
  loadedC = mmaHelper.loadC(op.c());

  return mmaHelper.convertDot(A, B, C, op.d(), loadedA, loadedB, loadedC, op,
                              adaptor);
}

/// ====================== mma codegen end ============================

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
  }

  Type convertTritonPointerType(triton::PointerType type) {
    return LLVM::LLVMPointerType::get(type.getPointeeType(),
                                      type.getAddressSpace());
  }

  llvm::Optional<Type> convertTritonTensorType(RankedTensorType type) {
    Attribute layout = type.getEncoding();
    if (layout &&
        (layout.isa<BlockedEncodingAttr>() || layout.isa<SliceEncodingAttr>() ||
         layout.isa<MmaEncodingAttr>())) {
      unsigned numElementsPerThread =
          getElemsPerThread(layout, type.getShape());
      SmallVector<Type, 4> types(numElementsPerThread,
                                 convertType(type.getElementType()));
      return LLVM::LLVMStructType::getLiteral(&getContext(), types);
    } else if (auto shared_layout =
                   layout.dyn_cast_or_null<SharedEncodingAttr>()) {
      return LLVM::LLVMPointerType::get(convertType(type.getElementType()), 3);
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
    auto &asyncWaitOp = *ptxBuilder.create<PTXCpAsyncWaitGroupInstr>();
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    asyncWaitOp(ptxBuilder.newConstantOperand(num));

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto ret = ptxBuilder.launch(rewriter, loc, voidTy);

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
    auto resElemTy = resTy.getElementType();
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
    auto srcElems = getLLVMElems(src, llSrc, srcBlockedLayout, rewriter, loc);

    // %dst
    auto axis = op->getAttrOfType<IntegerAttr>("axis").getInt();
    assert(axis == 0 && "insert_slice_async: Only axis=0 is supported for now");
    auto dstBase = createIndexAttrConstant(rewriter, loc,
                                           getTypeConverter()->getIndexType(),
                                           product<int64_t>(resTy.getShape()));
    Value offset = mul(llIndex, dstBase);
    auto dstPtrTy = LLVM::LLVMPointerType::get(
        getTypeConverter()->convertType(resTy.getElementType()), 3);
    Value dstPtrBase = gep(dstPtrTy, llDst, offset);

    // %mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = getLLVMElems(mask, llMask, srcBlockedLayout, rewriter, loc);
      assert(srcElems.size() == maskElems.size());
    }

    // %other
    SmallVector<Value> otherElems;
    if (llOther) {
      // TODO(Keren): support "other" tensor.
      // It's not necessary for now because the pipeline pass will skip
      // generating insert_slice_async if the load op has any "other" tensor.
      assert(false && "insert_slice_async: Other value not supported yet");
      otherElems =
          getLLVMElems(other, llOther, srcBlockedLayout, rewriter, loc);
      assert(srcElems.size() == otherElems.size());
    }

    unsigned inVec = getVectorizeSize(src, srcBlockedLayout);
    unsigned outVec = resSharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    unsigned numElems = getElemsPerThread(srcBlockedLayout, srcShape);
    unsigned perPhase = resSharedLayout.getPerPhase();
    unsigned maxPhase = resSharedLayout.getMaxPhase();
    auto sizePerThread = srcBlockedLayout.getSizePerThread();
    auto threadsPerWarp = srcBlockedLayout.getThreadsPerWarp();
    auto warpsPerCTA = srcBlockedLayout.getWarpsPerCTA();
    auto threadsPerCTA = getThreadsPerCTA(srcBlockedLayout);

    auto inOrder = srcBlockedLayout.getOrder();
    auto outOrder = resSharedLayout.getOrder();
    // If perPhase * maxPhase > threadsPerCTA, we need to swizzle over
    // elements across phases. If perPhase * maxPhase == threadsPerCTA,
    // swizzle is not allowd
    auto numSwizzleRows = std::max<unsigned>(
        (perPhase * maxPhase) / threadsPerCTA[inOrder[1]], 1);
    // A sharedLayout encoding has a "vec" parameter.
    // On the column dimension, if inVec > outVec, it means we have to divide
    // single vector read into multiple ones
    auto numVecCols = std::max<unsigned>(inVec / outVec, 1);

    auto srcIndices = emitIndices(loc, rewriter, srcBlockedLayout, srcShape);
    // <<tileVecIdxRow, tileVecIdxCol>, TileOffset>
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
      auto baseOffset = (baseOffsetRow * srcShape[inOrder[0]] + baseOffsetCol);
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
        //     | [1 2] [3 4]  ... [15 16] |
        //     | [3 4] [5 6]  ... [1 2]   |
        // Example2:
        // outVec = 4, inVec = 2, minVec = 2
        //     | [1 2 3 4] [5 6 7 8] ... [13 14 15 16] |
        //     | [5 6 7 8] [9 10 11 12] ... [1 2 3 4]  |
        auto srcIdx = srcIndices[tileVecIdxRow * sizePerThread[inOrder[0]]];
        Value phase = urem(udiv(srcIdx[inOrder[1]], i32_val(perPhase)),
                           i32_val(maxPhase));
        Value rowOffset =
            mul(srcIdx[inOrder[1]], i32_val(srcShape[inOrder[0]]));
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

      // XXX(Keren): Tune CG and CA here.
      CacheModifier srcCacheModifier =
          bitWidth == 128 ? CacheModifier::CG : CacheModifier::CA;
      assert(bitWidth == 128 || bitWidth == 64 || bitWidth == 32);

      for (int wordIdx = 0; wordIdx < numWords; ++wordIdx) {
        PTXBuilder ptxBuilder;
        auto &copyAsyncOp = *ptxBuilder.create<PTXCpAsyncLoadInstr>(
            srcCacheModifier, op.evict());

        auto tileOffset = tileOffsetMap[{tileVecIdxRow, tileVecIdxCol}];
        auto *dstOperand =
            ptxBuilder.newAddrOperand(tileOffset, "r", baseOffset);
        auto *srcOperand = ptxBuilder.newAddrOperand(srcElems[vecIdx], "l");
        auto *copySize = ptxBuilder.newConstantOperand(bitWidth);
        auto *srcSize = copySize;
        if (op.mask()) {
          // We don't use predicate in this case, setting src-size to 0
          // if there's any mask. cp.async will automatically fill the
          // remaining slots with 0 if cp-size > src-size.
          // XXX(Keren): Always assume other = 0 for now.
          auto selectOp = select(maskElems[vecIdx + wordIdx * numWordElems],
                                 i32_val(bitWidth), i32_val(0));
          srcSize = ptxBuilder.newOperand(selectOp, "r");
        }
        copyAsyncOp(dstOperand, srcOperand, copySize, srcSize);
        ptxBuilder.launch(rewriter, loc, LLVM::LLVMVoidType::get(getContext()));
      }
    }

    PTXBuilder ptxBuilder;
    ptxBuilder.create<PTXCpAsyncCommitGroupInstr>()->operator()();
    auto ret =
        ptxBuilder.launch(rewriter, loc, LLVM::LLVMVoidType::get(getContext()));
    rewriter.replaceOp(op, ret);
    return success();
  }
};

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
  patterns.add<BinaryOpConversion<arith::AddIOp, LLVM::AddOp>>(typeConverter,
                                                               benefit);
  patterns.add<BinaryOpConversion<arith::AddFOp, LLVM::FAddOp>>(typeConverter,
                                                                benefit);
  patterns.add<BinaryOpConversion<arith::MulIOp, LLVM::MulOp>>(typeConverter,
                                                               benefit);
  patterns.add<BinaryOpConversion<arith::MulFOp, LLVM::FMulOp>>(typeConverter,
                                                                benefit);

  patterns.add<BinaryOpConversion<arith::AndIOp, LLVM::AndOp>>(typeConverter,
                                                               benefit);
  patterns.add<BinaryOpConversion<arith::OrIOp, LLVM::OrOp>>(typeConverter,
                                                             benefit);

  patterns.add<CmpIOpConversion>(typeConverter, benefit);
  patterns.add<CmpFOpConversion>(typeConverter, benefit);
  patterns.add<BroadcastOpConversion>(typeConverter, benefit);
  patterns.add<ConvertLayoutOpConversion>(typeConverter, allocation, smem,
                                          benefit);
  patterns.add<ExtractSliceOpConversion>(typeConverter, allocation, smem,
                                         benefit);
  patterns.add<GetProgramIdOpConversion>(typeConverter, benefit);
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
}

class ConvertTritonGPUToLLVM
    : public ConvertTritonGPUToLLVMBase<ConvertTritonGPUToLLVM> {
public:
  ConvertTritonGPUToLLVM() = default;

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

    // step 1: Convert FuncOp to LLVMFuncOp via partial conversion
    // step 2: Allocate for shared memories
    // step 3: Convert the rest of ops via partial conversion
    // The reason for a seperation between 1/3 is that, step 2 is out of
    // the scope of Dialect Conversion, thus we need to make sure the smem
    // is not revised during the conversion of step 3.
    RewritePatternSet func_patterns(context);
    func_patterns.add<FuncOpConversion>(typeConverter, numWarps, 1 /*benefit*/);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(func_patterns))))
      return signalPassFailure();

    Allocation allocation(mod);
    auto axisAnalysis = runAxisAnalysis(mod);
    initSharedMemory(allocation.getSharedMemorySize(), typeConverter);

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
};

void ConvertTritonGPUToLLVM::initSharedMemory(
    size_t size, TritonGPUToLLVMTypeConverter &typeConverter) {
  ModuleOp mod = getOperation();
  OpBuilder b(mod.getBodyRegion());
  auto loc = mod.getLoc();
  auto elemTy = typeConverter.convertType(b.getIntegerType(8));
  auto arrayTy = LLVM::LLVMArrayType::get(elemTy, size);
  auto global = b.create<LLVM::GlobalOp>(
      loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::Internal,
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

TritonLLVMConversionTarget::TritonLLVMConversionTarget(
    MLIRContext &ctx, mlir::LLVMTypeConverter &typeConverter)
    : ConversionTarget(ctx), typeConverter(typeConverter) {
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
    : ConversionTarget(ctx), typeConverter(typeConverter) {
  addLegalDialect<LLVM::LLVMDialect>();
  // addLegalDialect<NVVM::NVVMDialect>();
  addIllegalOp<mlir::FuncOp>();
  addLegalOp<mlir::UnrealizedConversionCastOp>();
}

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass() {
  return std::make_unique<::ConvertTritonGPUToLLVM>();
}

} // namespace triton
} // namespace mlir
