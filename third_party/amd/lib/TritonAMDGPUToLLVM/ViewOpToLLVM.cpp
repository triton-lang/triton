#include "ViewOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::AMD::TritonGPUToLLVMTypeConverter;
using ::AMD::ConvertTritonGPUOpToLLVMPatternBase;
using ::AMD::ConvertTritonGPUOpToLLVMPattern;

namespace {
struct SplatOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::SplatOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::SplatOp>::ConvertTritonGPUOpToLLVMPattern;

  // Convert SplatOp or arith::ConstantOp with SplatElementsAttr to a
  // LLVM::StructType value.
  //
  // @elemType: the element type in operand.
  // @resType: the return type of the Splat-like op.
  // @constVal: a LLVM::ConstantOp or other scalar value.
  static Value convertSplatLikeOp(Type elemType, Type resType, Value constVal,
                                  TritonGPUToLLVMTypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) {
    auto tensorTy = resType.cast<RankedTensorType>();
    // Check the converted type for the tensor as depending on the encoding the
    // converter may pick different element types.
    auto srcType = typeConverter->convertType(tensorTy);
    if (auto structTy = dyn_cast<LLVM::LLVMStructType>(srcType))
      srcType = structTy.getBody()[0];
    // If the type sizes don't match we need to pack constants.
    if (srcType.isIntOrFloat() && constVal.getType().getIntOrFloatBitWidth() !=
                                      srcType.getIntOrFloatBitWidth()) {
      unsigned cstBitWidth = constVal.getType().getIntOrFloatBitWidth();
      unsigned srcBitWidth = srcType.getIntOrFloatBitWidth();
      assert(cstBitWidth <= srcBitWidth && srcBitWidth % cstBitWidth == 0);
      unsigned ratio = srcBitWidth / cstBitWidth;
      Type intTy = IntegerType::get(elemType.getContext(), cstBitWidth);
      VectorType vecType = VectorType::get(ratio, intTy);
      Value intCst = bitcast(constVal, intTy);
      Value vec = undef(vecType);
      for (unsigned i = 0; i < ratio; ++i)
        vec = insert_element(vecType, vec, intCst, int_val(32, i));
      constVal = vec;
    }
    auto llSrc = bitcast(constVal, srcType);
    size_t elemsPerThread = getTotalElemsPerThread(tensorTy);
    llvm::SmallVector<Value> elems(elemsPerThread, llSrc);
    return typeConverter->packLLElements(loc, elems, rewriter, resType);
  }

  LogicalResult matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto src = adaptor.getSrc();
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
    if (elemType.isBF16() || type::isFloat(elemType)) {
      val = values.getValues<FloatAttr>()[0];
    } else if (type::isInt(elemType)) {
      val = values.getValues<IntegerAttr>()[0];
    } else {
      llvm::errs() << "ArithConstantSplatOpConversion get unsupported type: "
                   << value.getType() << "\n";
      return failure();
    }

    auto constOp = rewriter.create<LLVM::ConstantOp>(loc, elemType, val);
    auto llStruct = SplatOpConversion::convertSplatLikeOp(
        elemType, op.getType(), constOp, getTypeConverter(), rewriter, loc);
    rewriter.replaceOp(op, llStruct);

    return success();
  }
};

struct CatOpConversion : public ConvertTritonGPUOpToLLVMPattern<CatOp> {
  using OpAdaptor = typename CatOp::Adaptor;

  explicit CatOpConversion(TritonGPUToLLVMTypeConverter &typeConverter,

                           PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<CatOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType().template cast<RankedTensorType>();
    unsigned elems = getTotalElemsPerThread(resultTy);
    Type elemTy =
        this->getTypeConverter()->convertType(resultTy.getElementType());
    SmallVector<Type> types(elems, elemTy);
    // unpack input values
    auto lhsVals = getTypeConverter()->unpackLLElements(
        loc, adaptor.getLhs(), rewriter, op.getOperand(0).getType());
    auto rhsVals = getTypeConverter()->unpackLLElements(
        loc, adaptor.getRhs(), rewriter, op.getOperand(1).getType());
    // concatenate (and potentially reorder) values
    SmallVector<Value> retVals;
    for (Value v : lhsVals)
      retVals.push_back(v);
    for (Value v : rhsVals)
      retVals.push_back(v);
    // pack and replace
    Value ret =
        getTypeConverter()->packLLElements(loc, retVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct InterleaveOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<ExperimentalInterleaveOp> {
  using OpAdaptor = typename ExperimentalInterleaveOp::Adaptor;

  explicit InterleaveOpConversion(TritonGPUToLLVMTypeConverter &typeConverter,
                                  PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<ExperimentalInterleaveOp>(typeConverter,
                                                                  benefit) {}

  LogicalResult
  matchAndRewrite(ExperimentalInterleaveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We rely on the following invariants of this op (which are checked by its
    // verifier):
    //
    // - The op has a blocked encoding.
    // - The last dimension (the one we're interleaving) is also the most minor
    //   dimension.
    // - The input and output encodings are the same, except the output has
    //   twice as many elements per thread in the last dimension.
    //
    // With these invariants, interleaving is trivial: We just return the i'th
    // element from lhs, followed by the i'th elem from rhs.
    Location loc = op->getLoc();
    auto resultTy = op.getType().cast<RankedTensorType>();

    SmallVector<Value> lhsVals = getTypeConverter()->unpackLLElements(
        loc, adaptor.getLhs(), rewriter, op.getLhs().getType());
    SmallVector<Value> rhsVals = getTypeConverter()->unpackLLElements(
        loc, adaptor.getRhs(), rewriter, op.getRhs().getType());
    assert(lhsVals.size() == rhsVals.size());

    SmallVector<Value> interleavedVals;
    for (int i = 0; i < lhsVals.size(); i++) {
      interleavedVals.push_back(lhsVals[i]);
      interleavedVals.push_back(rhsVals[i]);
    }

    Value ret = getTypeConverter()->packLLElements(loc, interleavedVals,
                                                   rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct ReshapeOpConversion : public ConvertTritonGPUOpToLLVMPattern<ReshapeOp> {
  using OpAdaptor = typename ReshapeOp::Adaptor;
  explicit ReshapeOpConversion(TritonGPUToLLVMTypeConverter &typeConverter,

                               PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<ReshapeOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    assert(!triton::gpu::isExpensiveView(op.getSrc().getType(), op.getType()) &&
           "expensive view not supported");
    auto resultTy = op.getType().template cast<RankedTensorType>();
    auto srcTy = op.getSrc().getType().template cast<RankedTensorType>();
    if (!op.getAllowReorder()) {
      // Only support trivial block layouts for now.
      auto mod = op->getParentOfType<ModuleOp>();
      int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
      int threadsPerWarp =
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
      assert(resultTy.getEncoding() == triton::gpu::getDefaultBlockedEncoding(
                                           op.getContext(), resultTy.getShape(),
                                           numWarps, threadsPerWarp, numCTAs) &&
             "ReshapeOp lowering only support block encoding right now.");
      assert(srcTy.getEncoding() == triton::gpu::getDefaultBlockedEncoding(
                                        op.getContext(), srcTy.getShape(),
                                        numWarps, threadsPerWarp, numCTAs) &&
             "ReshapeOp lowering only support block encoding right now.");
    }

    auto vals = this->getTypeConverter()->unpackLLElements(
        loc, adaptor.getSrc(), rewriter, op.getOperand().getType());
    Value ret =
        this->getTypeConverter()->packLLElements(loc, vals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct ExpandDimsOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<ExpandDimsOp> {
  using OpAdaptor = typename ExpandDimsOp::Adaptor;
  explicit ExpandDimsOpConversion(TritonGPUToLLVMTypeConverter &typeConverter,

                                  PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<ExpandDimsOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto srcVals = this->getTypeConverter()->unpackLLElements(
        loc, adaptor.getSrc(), rewriter, op.getOperand().getType());

    auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    auto resultTy = op.getType().template cast<RankedTensorType>();

    assert(srcTy.getEncoding().isa<SliceEncodingAttr>() &&
           "ExpandDimsOp only support SliceEncodingAttr");
    auto srcLayout = srcTy.getEncoding().dyn_cast<SliceEncodingAttr>();
    auto resultLayout = resultTy.getEncoding();

    auto srcOffsets = emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultTy);
    DenseMap<SmallVector<unsigned>, Value, SmallVectorKeyInfo> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }

    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      offset.erase(offset.begin() + srcLayout.getDim());
      resultVals.push_back(srcValues.lookup(offset));
    }
    Value ret = this->getTypeConverter()->packLLElements(loc, resultVals,
                                                         rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct TransOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::TransOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::TransOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto llvmElemTy = getTypeConverter()->convertType(
        op.getType().cast<RankedTensorType>().getElementType());
    auto srcSmemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                      llvmElemTy, rewriter);
    SmallVector<Value> dstStrides = {srcSmemObj.strides[1],
                                     srcSmemObj.strides[0]};
    SmallVector<Value> dstOffsets = {srcSmemObj.offsets[1],
                                     srcSmemObj.offsets[0]};
    auto dstSmemObj = SharedMemoryObject(
        srcSmemObj.base, srcSmemObj.baseElemType, dstStrides, dstOffsets);
    auto retVal = getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};
}

namespace AMD{
void populateViewOpToLLVMPatterns(TritonGPUToLLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns, int numWarps,
                                  ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                  ModuleAllocation &allocation,
                                  PatternBenefit benefit) {
  patterns.add<ReshapeOpConversion>(typeConverter, benefit);
  patterns.add<ExpandDimsOpConversion>(typeConverter, benefit);
  patterns.add<SplatOpConversion>(typeConverter, benefit);
  patterns.add<ArithConstantSplatOpConversion>(typeConverter, benefit);
  patterns.add<CatOpConversion>(typeConverter, benefit);
  patterns.add<InterleaveOpConversion>(typeConverter, benefit);
  patterns.add<TransOpConversion>(typeConverter, benefit);
}
}