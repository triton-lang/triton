#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.cpp.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
namespace {
struct SplatOpConversion : public ConvertOpToLLVMPattern<triton::SplatOp> {
  using ConvertOpToLLVMPattern<triton::SplatOp>::ConvertOpToLLVMPattern;
  // Convert SplatOp or arith::ConstantOp with SplatElementsAttr to a
  // LLVM::StructType value.
  //
  // @elemType: the element type in operand.
  // @resType: the return type of the Splat-like op.
  // @constVal: a LLVM::ConstantOp or other scalar value.
  static Value convertSplatLikeOp(Type elemType, Type resType, Value constVal,
                                  const LLVMTypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) {
    auto tensorTy = cast<RankedTensorType>(resType);
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
    return packLLElements(loc, typeConverter, elems, rewriter, resType);
  }
  LogicalResult matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto src = adaptor.getSrc();
    auto typeConverter = getTypeConverter();
    auto llStruct = convertSplatLikeOp(src.getType(), op.getType(), src,
                                       typeConverter, rewriter, loc);
    rewriter.replaceOp(op, {llStruct});
    return success();
  }
};
// This pattern helps to convert arith::ConstantOp(with SplatElementsAttr),
// the logic is the same as triton::SplatOp, so the underlying implementation
// is reused.
struct ArithConstantSplatOpConversion
    : public ConvertOpToLLVMPattern<arith::ConstantOp> {
  using ConvertOpToLLVMPattern<arith::ConstantOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = op.getValue();
    if (!mlir::dyn_cast<SplatElementsAttr>(value))
      return failure();
    auto loc = op->getLoc();
    LLVM::ConstantOp arithConstantOp;
    auto values = mlir::dyn_cast<SplatElementsAttr>(op.getValue());
    auto elemType = values.getElementType();
    Attribute val;
    if (type::isFloat(elemType)) {
      val = values.getValues<FloatAttr>()[0];
    } else if (type::isInt(elemType)) {
      val = values.getValues<IntegerAttr>()[0];
    } else {
      llvm::errs() << "ArithConstantSplatOpConversion get unsupported type: "
                   << value.getType() << "\n";
      return failure();
    }
    // Lower FP8 constant to int8 constant since FP8 types are not supported on
    // LLVM IR.
    if (type::isFloat8(elemType))
      elemType = rewriter.getIntegerType(8);
    auto constOp = rewriter.create<LLVM::ConstantOp>(loc, elemType, val);
    auto typeConverter = getTypeConverter();
    auto llStruct = SplatOpConversion::convertSplatLikeOp(
        elemType, op.getType(), constOp, typeConverter, rewriter, loc);
    rewriter.replaceOp(op, llStruct);
    return success();
  }
};
struct CatOpConversion : public ConvertOpToLLVMPattern<CatOp> {
  using OpAdaptor = typename CatOp::Adaptor;
  explicit CatOpConversion(LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit = patternBenefitDefault)
      : ConvertOpToLLVMPattern<CatOp>(typeConverter, benefit) {}
  LogicalResult
  matchAndRewrite(CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = cast<RankedTensorType>(op.getType());
    unsigned elems = getTotalElemsPerThread(resultTy);
    auto typeConverter = getTypeConverter();
    Type elemTy = typeConverter->convertType(resultTy.getElementType());
    SmallVector<Type> types(elems, elemTy);
    // unpack input values
    auto lhsVals = unpackLLElements(loc, adaptor.getLhs(), rewriter);
    auto rhsVals = unpackLLElements(loc, adaptor.getRhs(), rewriter);
    // concatenate (and potentially reorder) values
    SmallVector<Value> retVals;
    for (Value v : lhsVals)
      retVals.push_back(v);
    for (Value v : rhsVals)
      retVals.push_back(v);
    // pack and replace
    Value ret = packLLElements(loc, typeConverter, retVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};
struct JoinOpConversion : public ConvertOpToLLVMPattern<JoinOp> {
  using OpAdaptor = typename JoinOp::Adaptor;
  explicit JoinOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit = patternBenefitDefault)
      : ConvertOpToLLVMPattern<JoinOp>(typeConverter, benefit) {}
  LogicalResult
  matchAndRewrite(JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We rely on the following invariants of this op (which are checked by its
    // verifier):
    //
    // - The op has a blocked encoding.
    // - The last dimension (the one we're joining) is also the most minor
    //   dimension.
    // - The input and output encodings are the same, except the output has
    //   2 elements per thread in the last dim.
    //
    // With these invariants, join is trivial: We just return the i'th element
    // from lhs, followed by the i'th elem from rhs.
    Location loc = op->getLoc();
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto typeConverter = getTypeConverter();
    SmallVector<Value> lhsVals =
        unpackLLElements(loc, adaptor.getLhs(), rewriter);
    SmallVector<Value> rhsVals =
        unpackLLElements(loc, adaptor.getRhs(), rewriter);
    assert(lhsVals.size() == rhsVals.size());
    SmallVector<Value> joinedVals;
    for (int i = 0; i < lhsVals.size(); i++) {
      joinedVals.push_back(lhsVals[i]);
      joinedVals.push_back(rhsVals[i]);
    }
    Value ret =
        packLLElements(loc, typeConverter, joinedVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};
struct SplitOpConversion : public ConvertOpToLLVMPattern<SplitOp> {
  using OpAdaptor = typename SplitOp::Adaptor;
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We rely on the following invariants of this op (which are checked by its
    // verifier):
    //
    // - The op has a blocked encoding.
    // - The last dimension (the one we're spliting) is also the most minor
    //   dimension, and has sizePerThread=2.
    //
    // With these invariants, split is trivial: Every other value goes into
    // return value 0, and every other goes into return value 1.
    Location loc = op->getLoc();
    auto typeConverter = getTypeConverter();
    SmallVector<Value> srcVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    assert(srcVals.size() % 2 == 0);
    SmallVector<Value> outLhsVals;
    SmallVector<Value> outRhsVals;
    for (int i = 0; i < srcVals.size(); i += 2) {
      outLhsVals.push_back(srcVals[i]);
      outRhsVals.push_back(srcVals[i + 1]);
    }
    auto resultTy = cast<RankedTensorType>(op.getResult(0).getType());
    Value retLhs =
        packLLElements(loc, typeConverter, outLhsVals, rewriter, resultTy);
    Value retRhs =
        packLLElements(loc, typeConverter, outRhsVals, rewriter, resultTy);
    rewriter.replaceOp(op, {retLhs, retRhs});
    return success();
  }
};
struct ReshapeOpConversion : public ConvertOpToLLVMPattern<ReshapeOp> {
  using OpAdaptor = typename ReshapeOp::Adaptor;
  explicit ReshapeOpConversion(LLVMTypeConverter &typeConverter,
                               PatternBenefit benefit = patternBenefitDefault)
      : ConvertOpToLLVMPattern<ReshapeOp>(typeConverter, benefit) {}
  LogicalResult
  matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    if (triton::gpu::isExpensiveView(op.getSrc().getType(), op.getType())) {
      return emitOptionalError(loc,
                               "expensive view not supported on reshape op");
    }
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto typeConverter = getTypeConverter();
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    Value ret = packLLElements(loc, typeConverter, vals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};
struct ExpandDimsOpConversion : public ConvertOpToLLVMPattern<ExpandDimsOp> {
  using OpAdaptor = typename ExpandDimsOp::Adaptor;
  explicit ExpandDimsOpConversion(
      LLVMTypeConverter &typeConverter,
      PatternBenefit benefit = patternBenefitDefault)
      : ConvertOpToLLVMPattern<ExpandDimsOp>(typeConverter, benefit) {}
  LogicalResult
  matchAndRewrite(ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto typeConverter = getTypeConverter();
    auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto srcLayout = dyn_cast<SliceEncodingAttr>(srcTy.getEncoding());
    if (!srcLayout) {
      return emitOptionalError(
          loc, "ExpandDimsOp only supports SliceEncodingAttr as its input");
    }
    auto resultLayout = resultTy.getEncoding();
    auto srcOffsets = emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultTy);
    std::map<SmallVector<unsigned>, Value> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }
    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      offset.erase(offset.begin() + srcLayout.getDim());
      resultVals.push_back(srcValues.at(offset));
    }
    Value ret =
        packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};
struct TransOpConversion : public ConvertOpToLLVMPattern<TransOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = cast<TensorOrMemDesc>(op.getType());
    if (auto enc = dyn_cast<SharedEncodingAttr>(resultTy.getEncoding())) {
      auto llvmElemTy =
          getTypeConverter()->convertType(resultTy.getElementType());
      auto srcSmemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                        llvmElemTy, rewriter);
      auto dstSmemObj = SharedMemoryObject(
          srcSmemObj.base, srcSmemObj.baseElemType,
          /*strides=*/applyPermutation(srcSmemObj.strides, op.getOrder()),
          /*offsets=*/applyPermutation(srcSmemObj.offsets, op.getOrder()));
      auto retVal = getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter);
      rewriter.replaceOp(op, retVal);
      return success();
    } else if (auto enc = mlir::dyn_cast<BlockedEncodingAttr>(
                   resultTy.getEncoding())) {
      // If the dst encoding is blocked, then TransOp::inferReturnTypes
      // ensures that:
      //  - the src encoding is also blocked, and
      //  - the translation from src to dst is just a "renaming" of the
      //    registers, i.e. each thread has exactly the same values.
      // Thus the transpose op simply returns the same values it got.
      auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      Value ret = packLLElements(loc, this->getTypeConverter(), vals, rewriter,
                                 resultTy);
      rewriter.replaceOp(op, ret);
      return success();
    }
    return emitOptionalError(loc, "unsupported encoding for TransOp");
  }
};
struct BroadcastOpConversion
    : public ConvertOpToLLVMPattern<triton::BroadcastOp> {
  using ConvertOpToLLVMPattern<triton::BroadcastOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
    Location loc = op->getLoc();
    Value src = adaptor.getSrc();
    Value result = op.getResult();
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto resultTy = cast<RankedTensorType>(result.getType());
    auto srcLayout = srcTy.getEncoding();
    auto resultLayout = resultTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultShape = resultTy.getShape();
    unsigned rank = srcTy.getRank();
    auto typeConverter = getTypeConverter();
    assert(rank == resultTy.getRank());
    auto order = triton::gpu::getOrder(srcLayout);
    auto srcOffsets = emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultTy);
    SmallVector<Value> srcVals = unpackLLElements(loc, src, rewriter);
    std::map<SmallVector<unsigned>, Value> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }
    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      for (size_t j = 0; j < srcShape.size(); j++)
        if (srcShape[j] == 1)
          offset[j] = 0;
      resultVals.push_back(srcValues.at(offset));
    }
    Value resultStruct =
        packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct MemDescSubviewOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::MemDescSubviewOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::MemDescSubviewOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescSubviewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = extract_slice %src[%offsets]
    Location loc = op->getLoc();
    auto srcTy = op.getSrc().getType();
    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());

    // newBase = base + offset
    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);
    SmallVector<Value> opOffsetVals = op.getOffsets();
    size_t destRank = op.getResult().getType().getRank();
    SmallVector<Value> offsetVals;
    SmallVector<Value> strides;
    int rankReduced = srcTy.getRank() - destRank;
    for (int i = rankReduced; i < opOffsetVals.size(); i++) {
      strides.push_back(smemObj.strides[i]);
      offsetVals.push_back(opOffsetVals[i]);
    }
    // Compute the offset based on the original strides of the shared memory
    // object
    auto offset = dot(rewriter, loc, opOffsetVals, smemObj.strides);
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    smemObj =
        SharedMemoryObject(gep(elemPtrTy, llvmElemTy, smemObj.base, offset),
                           llvmElemTy, strides, offsetVals);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};
} // namespace

void mlir::triton::populateViewOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ReshapeOpConversion>(typeConverter, benefit);
  patterns.add<ExpandDimsOpConversion>(typeConverter, benefit);
  patterns.add<SplatOpConversion>(typeConverter, benefit);
  patterns.add<ArithConstantSplatOpConversion>(typeConverter, benefit);
  patterns.add<CatOpConversion>(typeConverter, benefit);
  patterns.add<JoinOpConversion>(typeConverter, benefit);
  patterns.add<SplitOpConversion>(typeConverter, benefit);
  patterns.add<TransOpConversion>(typeConverter, benefit);
  patterns.add<BroadcastOpConversion>(typeConverter, benefit);
  patterns.add<MemDescSubviewOpConversion>(typeConverter, benefit);
}
