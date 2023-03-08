#include "TritonGPUToSPIRV.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::spirv::getElementsFromStruct;
using ::mlir::spirv::getStructFromElements;
using ::mlir::triton::gpu::getElemsPerThread;

struct MakeRangeOpSPIRVConversion
        : public ConvertTritonGPUOpToSPIRVPattern<triton::MakeRangeOp> {

  MakeRangeOpSPIRVConversion(
          SPIRVTypeConverter &converter,
          MLIRContext *context,
          ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
          PatternBenefit benefit)
          : ConvertTritonGPUOpToSPIRVPattern<triton::MakeRangeOp>(
                  converter, context, /*Allocation*/ nullptr, Value{}, indexCacheInfo,
                  benefit) {}

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto rankedTy = op.getResult().getType().dyn_cast<RankedTensorType>();
    auto shape = rankedTy.getShape();
    auto layout = rankedTy.getEncoding();

    auto elemTy = rankedTy.getElementType();
    assert(elemTy.isInteger(32));
    Value start =  rewriter.create<spirv::ConstantOp>(
            loc, elemTy, rewriter.getIntegerAttr( converter->getIndexType(), op.getStart()));
    auto idxs = emitIndices(loc, rewriter, layout, rankedTy);
    unsigned elems = idxs.size();
    SmallVector<Value> retVals(elems);
    // TODO: slice layout has more elements than expected.
    // Unexpected behavior for make range, but generally OK when followed by
    // expand dims + broadcast. very weird behavior otherwise potentially.
    for (const auto multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = add(multiDim.value()[0], start);
    }

    SmallVector<Type> types(elems, elemTy);
    Type structTy = spirv::StructType::get(types);
    Value result = getStructFromElements(loc, retVals, rewriter, structTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};


struct GetProgramIdOpToSPIRVConversion
        : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<
          triton::GetProgramIdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    assert(op.getAxis() < 3);

    Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(
            loc, rewriter.getIndexType(), dims[op.getAxis()]);
    auto *typeConverter = this->template getTypeConverter<SPIRVTypeConverter>();
    auto indexType = typeConverter->getIndexType();

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
            op, TypeRange{indexType}, ValueRange{blockId});
    return success();
  }

  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
};

struct AddPtrOpSPIRVConversion
        : public ConvertTritonGPUOpToSPIRVPattern<triton::AddPtrOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
          triton::AddPtrOp>::ConvertTritonGPUOpToSPIRVPattern;

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
      Type structTy = spirv::StructType::get(types);
      auto ptrs = getElementsFromStruct(loc, adaptor.getPtr(), rewriter);
      auto offsets = getElementsFromStruct(loc, adaptor.getOffset(), rewriter);
      SmallVector<Value> resultVals(elems);
      for (unsigned i = 0; i < elems; ++i) {
        resultVals[i] = gep(elemTy, ptrs[i], offsets[i]);;
      }
      Value view = getStructFromElements(loc, resultVals, rewriter, structTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(resultTy.isa<triton::PointerType>());
      Type llResultTy = getTypeConverter()->convertType(resultTy);
      Value result = gep(llResultTy, adaptor.getPtr(), adaptor.getOffset());
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

struct BroadcastOpSPIRVConversion
        : public ConvertTritonGPUOpToSPIRVPattern<triton::BroadcastOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
          triton::BroadcastOp>::ConvertTritonGPUOpToSPIRVPattern;

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
    auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    auto resultTy = result.getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    auto resultLayout = resultTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultShape = resultTy.getShape();
    unsigned rank = srcTy.getRank();

    assert(rank == resultTy.getRank());
    auto order = triton::gpu::getOrder(srcLayout);
    auto srcOffsets = emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultTy);
    SmallVector<Value> srcVals = getElementsFromStruct(loc, src, rewriter);
    if (auto srcMma = srcLayout.dyn_cast<MmaEncodingAttr>()) {
      assert(0 && "mma not supported");
    }

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

    auto spirvStructTy = getTypeConverter()->convertType(resultTy);
    Value resultStruct =
            getStructFromElements(loc, resultVals, rewriter, spirvStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

void populateTritonGPUToSPIRVPatterns(
        SPIRVTypeConverter &typeConverter, MLIRContext *context,
        RewritePatternSet &patterns, int numWarps,
        AxisInfoAnalysis &axisInfoAnalysis,
        const Allocation *allocation, Value smem,
        ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
        PatternBenefit benefit) {
  patterns.add<AddPtrOpSPIRVConversion>(typeConverter, context, benefit);
//  patterns.add<AllocTensorOpConversion>(typeConverter, allocation, smem,
//                                        benefit);
//  patterns.add<AsyncWaitOpConversion>(typeConverter, benefit);
  patterns.add<BroadcastOpSPIRVConversion>(typeConverter, context, benefit);
//
//  patterns.add<ExtractSliceOpConversion>(typeConverter, allocation, smem,
//                                         benefit);
  patterns.add<GetProgramIdOpToSPIRVConversion>(typeConverter, context, benefit);
//  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<MakeRangeOpSPIRVConversion>(typeConverter, context, indexCacheInfo, benefit);
//  patterns.add<ReturnOpConversion>(typeConverter, benefit);
//  patterns.add<PrintfOpConversion>(typeConverter, benefit);
}