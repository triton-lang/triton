//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "xpu/lib/Conversion/TritonXPUToLLVM/PatternTritonXPUOpToLLVM.h"

namespace {
struct XPUExpandDimsOpConversion
    : public ConvertOpToLLVMPattern<triton::ExpandDimsOp> {

  XPUExpandDimsOpConversion(LLVMTypeConverter &converter,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::ExpandDimsOp>(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
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

    Value ret = packLLElements(loc, typeConverter, srcVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct XPUBroadcastOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::BroadcastOp> {
  using ConvertOpToLLVMPattern<
      triton::xpu::BroadcastOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::xpu::BroadcastOp op, OpAdaptor adaptor,
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
    Type resElemTy = getElementTypeOrSelf(resultTy);
    bool isVectorized = isa<VectorType>(resElemTy);

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
    if (isVectorized) {
      // TODO: Enhance VBroadcastOp
      assert(rank == 2 && "BroadcastOp is Vectorized, But Rank != 2");
      auto rowsPerCore = cast<triton::xpu::ClusterLayoutAttr>(resultLayout)
                             .getSizePerCore()[0];
      Type elemTy = getElementTypeOrSelf(resElemTy);
      unsigned vecSize = 512 / elemTy.getIntOrFloatBitWidth();
      if (srcShape[1] == 1) {
        for (size_t i = 0; i < srcOffsets.size(); i++) {
          Value srcVals_0_vector =
              rewriter.create<LLVM::UndefOp>(loc, resElemTy);
          for (size_t elemStart = 0; elemStart < vecSize; ++elemStart) {
            srcVals_0_vector = insert_element(resElemTy, srcVals_0_vector,
                                              srcVals[i], i32_val(elemStart));
          }
          srcValues[srcOffsets[i]] = srcVals_0_vector;
        }
      } else if (srcShape[0] == 1) {
        SmallVector<Value> srcVectorVals;
        for (size_t i = 0; i < resultOffsets.size() / rowsPerCore; i++) {
          Value srcVals_vector = rewriter.create<LLVM::UndefOp>(loc, resElemTy);
          for (size_t elemStart = 0; elemStart < vecSize; ++elemStart) {
            srcVals_vector = insert_element(resElemTy, srcVals_vector,
                                            srcVals[i * vecSize + elemStart],
                                            i32_val(elemStart));
          }
          srcVectorVals.push_back(srcVals_vector);
        }
        for (size_t i = 0; i < resultOffsets.size() / rowsPerCore; i++) {
          srcValues[resultOffsets[i]] = srcVectorVals[i];
        }
      } else {
        llvm_unreachable("Only Support Vectorized BroadcastOp: [Mx1xTy] -> "
                         "[MxNxTy] or [1xNxTy] -> [MxNxTy]");
      }
    } else {
      for (size_t i = 0; i < srcOffsets.size(); i++) {
        srcValues[srcOffsets[i]] = srcVals[i];
      }
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

} // namespace

void mlir::triton::xpu::populateViewOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<XPUExpandDimsOpConversion>(typeConverter, benefit);
  patterns.add<XPUBroadcastOpConversion>(typeConverter, benefit);
}
