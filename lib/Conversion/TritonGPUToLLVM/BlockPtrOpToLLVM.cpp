#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

struct MakeTensorPtrOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeTensorPtrOp> {
  using ConvertOpToLLVMPattern<triton::MakeTensorPtrOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptrType = cast<triton::PointerType>(op.getType());
    auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());

    SmallVector<Value> resultVals;

    // TODO:  offset should be 64 bits
    auto offsets = op.getOffsets();
    auto strides = adaptor.getStrides();
    auto shape = adaptor.getShape();
    for (int i = 0; i < offsets.size(); i++) {
      resultVals.push_back(offsets[i]);
    }
    for (int i = 0; i < shape.size(); i++) {
      resultVals.push_back(shape[i]);
    }
    for (int i = 0; i < strides.size(); i++) {
      resultVals.push_back(strides[i]);
    }

    resultVals.push_back(adaptor.getBase());
    Type structTy = getTypeConverter()->convertType(ptrType);
    Value resultStruct =
        packLLElements(loc, getTypeConverter(), resultVals, rewriter, structTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct AdvanceOpConversion : public ConvertOpToLLVMPattern<triton::AdvanceOp> {
  using ConvertOpToLLVMPattern<triton::AdvanceOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptrType = cast<triton::PointerType>(op.getType());
    auto addedOffsets = adaptor.getOffsets();
    auto blockPointerElems = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto rank = (blockPointerElems.size() - 1) / 3;
    // TODO: maybe offset in op should be 64 bit
    SmallVector<Value> origOffsets(blockPointerElems.begin(),
                                   blockPointerElems.begin() + rank);
    SmallVector<Value> newOffsets;
    for (int i = 0; i < addedOffsets.size(); i++) {
      auto newOffset = add(addedOffsets[i], origOffsets[i]);
      newOffsets.push_back(newOffset);
    }

    std::copy(newOffsets.begin(), newOffsets.end(), blockPointerElems.begin());
    Type structTy = getTypeConverter()->convertType(ptrType);
    Value resultStruct = packLLElements(loc, getTypeConverter(),
                                        blockPointerElems, rewriter, structTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

void mlir::triton::populateBlockPtrOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MakeTensorPtrOpConversion>(typeConverter, benefit);
  patterns.add<AdvanceOpConversion>(typeConverter, benefit);
}
