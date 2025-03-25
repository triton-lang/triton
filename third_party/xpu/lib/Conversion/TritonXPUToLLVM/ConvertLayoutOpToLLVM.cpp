//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "xpu/lib/Conversion/TritonXPUToLLVM/PatternTritonXPUOpToLLVM.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct XPUConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::ConvertLayoutOp> {
  XPUConvertLayoutOpConversion(LLVMTypeConverter &converter,
                               const xpu::TargetInfo &targetInfo,
                               PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::ConvertLayoutOp>(converter,
                                                             benefit) {}

  bool isaXPUValidLayout(const Attribute &layout) const {
    return mlir::isa<triton::xpu::ClusterLayoutAttr>(layout) ||
           mlir::isa<triton::xpu::ClusterLayoutAttr>(
               mlir::cast<triton::gpu::SliceEncodingAttr>(layout).getParent());
  }

  LogicalResult
  matchAndRewrite(triton::xpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<RankedTensorType>(dst.getType());
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    if (isaXPUValidLayout(srcLayout) && isaXPUValidLayout(dstLayout)) {
      return lowerOperand(op, adaptor, rewriter);
    }
    return failure();
  };

  LogicalResult lowerOperand(triton::xpu::ConvertLayoutOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();

    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    Value ret = packLLElements(loc, typeConverter, vals, rewriter, dstTy);

    rewriter.replaceOp(op, ret);
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<XPUConvertLayoutOpConversion>(typeConverter, targetInfo,
                                             benefit);
}
