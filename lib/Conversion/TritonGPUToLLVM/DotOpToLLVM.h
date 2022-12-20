#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_DOT_OP_CONVERSIONS_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_DOT_OP_CONVERSIONS_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

struct DotOpConversion : public ConvertTritonGPUOpToLLVMPattern<triton::DotOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::DotOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  // Convert to mma.m16n8k16
  LogicalResult convertMMA16816(triton::DotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const;
  /// Convert to mma.m8n8k4
  LogicalResult convertMMA884(triton::DotOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const;

  LogicalResult convertFMADot(triton::DotOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const;
};

#endif
