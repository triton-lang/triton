#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCE_OP_CONVERSIONS_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCE_OP_CONVERSIONS_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

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

#endif