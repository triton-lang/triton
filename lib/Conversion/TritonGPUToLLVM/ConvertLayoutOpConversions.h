#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_CONVERT_LAYOUT_OP_CONVERSIONS_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_CONVERT_LAYOUT_OP_CONVERSIONS_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;

struct ConvertLayoutOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::ConvertLayoutOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

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
                                   ConversionPatternRewriter &rewriter);

private:
  SmallVector<Value> getMultiDimOffset(Attribute layout, Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       unsigned elemId, ArrayRef<int64_t> shape,
                                       ArrayRef<unsigned> multiDimCTAInRepId,
                                       ArrayRef<unsigned> shapePerCTA) const;

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

#endif
