#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCEOP_TO_LLVM_BASE_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCEOP_TO_LLVM_BASE_H

#include "TargetInfoBase.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

class ReduceOpConversionBase : public ConvertOpToLLVMPattern<ReduceOp> {
public:
  using Base = ConvertOpToLLVMPattern<ReduceOp>;
  using OpAdaptor = Base::OpAdaptor;

  ReduceOpConversionBase(LLVMTypeConverter &typeConverter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit);

  LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

protected:
  SmallVector<Value> combineValues(Location loc,
                                   ConversionPatternRewriter &rewriter,
                                   Region &combineOp, ValueRange lhs,
                                   ValueRange rhs) const;

  // The default is a target-arity tree. Target-specific patterns may override
  // the in-thread schedule.
  virtual SmallVector<Value> reduceInThreadValues(
      Location loc, ConversionPatternRewriter &rewriter, Region &combineOp,
      SmallVector<SmallVector<Value>> values, unsigned arity) const;

private:
  SmallVector<Value> treeReduce(Location loc,
                                ConversionPatternRewriter &rewriter,
                                Region &combineOp,
                                SmallVector<SmallVector<Value>> values,
                                unsigned arity) const;

  class Impl;
  const TargetInfoBase &targetInfo;
};

} // namespace mlir::triton

#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCEOP_TO_LLVM_BASE_H
