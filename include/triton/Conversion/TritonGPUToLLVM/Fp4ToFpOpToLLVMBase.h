#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_FP4_TO_FP_OP_TO_LLVM_BASE_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_FP4_TO_FP_OP_TO_LLVM_BASE_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <array>

namespace mlir::triton::gpu {

class Fp4ToFpOpConversionBase : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
public:
  Fp4ToFpOpConversionBase(LLVMTypeConverter &typeConverter,
                          PatternBenefit benefit);

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;

protected:
  /// Backend-specific implementation that unpacks a 4 element packed vector of
  /// fp4x2 into 8 elemements of \p elemType.
  virtual std::array<Value, 8>
  upcastPackedFp4(Fp4ToFpOp op, ConversionPatternRewriter &rewriter,
                  Value packedVec, Type elemType) const = 0;
};

} // namespace mlir::triton::gpu

#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_FP4_TO_FP_OP_TO_LLVM_BASE_H
