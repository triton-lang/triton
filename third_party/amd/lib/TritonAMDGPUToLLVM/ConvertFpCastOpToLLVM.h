#ifndef TRITON_THIRD_PARTY_AMD_LIB_CONVERTFPCASTOPTOLLVM_H_
#define TRITON_THIRD_PARTY_AMD_LIB_CONVERTFPCASTOPTOLLVM_H_

#include "mlir/Transforms/DialectConversion.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"

namespace mlir::triton::AMD {
Value convertBf16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                        const Value &v);
Value convertFp32ToBf16(Location loc, ConversionPatternRewriter &rewriter,
                        const Value &v, const RoundingMode rounding);
SmallVector<Value> convertFp32ToF16rtne(Location loc,
                                        ConversionPatternRewriter &rewriter,
                                        Type inElemTy, Type outElemTy,
                                        gpu::MultipleOperandsRange operands,
                                        amdgpu::ISAFamily isaFamily);
} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_CONVERTFPCASTOPTOLLVM_H_
