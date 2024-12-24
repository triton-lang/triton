#ifndef TRITON_CONVERSION_FMA_DOT_UTILITY_H
#define TRITON_CONVERSION_FMA_DOT_UTILITY_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::gpu {

/// \brief Abstract interface for scalar multiplication of Value vectors.
///
/// Enable generation of hardware specific code in different backends.
class FMAVectorMultiplier {
public:
  /// \returns scalar product of two arrays, plus c: a·b + c
  virtual mlir::Value multiplyVectors(mlir::ArrayRef<mlir::Value> a,
                                      mlir::ArrayRef<mlir::Value> b,
                                      mlir::Value c) = 0;

  virtual ~FMAVectorMultiplier() = default;
};

/// \brief Implements an abstract framework for dot conversion to llvm.
LogicalResult parametricConvertFMADot(triton::DotOp op,
                                      triton::DotOp::Adaptor adaptor,
                                      const LLVMTypeConverter *typeConverter,
                                      ConversionPatternRewriter &rewriter,
                                      FMAVectorMultiplier &multiplier);

} // namespace mlir::triton::gpu

#endif // TRITON_CONVERSION_FMA_DOT_UTILITY_H
