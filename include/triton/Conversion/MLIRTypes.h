#ifndef TRITON_CONVERSION_MLIR_TYPES_H
#define TRITON_CONVERSION_MLIR_TYPES_H

#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// This file redefines some common MLIR types for easy usage.
namespace mlir {
namespace triton {
namespace type {

// Integer types
inline Type i32Ty(MLIRContext *ctx) { return IntegerType::get(ctx, 32); }
inline Type i16Ty(MLIRContext *ctx) { return IntegerType::get(ctx, 16); }
inline Type i8Ty(MLIRContext *ctx) { return IntegerType::get(ctx, 8); }
inline Type u32Ty(MLIRContext *ctx) {
  return IntegerType::get(ctx, 32, IntegerType::Unsigned);
}
inline Type u1Ty(MLIRContext *ctx) {
  return IntegerType::get(ctx, 1, IntegerType::Unsigned);
}

// Float types
inline Type f16Ty(MLIRContext *ctx) { return Float16Type::get(ctx); }
inline Type f32Ty(MLIRContext *ctx) { return Float32Type::get(ctx); }
inline Type f64Ty(MLIRContext *ctx) { return Float64Type::get(ctx); }
inline Type bf16Ty(MLIRContext *ctx) { return BFloat16Type::get(ctx); }

inline bool isFloat(Type type) {
  return type.isF32() || type.isF64() || type.isF16() || type.isF128() ||
         type.isBF16() || type.isFloat8E4M3B11FNUZ() || type.isFloat8E4M3FN() ||
         type.isFloat8E4M3FNUZ() || type.isFloat8E5M2() ||
         type.isFloat8E5M2FNUZ();
}

inline bool isFloat8(Type type) {
  return type.isFloat8E4M3B11FNUZ() || type.isFloat8E4M3FN() ||
         type.isFloat8E4M3FNUZ() || type.isFloat8E5M2() ||
         type.isFloat8E5M2FNUZ();
}

inline bool isInt(Type type) { return type.isIntOrFloat() && !isFloat(type); }

} // namespace type
} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_MLIR_TYPES_H
