#ifndef TRITON_CONVERSION_MLIR_TYPES_H_
#define TRITON_CONVERSION_MLIR_TYPES_H_

#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// This file redefines some common MLIR types for easy usage.
namespace mlir {
namespace triton {
namespace type {

// Integer types
Type i32Ty(MLIRContext *ctx) {
  return IntegerType::get(ctx, 32, IntegerType::Signed);
}
Type i8Ty(MLIRContext *ctx) {
  return IntegerType::get(ctx, 8, IntegerType::Signed);
}
Type u32Ty(MLIRContext *ctx) {
  return IntegerType::get(ctx, 32, IntegerType::Signless);
}
Type u1Ty(MLIRContext *ctx) { return IntegerType::get(ctx, 1); }

// Float types
Type f16Ty(MLIRContext *ctx) { return FloatType::getF16(ctx); }
Type f32Ty(MLIRContext *ctx) { return FloatType::getF32(ctx); }
Type f64Ty(MLIRContext *ctx) { return FloatType::getF64(ctx); }

static bool isFloat(Type type) {
  return type.isF32() || type.isF64() || type.isF16() || type.isF128();
}

static bool isInt(Type type) { return type.isIntOrFloat() && !isFloat(type); }

} // namespace type
} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_MLIR_TYPES_H_
