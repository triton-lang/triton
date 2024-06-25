#ifndef TRITONCPU_CONVERSION_TRITONCPUOPT_OPTCOMMON_H
#define TRITONCPU_CONVERSION_TRITONCPUOPT_OPTCOMMON_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace triton {
namespace cpu {

inline bool isTyOrVectorOf(mlir::Type ty, mlir::Type elemTy) {
  if (auto vecTy = dyn_cast<mlir::VectorType>(ty))
    return vecTy.getElementType() == elemTy;
  return ty == elemTy;
}

inline bool isBf16(mlir::Type ty) {
  return isTyOrVectorOf(ty, mlir::BFloat16Type::get(ty.getContext()));
}

inline bool isFp32(mlir::Type ty) {
  return isTyOrVectorOf(ty, mlir::Float32Type::get(ty.getContext()));
}

inline mlir::Type toTyOrVectorOf(mlir::Type ty, mlir::Type elemTy) {
  if (auto vecTy = dyn_cast<mlir::VectorType>(ty))
    return vecTy.cloneWith(std::nullopt, elemTy);
  return elemTy;
}

inline mlir::Type toInt16(mlir::Type ty) {
  return toTyOrVectorOf(ty, mlir::IntegerType::get(ty.getContext(), 16));
}

inline mlir::Type toInt32(mlir::Type ty) {
  return toTyOrVectorOf(ty, mlir::IntegerType::get(ty.getContext(), 32));
}

inline mlir::Type toFp32(mlir::Type ty) {
  return toTyOrVectorOf(ty, mlir::Float32Type::get(ty.getContext()));
}

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif
