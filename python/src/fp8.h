#pragma once

#include "llvm/ADT/APFloat.h"

// Match FP8 casts: saturate overflow, but preserve FNUZ infinity-to-NaN
// behavior.
inline llvm::APFloat convertFp8(float value,
                                const llvm::fltSemantics &semantics) {
  llvm::APFloat result(value);
  bool isFnuz = &semantics == &llvm::APFloat::Float8E4M3FNUZ() ||
                &semantics == &llvm::APFloat::Float8E5M2FNUZ();
  bool saturate = result.isFinite() || (result.isInfinity() && !isFnuz);
  bool negative = result.isNegative();
  bool losesInfo;
  result.convert(semantics, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  if (saturate && !result.isFinite())
    return llvm::APFloat::getLargest(semantics, negative);
  return result;
}
