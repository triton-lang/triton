#ifndef TRITON_GENERIC_SWIZZLING_H
#define TRITON_GENERIC_SWIZZLING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <utility>

namespace mlir::triton {
class LinearLayout;
class TargetInfoBase;
} // namespace mlir::triton

namespace mlir::triton::gpu {
enum class InstrType { Vec, Matrix, MatrixTrans };

LinearLayout optimalSwizzlingLdSt(const LinearLayout &src,
                                  const LinearLayout &dst, int32_t bitwidth);

std::pair<LinearLayout, std::pair<InstrType, InstrType>>
optimalSwizzling(const LinearLayout &src, const LinearLayout &dst,
                 int32_t bitwidth, const TargetInfoBase &targetInfo);

std::pair<int, int> logBankConflictsLdSt(const LinearLayout &src,
                                         const LinearLayout &dst,
                                         const LinearLayout &smem,
                                         int32_t bitwidth);

std::pair<int, int> logBankConflicts(llvm::ArrayRef<int32_t> tileSrc,
                                     llvm::ArrayRef<int32_t> tileDst,
                                     const LinearLayout &smem,
                                     int32_t bitwidth);
} // namespace mlir::triton::gpu

#endif // TRITON_GENERIC_SWIZZLING_H
