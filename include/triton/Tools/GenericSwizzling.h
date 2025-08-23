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
// Store the lane indices that are used in the contiguous part
// of an operation and in the address part.
// The laneAddr part just represents the indices used in one wavefront
// For now we just represent tiles with full vectorisation, meaning
// ld.shared.b32.v4/st.shared.b32.v4
// ldmatrix.v4 / stmatrix.v4
// ldmatrix.trans.v4 / stmatrix.trans.v4
struct LocalMemOpTile {
  // If laneContig.size() < log2(128/bitwidth), we assume that
  // the first log2(128/bitwidth) - laneContig.size() bases are registers
  llvm::SmallVector<int32_t> laneContig;
  // If laneAddr.size() < 3, we assume that the first
  // 3 - laneAddr.size() bases are registers
  llvm::SmallVector<int32_t> laneAddr;
};

// Given a set of possible instructions given by
// targetInfo.laneIdTiles(bitwidth) returns the optimal swizzling given these
// instructions and a pair of indices into the ldStTiles that's needed to lower
// this swizzling
std::pair<LinearLayout, std::pair<int32_t, int32_t>>
optimalSwizzling(const LinearLayout &src, const LinearLayout &dst,
                 llvm::ArrayRef<LocalMemOpTile> srcTiles,
                 llvm::ArrayRef<LocalMemOpTile> dstTiles, int32_t bitwidth);

LinearLayout optimalSwizzlingLdSt(const LinearLayout &src,
                                  const LinearLayout &dst, int32_t bitwidth);

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
