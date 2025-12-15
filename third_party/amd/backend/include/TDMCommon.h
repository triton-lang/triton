#ifndef TRITON_THIRD_PARTY_AMD_BACKEND_INCLUDE_TDMCOMMON_H
#define TRITON_THIRD_PARTY_AMD_BACKEND_INCLUDE_TDMCOMMON_H

#if defined(__cplusplus)
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// C-compatible TDM utilities shared between host-side (driver.c) and
// device-side (TDMUtility.cpp) code.
//
// This is intentionally kept header-only to avoid introducing
// dependencies between the compiler and runtime components.
//===----------------------------------------------------------------------===//

#include <stdint.h>

// Compute warp distribution across dimensions.
// Distributes warps starting from the first dimension, assigning as many
// warps as possible without exceeding the block shape.
static inline void tdmGetWarpDistribution(const int64_t *blockShape,
                                          int numDims, int numWarps,
                                          int *warpsOut) {
  for (int i = 0; i < numDims; ++i)
    warpsOut[i] = 1;

  int remainingWarps = numWarps;
  for (int i = 0; i < numDims && remainingWarps > 1; ++i) {
    while (remainingWarps > 1 && warpsOut[i] * 2 <= blockShape[i]) {
      warpsOut[i] *= 2;
      remainingWarps /= 2;
    }
  }

  if (remainingWarps > 1)
    warpsOut[numDims - 1] *= remainingWarps;
}

// Compute per-warp block sizes after distributing warps.
// Only adjusts first 2 dimensions; higher dimensions remain unchanged.
static inline void tdmGetAdjustedBlockShape(const int64_t *blockShape,
                                            int numDims, int numWarps,
                                            int64_t *adjustedOut) {
  int warps[5];
  tdmGetWarpDistribution(blockShape, numDims, numWarps, warps);

  if (numDims >= 2) {
    for (int i = 0; i < numDims; i++) {
      int warpDiv = warps[i];
      adjustedOut[i] = (blockShape[i] + warpDiv - 1) / warpDiv;
    }
  } else {
    adjustedOut[0] = (blockShape[0] + numWarps - 1) / numWarps;
  }
}

#if defined(__cplusplus)
}
#endif
#endif // TRITON_THIRD_PARTY_AMD_BACKEND_INCLUDE_TDMCOMMON_H
