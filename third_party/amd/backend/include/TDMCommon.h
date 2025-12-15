#ifndef TRITON_THIRD_PARTY_AMD_BACKEND_INCLUDE_TDMCOMMON_H
#define TRITON_THIRD_PARTY_AMD_BACKEND_INCLUDE_TDMCOMMON_H

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
    adjustedOut[0] = (blockShape[0] + warps[0] - 1) / warps[0];
    adjustedOut[1] = (blockShape[1] + warps[1] - 1) / warps[1];
  } else {
    adjustedOut[0] = (blockShape[0] + numWarps - 1) / numWarps;
  }

  // Higher dimensions are not divided by warps
  for (int i = 2; i < numDims; ++i)
    adjustedOut[i] = blockShape[i];
}

#endif // TRITON_THIRD_PARTY_AMD_BACKEND_INCLUDE_TDMCOMMON_H
