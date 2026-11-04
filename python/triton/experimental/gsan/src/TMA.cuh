#pragma once

#include "Compat.h"

namespace gsan {

struct TensorDescInfo {
  static constexpr int kShapeWordBase = 8;
  static constexpr int kStrideWordBase = 3;
  static constexpr int kStrideUnitBytes = 16;

  uintptr_t base;
  const uint32_t *words;
  int rank;
  int bytesPerElem;

  GSAN_DEVICE int64_t shape(int dim) const {
    return static_cast<int64_t>(words[kShapeWordBase + rank - 1 - dim]) + 1;
  }

  GSAN_DEVICE uintptr_t byteStride(int dim) const {
    if (dim + 1 == rank)
      return bytesPerElem;
    return static_cast<uintptr_t>(words[kStrideWordBase + rank - 2 - dim]) *
           kStrideUnitBytes;
  }
};

inline GSAN_DEVICE TensorDescInfo decodeTensorDesc(const void *descriptor,
                                                   int rank, int bytesPerElem) {
  return {*reinterpret_cast<const uintptr_t *>(descriptor),
          reinterpret_cast<const uint32_t *>(descriptor), rank, bytesPerElem};
}

} // namespace gsan
