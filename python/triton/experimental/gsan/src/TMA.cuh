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
    int index = rank - 2 - dim;
    uintptr_t low = words[kStrideWordBase + index];
    uintptr_t high = (words[7] >> (4 * index)) & 0xf;
    return (low | (high << 32)) * kStrideUnitBytes;
  }

  GSAN_DEVICE int elementStride(int dim) const {
    return ((words[13] >> (3 * (rank - 1 - dim))) & 7) + 1;
  }

  GSAN_DEVICE int im2colCorner(int dim, bool upper) const {
    // Word 14 packs the W/H/D lower corners in its low half and the upper
    // corners in its high half. Each signed field has 16/8/5 bits at rank 3/4/5.
    int bits = rank == 3 ? 16 : rank == 4 ? 8 : 5;
    int shift = bits * (rank - 2 - dim) + (upper ? 16 : 0);
    int field = (words[14] >> shift) & ((1u << bits) - 1);
    int sign = 1 << (bits - 1);
    return (field ^ sign) - sign;
  }
};

inline GSAN_DEVICE TensorDescInfo decodeTensorDesc(const void *descriptor,
                                                   int rank, int bytesPerElem) {
  return {*reinterpret_cast<const uintptr_t *>(descriptor),
          reinterpret_cast<const uint32_t *>(descriptor), rank, bytesPerElem};
}

template <typename RowHandler>
GSAN_DEVICE void forEachIm2ColRow(const void *descriptor, const int *coords,
                                  int rank, const int16_t *blockShape,
                                  const int16_t *offsets, int bytesPerElem,
                                  int warpId, int numWarps,
                                  RowHandler &handleRow) {
  TensorDescInfo desc = decodeTensorDesc(descriptor, rank, bytesPerElem);
  int channelDim = rank - 1;
  int64_t channels = desc.shape(channelDim);
  int elementType = (desc.words[2] >> 7) & 0xf;
  // FP4-padded descriptors count individual nibbles. Triton's i8 descriptor
  // coordinates and block shape count packed bytes in global memory.
  if (elementType == 12)
    channels /= 2;
  // FP6 coordinates count logical six-bit values packed into global bytes.
  int channelBits = elementType == 13 ? 6 : bytesPerElem * 8;
  int64_t firstChannel = coords[channelDim] > 0 ? coords[channelDim] : 0;
  int64_t endChannel = static_cast<int64_t>(coords[channelDim]) + blockShape[1];
  endChannel = endChannel < channels ? endChannel : channels;
  if (endChannel <= firstChannel)
    return;

  int64_t pixel[5];
  int64_t lower[5], upper[5];
  int step[5];
  pixel[0] = coords[0];
  step[0] = desc.elementStride(0);
  for (int dim = 1; dim < channelDim; ++dim) {
    pixel[dim] = coords[dim];
    lower[dim] = desc.im2colCorner(dim, /*upper=*/false);
    upper[dim] = desc.shape(dim) + desc.im2colCorner(dim, /*upper=*/true);
    step[dim] = desc.elementStride(dim);
  }

  int64_t firstByte = firstChannel * channelBits / 8;
  int rowBytes = static_cast<int>((endChannel * channelBits + 7) / 8 - firstByte);
  for (int row = 0; row < blockShape[0]; ++row) {
    if (row % numWarps == warpId) {
      bool inBounds = pixel[0] >= 0 && pixel[0] < desc.shape(0);
      uintptr_t address = desc.base + firstByte;
      address += pixel[0] * desc.byteStride(0);
      for (int dim = 1; dim < channelDim; ++dim) {
        // PTX im2col offsets are unsigned, including the rank-3 16-bit offset.
        int64_t coord = pixel[dim] + static_cast<uint16_t>(offsets[dim - 1]);
        inBounds &= coord >= 0 && coord < desc.shape(dim);
        address += coord * desc.byteStride(dim);
      }
      if (inBounds)
        handleRow(address, rowBytes);
    }

    // Advance through the spatial pixel box, then into the next image. The
    // filter offsets move the accessed coordinates along with the pixel box.
    for (int dim = channelDim - 1; dim >= 0; --dim) {
      pixel[dim] += step[dim];
      if (dim == 0 || pixel[dim] < upper[dim])
        break;
      pixel[dim] = lower[dim];
    }
  }
}

} // namespace gsan
