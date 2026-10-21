#pragma once

#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir::triton::nvidia_gpu {

inline bool isFp4Padded(Attribute encoding) {
  auto mmaEnc = dyn_cast<gpu::NVMMASharedEncodingAttr>(encoding);
  return mmaEnc && mmaEnc.getFp4Padded();
}

inline SmallVector<int64_t> getTMABlockShape(Attribute encoding,
                                             ArrayRef<int64_t> shapePerCTA,
                                             bool packedSize,
                                             gpu::TMAMode mode) {
  auto mmaEnc = cast<gpu::NVMMASharedEncodingAttr>(encoding);
  return gpu::getTMABlockShape(
      shapePerCTA, mmaEnc.getElementBitWidth(), mmaEnc.getSwizzlingByteWidth(),
      mmaEnc.getFp4Padded(), mmaEnc.getTransposed(), packedSize, mode);
}

inline SmallVector<int64_t>
getTMABlockShape(gpu::MemDescType ty, bool packedSize, gpu::TMAMode mode) {
  return getTMABlockShape(ty.getEncoding(), gpu::getShapePerCTA(ty), packedSize,
                          mode);
}

inline SmallVector<int64_t>
getTMABlockShape(TensorDescInterface ty, bool packedSize, gpu::TMAMode mode) {
  return getTMABlockShape(
      ty.getSharedLayout(),
      gpu::getShapePerCTA(ty.getSharedLayout(), ty.getShape()), packedSize,
      mode);
}

// Maps message starts, before the descriptor's affine subview translation.
struct TMAMessageLayouts {
  LinearLayout messageToPackedOffset;
  // Instruction coordinates, including FP4 expansion.
  LinearLayout messageToOffset;
  LinearLayout messageToShared;
};

TMAMessageLayouts getTMAMessageLayouts(gpu::MemDescType type,
                                       gpu::TMAMode mode);

struct TMAGatherScatterLayouts {
  LinearLayout xOffsetsLayout;
  LinearLayout messageToPackedOffset;
  LinearLayout messageToShared;
  unsigned numMessagesPerRow;
  unsigned messageSize;
  unsigned warpBroadcastMask;
};

// The x-offset values select global rows; their layout determines the shared
// destination of each gather4 message (or source for scatter4).
// Operand types must satisfy the TMA gather/scatter verifier.
TMAGatherScatterLayouts
getTMAGatherScatterLayouts(RankedTensorType xOffsetsType,
                           gpu::MemDescType type);

} // namespace mlir::triton::nvidia_gpu
