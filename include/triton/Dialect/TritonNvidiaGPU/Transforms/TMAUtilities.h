#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "llvm/Support/Casting.h"

namespace mlir::triton::nvidia_gpu {

constexpr inline int TMA_SIZE_BYTES = 128;
constexpr inline int TMA_ALIGN = 128;

inline bool isFp4Padded(Attribute encoding) {
  auto mmaEnc = dyn_cast<gpu::NVMMASharedEncodingAttr>(encoding);
  return mmaEnc && mmaEnc.getFp4Padded();
}

gpu::CGAEncodingAttr updateCGALayoutForShape(gpu::CGAEncodingAttr cgaLayout,
                                             ArrayRef<int64_t> shape);

gpu::SharedEncodingTrait
updateEncodingForShape(Operation *op, gpu::SharedEncodingTrait encoding,
                       RankedTensorType tensorType);

triton::gpu::SharedEncodingTrait
getEncodingFromDescriptor(Operation *op, RankedTensorType tensorType,
                          Value desc);

inline SmallVector<int64_t>
getTMABlockShape(Attribute encoding, ArrayRef<int64_t> shapePerCTA,
                 bool packedSize, gpu::TMAMode mode = gpu::TMAMode::Tiled) {
  auto mmaEnc = cast<gpu::NVMMASharedEncodingAttr>(encoding);
  return triton::gpu::getTMABlockShape(
      shapePerCTA, mmaEnc.getElementBitWidth(), mmaEnc.getSwizzlingByteWidth(),
      mmaEnc.getFp4Padded(), mmaEnc.getTransposed(), packedSize, mode);
}

inline SmallVector<int64_t>
getTMABlockShape(RankedTensorType ty, bool packedSize,
                 gpu::TMAMode mode = gpu::TMAMode::Tiled) {
  auto shapePerCTA = gpu::getShapePerCTA(ty);
  return getTMABlockShape(ty.getEncoding(), shapePerCTA, packedSize, mode);
}

inline SmallVector<int64_t>
getTMABlockShape(triton::gpu::MemDescType ty, bool packedSize,
                 gpu::TMAMode mode = gpu::TMAMode::Tiled) {
  auto shapePerCTA = gpu::getShapePerCTA(ty);
  return getTMABlockShape(ty.getEncoding(), shapePerCTA, packedSize, mode);
}

FailureOr<int> getTMASwizzleMode(Location loc, TensorDescType ty);
FailureOr<int> getTMAElementType(Location loc, TensorDescType ty);

LogicalResult createTMADesc(Value tmaPtr, MakeTensorDescOp op,
                            OpBuilder &builder);

// Compute a LinearLayout mapping TMA message indices to tensor coordinates.
// Maps from input dimension "msg" (and "block" for CGA) to output dimensions
// "dim0", "dim1", etc.
//
// For a tensor with shape [512, 64] and TMA block shape [256, 32]:
//   - numMsgs = (512/256) * (64/32) = 4
//   - msg=0 -> (0, 0), msg=1 -> (256, 0), msg=2 -> (0, 32), msg=3 -> (256, 32)
//
// The `packedSize` parameter controls whether the block shape accounts for
// FP4 padding (true for address computation, false for descriptor setup).
LinearLayout getMsgToPackedOffsetLayout(gpu::MemDescType ty, gpu::TMAMode mode,
                                        bool packedSize = true);

} // namespace mlir::triton::nvidia_gpu
