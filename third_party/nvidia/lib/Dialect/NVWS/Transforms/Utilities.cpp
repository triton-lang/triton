#include "Utilities.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;

namespace mlir::triton::nvws {

Operation *createAlloc(OpBuilder &builder, Location loc,
                       MemDescType memDescType, Value src) {
  if (isa<SharedMemorySpaceAttr>(memDescType.getMemorySpace())) {
    return builder.create<LocalAllocOp>(loc, memDescType, src);
  } else {
    assert(isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace()));
    return builder.create<TMEMAllocOp>(loc, memDescType, src);
  }
}

ArefCreateOp createArefCreateOp(OpBuilder &builder, ArrayRef<Type> arefTypes,
                                ValueRange allocOps, Location loc) {
  auto ctx = builder.getContext();
  auto arefTy = ArefType::get(ctx, TypeArrayAttr::get(ctx, arefTypes));
  return builder.create<ArefCreateOp>(loc, arefTy, allocOps);
}

#if 0
std::optional<PartitionId> getPartitionId(Operation *op) {
  if (auto partitionAttr = op->getAttrOfType<IntegerAttr>(kPartitionAttrName)) {
    IntegerAttr tagAttr;
    while (op && !tagAttr) {
      tagAttr = op->getAttrOfType<IntegerAttr>(kWarpSpecializeTagAttrName);
      op = op->getParentOp();
    }
    if (tagAttr)
      return PartitionId(partitionAttr.getInt(), tagAttr.getInt());
  }
  return {};
}
#endif

int arefDepth(MemDescType bufTy) {
  auto shape = bufTy.getShape();
  return isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(bufTy.getEncoding())
             ? 1
             : shape[0];
}

MemDescType arefViewBufferType(MemDescType bufTy) {
  auto isScalesEnc =
      isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(bufTy.getEncoding());
  auto shape = bufTy.getShape();
  return gpu::MemDescType::get(isScalesEnc ? shape : shape.drop_front(),
                               bufTy.getElementType(), bufTy.getEncoding(),
                               bufTy.getMemorySpace(),
                               /*mutableMemory*/ true,
                               /*allocShape=*/bufTy.getAllocShape());
}

MemDescType arefMultiBufferedType(MemDescType bufTy, int depth) {
  auto shape = bufTy.getShape();
  SmallVector<int64_t> bufferShape(shape.begin(), shape.end());
  if (!isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(bufTy.getEncoding()))
    bufferShape.insert(bufferShape.begin(), depth);
  return gpu::MemDescType::get(bufferShape, bufTy.getElementType(),
                               bufTy.getEncoding(), bufTy.getMemorySpace(),
                               /*mutableMemory*/ true);
}

} // namespace mlir::triton::nvws
