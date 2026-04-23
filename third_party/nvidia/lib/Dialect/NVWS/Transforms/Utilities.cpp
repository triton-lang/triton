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
    return LocalAllocOp::create(builder, loc, memDescType, src);
  } else {
    assert(isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace()));
    return TMEMAllocOp::create(builder, loc, memDescType, src);
  }
}

int getSemaphoreDepth(MemDescType bufTy) {
  auto shape = bufTy.getShape();
  return isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(bufTy.getEncoding())
             ? 1
             : shape[0];
}

MemDescType getSemaphoreViewBufferType(MemDescType bufTy) {
  auto isScalesEnc =
      isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(bufTy.getEncoding());
  auto shape = bufTy.getShape();
  return gpu::MemDescType::get(isScalesEnc ? shape : shape.drop_front(),
                               bufTy.getElementType(), bufTy.getEncoding(),
                               bufTy.getMemorySpace(),
                               /*mutableMemory*/ true,
                               /*allocShape=*/bufTy.getAllocShape());
}

MemDescType getSemaphoreMultiBufferedType(MemDescType bufTy, int depth) {
  auto shape = bufTy.getShape();
  SmallVector<int64_t> bufferShape(shape.begin(), shape.end());
  if (!isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(bufTy.getEncoding()))
    bufferShape.insert(bufferShape.begin(), depth);
  return gpu::MemDescType::get(bufferShape, bufTy.getElementType(),
                               bufTy.getEncoding(), bufTy.getMemorySpace(),
                               /*mutableMemory*/ true);
}

scf::ForOp getOuterWSLoop(scf::ForOp innerFor) {
  auto wsLoop = innerFor;
  while (wsLoop && !wsLoop->hasAttr(triton::kWarpSpecializeAttrName)) {
    wsLoop = wsLoop->getParentOfType<scf::ForOp>();
  }
  return wsLoop;
}

} // namespace mlir::triton::nvws
