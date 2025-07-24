#include "Utilities.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace nvws {

Operation *createAlloc(OpBuilder &builder, Location loc,
                       MemDescType memDescType, Value src) {
  if (isa<SharedMemorySpaceAttr>(memDescType.getMemorySpace())) {
    return builder.create<LocalAllocOp>(loc, memDescType, src);
  } else {
    assert(isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace()));
    return builder.create<TMEMAllocOp>(loc, memDescType, src);
  }
}

MemDescType getArefbufMemDescType(MemDescType memDescType, int32_t depth) {
  auto shape = memDescType.getShape();
  SmallVector<int64_t> bufferShape(shape.begin(), shape.end());
  bufferShape.insert(bufferShape.begin(), depth);
  return MemDescType::get(bufferShape, memDescType.getElementType(),
                          memDescType.getEncoding(),
                          memDescType.getMemorySpace(), true);
}

ArefCreateOp createArefCreateOp(OpBuilder &builder,
                                const SmallVector<Type> &arefTypes,
                                const SmallVector<Value> &allocOps,
                                Location loc) {
  auto ctx = builder.getContext();
  auto arefTy = ArefType::get(ctx, TypeArrayAttr::get(ctx, arefTypes));
  return builder.create<ArefCreateOp>(loc, arefTy, allocOps);
}

} // namespace nvws
} // namespace triton
} // namespace mlir
