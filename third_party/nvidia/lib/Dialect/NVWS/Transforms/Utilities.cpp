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

ArefCreateOp createArefCreateOp(OpBuilder &builder, ArrayRef<Type> arefTypes,
                                ValueRange allocOps, Location loc) {
  auto ctx = builder.getContext();
  auto arefTy = ArefType::get(ctx, TypeArrayAttr::get(ctx, arefTypes));
  return ArefCreateOp::create(builder, loc, arefTy, allocOps);
}

} // namespace mlir::triton::nvws
