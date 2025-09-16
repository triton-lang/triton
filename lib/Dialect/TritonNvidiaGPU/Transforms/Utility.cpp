#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#define DEBUG_TYPE "ttng-utility"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
namespace mlir::triton::nvidia_gpu {

using namespace triton;

LogicalResult verifyBarrierType(Operation *op,
                                mlir::triton::gpu::MemDescType barrierType) {
  if (!barrierType.getElementType().isInteger(64) ||
      barrierType.getShape() != ArrayRef<int64_t>({1}))
    return op->emitOpError(
        "barrier allocation must be a descriptor of 1xi64 type");
  return success();
}

} // namespace mlir::triton::nvidia_gpu
