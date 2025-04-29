#include "third_party/amd/include/TritonAMDGPUToLLVM/MembarUtility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {
namespace {
bool isSyncedByAsyncWait(Operation *op1, Operation *op2) {
  auto checkForAsyncLocalLoad = [](Operation *op) {
    auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
    if (!localLoad)
      return false;
    auto token = localLoad.getToken();
    if (!token || !token.getDefiningOp<triton::gpu::AsyncWaitOp>())
      return false;
    return true;
  };

  return checkForAsyncLocalLoad(op1) || checkForAsyncLocalLoad(op2);
};
} // namespace

bool membarFilter(Operation *op1, Operation *op2) {
  return isSyncedByAsyncWait(op1, op2);
}
} // namespace mlir::triton::AMD
