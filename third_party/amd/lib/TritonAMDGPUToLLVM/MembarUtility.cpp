#include "third_party/amd/include/TritonAMDGPUToLLVM/MembarUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {
namespace {
// Returns true if one of the operands is a LocalLoad synced via AsyncWait.
bool filterAsyncLocalLoadsDeppendencies(Operation *op1, Operation *op2) {
  auto isAsyncLoad = [](Operation *op) {
    return llvm::isa<triton::gpu::AsyncCopyGlobalToLocalOp,
                     triton::amdgpu::BufferLoadToLocalOp>(op);
  };
  auto isLocalLoadWithAsyncWaitToken = [](Operation *op) {
    auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
    if (!localLoad)
      return false;
    auto token = localLoad.getToken();
    if (!token || !token.getDefiningOp<triton::gpu::AsyncWaitOp>())
      return false;
    return true;
  };

  // Early return if neither or both operands are an AsyncLoad
  if (isAsyncLoad(op1) == isAsyncLoad(op2)) {
    return false;
  }

  return isLocalLoadWithAsyncWaitToken(op1) ||
         isLocalLoadWithAsyncWaitToken(op2);
};
} // namespace

bool membarFilter(Operation *op1, Operation *op2) {
  return filterAsyncLocalLoadsDeppendencies(op1, op2);
}
} // namespace mlir::triton::AMD
