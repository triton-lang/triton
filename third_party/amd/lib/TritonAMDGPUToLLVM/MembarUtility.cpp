#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {
namespace {
// Returns true if one of the operands is a LocalLoad synced via AsyncWait.
bool filterAsyncLocalLoadsDependencies(Operation *op1, Operation *op2) {
  auto isAsyncLoad = [](Operation *op) {
    return llvm::isa<triton::gpu::AsyncCopyGlobalToLocalOp,
                     triton::amdgpu::BufferLoadToLocalOp>(op);
  };
  auto isLocalLoadWithAsyncWaitToken = [](Operation *op) {
    auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
    return localLoad && isSyncedViaAsyncWait(localLoad);
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
  return filterAsyncLocalLoadsDependencies(op1, op2);
}

void membarInsertBarrierCDNA4(Operation *op, OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
  if (isa<triton::gpu::AsyncWaitOp>(op)) {
    constexpr int32_t ldsOnlyBits = ~(0x1f << 8);
    builder->create<ROCDL::SWaitcntOp>(op->getLoc(),
                                       builder->getI32IntegerAttr(ldsOnlyBits));
    builder->create<ROCDL::SBarrierOp>(op->getLoc());
  } else {
    builder->create<mlir::gpu::BarrierOp>(op->getLoc());
  }
}

} // namespace mlir::triton::AMD
