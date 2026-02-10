#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {
namespace {

bool isAsyncWrite(Operation *op) {
  return op->hasTrait<OpTrait::MemAsyncWriteOpTrait>();
}

bool isLocalLoadSyncedViaWait(Operation *op) {
  auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
  return localLoad && isSyncedViaAsyncWait(localLoad);
}

// s_barrier synchronizes threads but not the DMA engine; only an explicit
// async wait can make in-flight DMA writes visible.  Filter such pairs.
bool filterAsyncWriteDependencies(Operation *op1, Operation *op2) {
  bool op1Async = isAsyncWrite(op1);
  bool op2Async = isAsyncWrite(op2);
  if (op1Async && op2Async)
    return true;
  if (!op1Async && !op2Async)
    return false;
  return isLocalLoadSyncedViaWait(op1) || isLocalLoadSyncedViaWait(op2);
}

bool filterLDSMemoryBarriersDependencies(Operation *op1, Operation *op2) {
  auto isLDSMemoryBarrierOp = [](Operation *op) {
    return llvm::isa<triton::amdgpu::InitBarrierOp,
                     triton::amdgpu::ArriveBarrierOp,
                     triton::amdgpu::AsyncCopyMbarrierArriveOp,
                     triton::amdgpu::WaitBarrierOp>(op);
  };

  return (isLDSMemoryBarrierOp(op1) && isLDSMemoryBarrierOp(op2));
}
} // namespace

bool membarFilter(Operation *op1, Operation *op2, bool /*op1IsRead*/,
                  bool /*op2IsRead*/, Allocation *allocation) {
  return (filterAsyncWriteDependencies(op1, op2) ||
          filterLDSMemoryBarriersDependencies(op1, op2));
}
} // namespace mlir::triton::AMD
