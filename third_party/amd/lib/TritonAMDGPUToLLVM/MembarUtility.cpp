#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::AMD {
namespace {

bool isAsyncWrite(Operation *op) {
  return op->hasTrait<OpTrait::MemAsyncWriteOpTrait>();
}

bool isLocalLoadSyncedViaWait(Operation *op) {
  auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
  return localLoad && isSyncedViaAsyncWait(localLoad);
}

Value getMemdescValue(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<triton::amdgpu::BufferLoadToLocalOp>(
          [](auto op) { return op.getDest(); })
      .Case<triton::gpu::AsyncCopyGlobalToLocalOp>(
          [](auto op) { return op.getResult(); })
      .Case<triton::amdgpu::AsyncTDMCopyGlobalToLocalOp>(
          [](auto op) { return op.getResult(); })
      .Case<triton::amdgpu::AsyncTDMGatherOp>(
          [](auto op) { return op.getDst(); })
      .Case<triton::gpu::LocalLoadOp>([](auto op) { return op.getSrc(); })
      .Default([](Operation *) { return Value(); });
}

// s_barrier cannot make in-flight async writes visible; lgkmcnt is emitted
// in lowering of wait/barrier ops.  Filter async write hazard pairs when
// both ops access the same buffer and the local_load is synced via wait.
bool filterAsyncWriteDependencies(Operation *op1, Operation *op2,
                                  Allocation *allocation) {
  if (!isAsyncWrite(op1) && !isAsyncWrite(op2))
    return false;

  // One async, one not: require same-buffer and synced-via-wait.
  if (isAsyncWrite(op1) == isAsyncWrite(op2))
    return false;

  Value op1Memdesc = getMemdescValue(op1);
  Value op2Memdesc = getMemdescValue(op2);
  if (!op1Memdesc || !op2Memdesc)
    return false;
  auto op1BufferIds = allocation->getAllBufferIdsWithAliases(op1Memdesc);
  auto op2BufferIds = allocation->getAllBufferIdsWithAliases(op2Memdesc);
  bool sameBuffer = llvm::any_of(
      op1BufferIds, [&](auto id) { return op2BufferIds.count(id); });
  if (!sameBuffer)
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
  return (filterAsyncWriteDependencies(op1, op2, allocation) ||
          filterLDSMemoryBarriersDependencies(op1, op2));
}
} // namespace mlir::triton::AMD
