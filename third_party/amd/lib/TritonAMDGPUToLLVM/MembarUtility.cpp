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

// Suppress the barrier that membar analysis would insert between an async DMA
// write (MemAsyncWriteOpTrait) and a local_load on the same shared buffer.
//
// Async DMA writes are not visible until an explicit wait completes.  A plain
// thread barrier does not provide this guarantee.  When the local_load is
// already guarded by an async wait, the barrier is redundant.
//
// Returns true (suppress barrier) when all three conditions hold:
//   1. Exactly one op is an async write and the other is not.
//   2. Both ops access the same shared-memory buffer.
//   3. The non-async op is a local_load that is synced via an async wait.
bool filterAsyncWriteDependencies(Operation *op1, Operation *op2,
                                  Allocation *allocation) {
  // Not relevant if neither op is an async write.
  if (!isAsyncWrite(op1) && !isAsyncWrite(op2))
    return false;

  // Two async writes (WAW) — keep the barrier; DMA ordering between
  // different async ops is not guaranteed by a wait alone.
  if (isAsyncWrite(op1) == isAsyncWrite(op2))
    return false;

  // Require that both ops touch the same buffer; unrelated allocations
  // should go through the normal hazard path.
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

  // The local_load must already be synchronized by a wait.
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
