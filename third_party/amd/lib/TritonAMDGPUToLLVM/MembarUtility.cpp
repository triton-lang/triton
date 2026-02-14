#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::AMD {
namespace {

bool isAsyncWrite(Operation *op) {
  return op->hasTrait<OpTrait::MemAsyncLocalWriteOpTrait>();
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

// Suppress false-positive RAW/WAR barriers between an async write and a
// local_load on the same shared allocation in multi-buffered pipelines.
// See MembarUtility.h for the full rationale.
//
// Returns true (suppress barrier) when all three conditions hold:
//   1. Exactly one op is an async write and the other is not.
//   2. Both ops access the same shared-memory buffer.
//   3. The non-async op is a local_load whose token chains to an async wait.
bool filterAsyncWriteDependencies(Operation *op1, Operation *op2,
                                  Allocation *allocation) {
  // Exactly one op must be an async write and the other must not.
  // Neither-async and both-async (WAW) pairs are not handled here.
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
