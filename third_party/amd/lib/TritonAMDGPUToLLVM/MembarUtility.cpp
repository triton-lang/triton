#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::AMD {
namespace {
// Returns true if
// 1) one is LocalLoad synced via AsyncWait.
// 2) both are AsyncLoad
bool filterAsyncLocalLoadsDependencies(Operation *op1, Operation *op2,
                                       Allocation *allocation) {
  auto isAsyncLoad = [](Operation *op) {
    return op->hasTrait<mlir::OpTrait::GlobalToLocalCopyTrait>();
  };
  auto isLocalLoadWithAsyncWaitToken = [](Operation *op) {
    auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
    return localLoad && isSyncedViaAsyncWait(localLoad);
  };
  // Returns the local memory operands an op reads from or writes to. Fused TDM
  // copies write to several destinations, hence the vector.
  auto getMemdescValues = [](Operation *op) -> SmallVector<Value> {
    return llvm::TypeSwitch<Operation *, SmallVector<Value>>(op)
        .Case<triton::amdgpu::BufferLoadToLocalOp>(
            [](auto op) -> SmallVector<Value> { return {op.getDest()}; })
        .Case<triton::gpu::AsyncCopyGlobalToLocalOp,
              triton::amdgpu::AsyncTDMCopyGlobalToLocalOp>(
            [](auto op) -> SmallVector<Value> { return {op.getResult()}; })
        .Case<triton::amdgpu::AsyncTDMFusedCopyGlobalToLocalOp>(
            [](auto op) -> SmallVector<Value> {
              return llvm::to_vector(op.getDests());
            })
        .Case<triton::amdgpu::AsyncTDMGatherOp>(
            [](auto op) -> SmallVector<Value> { return {op.getDst()}; })
        .Case<triton::gpu::LocalLoadOp>(
            [](auto op) -> SmallVector<Value> { return {op.getSrc()}; })
        .Default([](Operation *) { return SmallVector<Value>{}; });
  };

  // Early return if neither operands are an AsyncLoad
  if (!isAsyncLoad(op1) && !isAsyncLoad(op2)) {
    return false;
  }
  // Filter if both operands are an AsyncLoad
  if (isAsyncLoad(op1) && isAsyncLoad(op2)) {
    return true;
  }

  SmallVector<Value> op1Memdescs = getMemdescValues(op1);
  SmallVector<Value> op2Memdescs = getMemdescValues(op2);
  if (op1Memdescs.empty() || op2Memdescs.empty())
    return false;

  auto collectBufferIds = [&](ArrayRef<Value> memdescs) {
    Allocation::BufferIdSetT bufferIds;
    for (Value memdesc : memdescs) {
      auto ids = allocation->getAllBufferIdsWithAliases(memdesc);
      bufferIds.insert(ids.begin(), ids.end());
    }
    return bufferIds;
  };
  auto op1BufferIds = collectBufferIds(op1Memdescs);
  auto op2BufferIds = collectBufferIds(op2Memdescs);

  // Check if operations access the same buffer
  bool sameBuffer = llvm::any_of(
      op1BufferIds, [&](auto id) { return op2BufferIds.count(id); });

  if (!sameBuffer)
    return false;

  return isLocalLoadWithAsyncWaitToken(op1) ||
         isLocalLoadWithAsyncWaitToken(op2);
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
  return (filterAsyncLocalLoadsDependencies(op1, op2, allocation) ||
          filterLDSMemoryBarriersDependencies(op1, op2));
}
} // namespace mlir::triton::AMD
