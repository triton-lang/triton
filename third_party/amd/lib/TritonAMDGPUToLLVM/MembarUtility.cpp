#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::AMD {
namespace {
bool filterLDSMemoryBarriersDependencies(Operation *op1, Operation *op2) {
  auto isLDSMemoryBarrierOp = [](Operation *op) {
    if (llvm::isa<triton::amdgpu::InitBarrierOp,
                  triton::amdgpu::ArriveBarrierOp,
                  triton::amdgpu::AsyncCopyMbarrierArriveOp,
                  triton::amdgpu::WaitBarrierOp>(op))
      return true;
    // A copy carrying an mbarrier arrives on it once complete, so the mbarrier
    // already orders it workgroup-wide. Copies without one are ordered by an
    // async wait, whose counter is per wave, so they are not covered here.
    if (auto mbarrierOp = llvm::dyn_cast<triton::gpu::MBarrierOpInterface>(op))
      return mbarrierOp.getBarrier() != nullptr;
    return false;
  };

  return (isLDSMemoryBarrierOp(op1) && isLDSMemoryBarrierOp(op2));
}
} // namespace

bool membarFilter(Operation *op1, Operation *op2, bool /*op1IsRead*/,
                  bool /*op2IsRead*/, Allocation *allocation,
                  const AllocationSlice &, const AllocationSlice &) {
  return filterLDSMemoryBarriersDependencies(op1, op2);
}
} // namespace mlir::triton::AMD
