#include "Analysis/AMDGPUAllocation.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"

namespace mlir::triton::AMD {

// Max shmem instruction in bits
constexpr int kMaxShmemVecBitLength = 128;

unsigned getConvertLayoutScratchInBytes(RankedTensorType srcTy,
                                        RankedTensorType dstTy) {
  if (!cvtNeedsSharedMemory(srcTy, dstTy))
    return 0;
  unsigned elems = getNumScratchElemsSwizzledCvt(srcTy, dstTy);
  return elems * getBitwidth(srcTy) / 8;
}

static unsigned getBufferAtomicScratchSizeInBytes(Operation *op) {
  Value result = op->getResult(0);
  if (result.use_empty())
    return 0;
  auto tensorTy = dyn_cast<RankedTensorType>(result.getType());
  if (!tensorTy)
    return 0;
  auto freeVariableMasks =
      gpu::toLinearLayout(tensorTy).getFreeVariableMasks();
  bool hasBroadcast = llvm::any_of(
      freeVariableMasks, [](auto mask) { return mask.second != 0; });
  if (!hasBroadcast)
    return 0;
  auto smemShape = convertType<unsigned>(gpu::getShapePerCTA(tensorTy));
  auto elems = getNumScratchElements(smemShape);
  if (elems == 0)
    return 0;
  auto elemTy = tensorTy.getElementType();
  return elems * std::max<int>(8, elemTy.getIntOrFloatBitWidth()) / 8;
}

unsigned AMDAllocationAnalysisScratchSizeFn(Operation *op) {

  if (auto cvtLayout = dyn_cast<mlir::triton::gpu::ConvertLayoutOp>(op)) {
    auto srcTy = cvtLayout.getSrc().getType();
    auto dstTy = cvtLayout.getType();
    return getConvertLayoutScratchInBytes(srcTy, dstTy);
  }

  if (auto ws = dyn_cast<mlir::triton::gpu::WarpSpecializeOp>(op)) {
    uint64_t captureSize = 0;
    // Tightly pack the captures in memory.
    for (Type type : ws.getPartitionOp().getOperandTypes()) {
      if (auto descType = dyn_cast<TensorDescType>(type))
        captureSize +=
            mlir::triton::amdgpu::getTensorDescNumDwords(descType) * 4;
      else
        captureSize += mlir::triton::gpu::getSharedMemorySize(type);
    }
    return captureSize;
  }

  if (isa<amdgpu::BufferAtomicCASOp, amdgpu::BufferAtomicRMWOp>(op))
    return getBufferAtomicScratchSizeInBytes(op);

  return defaultAllocationAnalysisScratchSizeFn(op);
}

} // namespace mlir::triton::AMD
