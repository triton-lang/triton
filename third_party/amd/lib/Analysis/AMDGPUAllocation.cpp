#include "Analysis/AMDGPUAllocation.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

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

unsigned AMDAllocationAnalysisScratchSizeFn(Operation *op) {

  if (auto cvtLayout = dyn_cast<mlir::triton::gpu::ConvertLayoutOp>(op)) {
    auto srcTy = cvtLayout.getSrc().getType();
    auto dstTy = cvtLayout.getType();
    return getConvertLayoutScratchInBytes(srcTy, dstTy);
  }

  return defaultAllocationAnalysisScratchSizeFn(op);
}

} // namespace mlir::triton::AMD
