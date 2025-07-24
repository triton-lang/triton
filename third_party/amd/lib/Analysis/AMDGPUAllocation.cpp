#include "Analysis/AMDGPUAllocation.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {

constexpr int globalPtrBitWidth = 64;

static unsigned getBitwidth(RankedTensorType ty) {
  auto isPtr = isa<PointerType>(ty.getElementType());
  return isPtr ? globalPtrBitWidth : std::max(ty.getElementTypeBitWidth(), 8u);
}

unsigned getConvertLayoutScratchInBytes(RankedTensorType srcTy,
                                        RankedTensorType dstTy,
                                        bool usePadding) {
  if (!cvtNeedsSharedMemory(srcTy, dstTy))
    return 0;
  unsigned elems = 0;
  if (usePadding) {
    elems = getNumScratchElemsPaddedCvt(srcTy, dstTy);
  } else {
    elems = getNumScratchElemsSwizzledCvt(srcTy, dstTy);
  }
  return elems * getBitwidth(srcTy) / 8;
}

unsigned AMDAllocationAnalysisScratchSizeFn(Operation *op) {

  if (auto cvtLayout = dyn_cast<mlir::triton::gpu::ConvertLayoutOp>(op)) {
    auto srcTy = cvtLayout.getSrc().getType();
    auto dstTy = cvtLayout.getType();
    return getConvertLayoutScratchInBytes(srcTy, dstTy,
                                          op->hasAttr(AttrSharedMemPadded));
  }

  return defaultAllocationAnalysisScratchSizeFn(op);
}

} // namespace mlir::triton::AMD
