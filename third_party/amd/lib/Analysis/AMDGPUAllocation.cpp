#include "Analysis/AMDGPUAllocation.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"

namespace mlir::triton::AMD {

// Max shmem instruction in bits
constexpr int kMaxShmemVecBitLength = 128;

unsigned getNumScratchElemsPaddedCvt(RankedTensorType srcTy,
                                     RankedTensorType dstTy) {
  auto scratchConfig = getScratchConfigForCvt(srcTy, dstTy);
  return getNumScratchElements(scratchConfig.paddedRepShape);
}

SmallVector<unsigned> getRepShapeForCvt(RankedTensorType srcTy,
                                        RankedTensorType dstTy) {
  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();

  if (!cvtNeedsSharedMemory(srcTy, dstTy)) {
    return {};
  }

  if (shouldUseDistSmem(srcLayout, dstLayout)) {
    // TODO: padding to avoid bank conflicts
    return convertType<unsigned, int64_t>(gpu::getShapePerCTA(srcTy));
  }

  assert(srcLayout && dstLayout && "Unexpected layout in getRepShapeForCvt()");

  auto srcShapePerCTA = gpu::getShapePerCTA(srcTy);
  auto dstShapePerCTA = gpu::getShapePerCTA(dstTy);
  auto srcShapePerCTATile = ::mlir::triton::AMD::getShapePerCTATile(srcTy);
  auto dstShapePerCTATile = ::mlir::triton::AMD::getShapePerCTATile(dstTy);

  assert(srcTy.getRank() == dstTy.getRank() &&
         "src and dst must have the same rank");

  unsigned rank = dstTy.getRank();
  SmallVector<unsigned> repShape(rank);
  for (unsigned d = 0; d < rank; ++d) {
    repShape[d] =
        std::max(std::min<unsigned>(srcShapePerCTA[d], srcShapePerCTATile[d]),
                 std::min<unsigned>(dstShapePerCTA[d], dstShapePerCTATile[d]));
  }
  return repShape;
}

std::pair<unsigned, unsigned>
getScratchCvtInOutVecLengths(RankedTensorType srcTy, RankedTensorType dstTy) {
  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();

  auto srcLinAttr = gpu::toLinearEncoding(srcTy);
  auto dstLinAttr = gpu::toLinearEncoding(dstTy);
  auto inOrd = srcLinAttr.getOrder();
  auto outOrd = dstLinAttr.getOrder();

  unsigned rank = srcTy.getRank();

  unsigned srcContigPerThread = srcLinAttr.getContigPerThread()[inOrd[0]];
  unsigned dstContigPerThread = dstLinAttr.getContigPerThread()[outOrd[0]];
  unsigned innerDim = rank - 1;
  unsigned inVec = outOrd[0] != innerDim  ? 1
                   : inOrd[0] != innerDim ? 1
                                          : srcContigPerThread;
  unsigned outVec = outOrd[0] != innerDim ? 1 : dstContigPerThread;

  return {inVec, outVec};
}

ScratchConfig getScratchConfigForCvt(RankedTensorType srcTy,
                                     RankedTensorType dstTy) {
  // Initialize vector sizes and stride
  auto repShape = getRepShapeForCvt(srcTy, dstTy);
  if (repShape.empty())
    return ScratchConfig({}, {});
  ScratchConfig scratchConfig(repShape, repShape);
  auto rank = repShape.size();
  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();

  assert(cvtNeedsSharedMemory(srcTy, dstTy));
  auto outOrd = gpu::getOrder(dstTy);
  scratchConfig.order = outOrd;

  std::tie(scratchConfig.inVec, scratchConfig.outVec) =
      getScratchCvtInOutVecLengths(srcTy, dstTy);
  // We can't write a longer vector than the shape of shared memory.
  // This shape might be smaller than the tensor shape in case we decided to
  // do the conversion in multiple iterations.
  unsigned contiguousShapeDim = scratchConfig.repShape[scratchConfig.order[0]];
  scratchConfig.inVec = std::min(scratchConfig.inVec, contiguousShapeDim);
  scratchConfig.outVec = std::min(scratchConfig.outVec, contiguousShapeDim);
  // Clamp the vector length to kMaxShmemVecBitLength / element bitwidth as this
  // is the max vectorisation
  auto inBitWidth = getBitwidth(srcTy);
  auto outBitWidth = getBitwidth(dstTy);
  scratchConfig.inVec =
      std::min(scratchConfig.inVec, kMaxShmemVecBitLength / inBitWidth);
  scratchConfig.outVec =
      std::min(scratchConfig.outVec, kMaxShmemVecBitLength / outBitWidth);

  // No padding is required if the tensor is 1-D, or if all dimensions except
  // the first accessed dimension have a size of 1.
  if (rank <= 1 || product(repShape) == repShape[outOrd[0]])
    return scratchConfig;

  auto paddedSize = std::max(scratchConfig.inVec, scratchConfig.outVec);
  scratchConfig.paddedRepShape[outOrd[0]] += paddedSize;
  return scratchConfig;
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
