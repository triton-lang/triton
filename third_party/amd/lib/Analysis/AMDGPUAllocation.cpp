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

unsigned getConvertLayoutScratchInBytes(RankedTensorType srcTy,
                                        RankedTensorType dstTy,
                                        TargetInfoBase &targetInfo) {
  if (!cvtNeedsSharedMemory(srcTy, dstTy))
    return 0;
  int numBanks = targetInfo.getSharedMemoryBanks();
  auto srcLayout = gpu::toLinearLayout(srcTy);
  auto dstLayout = gpu::toLinearLayout(dstTy);
  auto vecBitwidth =
      triton::gpu::getLdStVecBitwidth(srcLayout, dstLayout, getBitwidth(srcTy));
  auto [dstTile, srcTile] = targetInfo.getSharedLdStTiles(vecBitwidth);
  unsigned elems =
      getNumScratchElemsSwizzledCvt(srcTy, dstTy, numBanks, srcTile, dstTile);
  return elems * getBitwidth(srcTy) / 8;
}

static unsigned getBufferAtomicScratchSizeInBytes(Operation *op) {
  Value result = op->getResult(0);
  if (result.use_empty())
    return 0;
  auto tensorTy = dyn_cast<RankedTensorType>(result.getType());
  if (!tensorTy)
    return 0;
  auto freeVariableMasks = gpu::toLinearLayout(tensorTy).getFreeVariableMasks();
  bool hasBroadcast = llvm::any_of(freeVariableMasks,
                                   [](auto mask) { return mask.second != 0; });
  if (!hasBroadcast)
    return 0;
  auto smemShape = convertType<unsigned>(gpu::getShapePerCTA(tensorTy));
  auto elems = getNumScratchElements(smemShape);
  if (elems == 0)
    return 0;
  auto elemTy = tensorTy.getElementType();
  return elems * std::max<int>(8, elemTy.getIntOrFloatBitWidth()) / 8;
}

static unsigned getReduceScratchInBytes(ReduceOp op,
                                        TargetInfoBase &targetInfo) {
  auto srcTy = cast<RankedTensorType>(op.getOperands()[0].getType());
  auto axis = op.getAxis();
  auto kLane = StringAttr::get(op.getContext(), "lane");
  int numBanks = targetInfo.getSharedMemoryBanks();

  auto isReduced = [axis = axis](const LinearLayout &layout) {
    return layout.getOutDimSizes().begin()[axis] == 1;
  };
  auto regLl = ReduceOpHelper::reducedRegLaneLayout(srcTy, axis);

  // All the inputs have the same layout so, since we order them from largest
  // bitsize to smallest, and the first one is aligned, by induction, they are
  // all aligned, so we don't need to align the byte numbers returned here.
  unsigned bytesRegToTmp = 0;
  while (!isReduced(regLl)) {
    auto tmpLl = ReduceOpHelper::getInterLayout(regLl, axis);
    // We take the maximum of the elements and multiply by the total bitwidth.
    // We do this as otherwise it's quite tricky to find the correct
    // BaseOffsets in the lowering.
    int bytes = 0;
    for (auto inputTy : op.getInputTypes()) {
      auto vecBitwidth =
          triton::gpu::getLdStVecBitwidth(regLl, tmpLl, getBitwidth(inputTy));
      auto [dstTile, srcTile] = targetInfo.getSharedLdStTiles(vecBitwidth);
      auto nelem = getNumScratchElemsSwizzledCvt(
          regLl, tmpLl, getBitwidth(inputTy), numBanks, srcTile, dstTile);
      bytes += nelem * (getBitwidth(inputTy) / 8);
    }
    bytesRegToTmp = std::max<unsigned>(bytesRegToTmp, bytes);
    regLl = ReduceOpHelper::zeroBasesAlongDimAndReorder(tmpLl, axis, kLane);
  }
  return bytesRegToTmp;
}

unsigned AMDAllocationAnalysisScratchSizeFn(Operation *op,
                                            TargetInfoBase &targetInfo) {

  if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    return getReduceScratchInBytes(reduceOp, targetInfo);
  }

  if (auto cvtLayout = dyn_cast<mlir::triton::gpu::ConvertLayoutOp>(op)) {
    auto srcTy = cvtLayout.getSrc().getType();
    auto dstTy = cvtLayout.getType();
    return getConvertLayoutScratchInBytes(srcTy, dstTy, targetInfo);
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
    // ConSan adds captures after allocation; reserve space pre-computed by
    // the PrepareConSanCaptures pass.
    if (auto extra =
            ws->getAttrOfType<IntegerAttr>("consan.extra_capture_bytes"))
      captureSize += extra.getInt();
    return captureSize;
  }

  if (isa<amdgpu::BufferAtomicCASOp, amdgpu::BufferAtomicRMWOp>(op))
    return getBufferAtomicScratchSizeInBytes(op);

  return defaultAllocationAnalysisScratchSizeFn(op);
}

} // namespace mlir::triton::AMD
