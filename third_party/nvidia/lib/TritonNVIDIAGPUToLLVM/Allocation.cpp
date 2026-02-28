#include <memory>

#include "Allocation.h"
#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ALLOCATESHAREDMEMORYNV
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {
struct AllocateSharedMemoryNv
    : public mlir::triton::impl::AllocateSharedMemoryNvBase<
          AllocateSharedMemoryNv> {
  using AllocateSharedMemoryNvBase::AllocateSharedMemoryNvBase;

  AllocateSharedMemoryNv(int32_t computeCapability, int32_t ptxVersion)
      : AllocateSharedMemoryNvBase({computeCapability, ptxVersion}) {}

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mlir::triton::NVIDIA::TargetInfo targetInfo(computeCapability, ptxVersion);
    ModuleAllocation allocation(
        mod, mlir::triton::nvidia_gpu::getNvidiaAllocationAnalysisScratchSizeFn(
                 targetInfo));
    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};
} // namespace

namespace mlir::triton::nvidia_gpu {

static unsigned getNumScratchElemsSwizzledCvt(RankedTensorType srcTy,
                                              RankedTensorType dstTy,
                                              TargetInfoBase &targetInfo) {
  auto *ctx = srcTy.getContext();
  auto srcLayout = triton::gpu::toLinearLayout(srcTy);
  auto dstLayout = triton::gpu::toLinearLayout(dstTy);
  srcLayout = actionRemoveBroadcastedRegs(srcLayout).apply(srcLayout);
  dstLayout = actionRemoveBroadcastedRegs(dstLayout).apply(dstLayout);
  auto bitwidth = getBitwidth(srcTy);
  auto [srcTiles, dstTiles] = gpu::getSrcDstTiles(targetInfo, bitwidth);
  auto [smem, _] = triton::gpu::optimalSwizzling(srcLayout, dstLayout, srcTiles,
                                                 dstTiles, bitwidth);
  auto reps = smem.getInDimSize(StringAttr::get(ctx, "reps"));
  return smem.getTotalOutDimSize() / reps;
}

std::function<unsigned(Operation *)>
getNvidiaAllocationAnalysisScratchSizeFn(TargetInfoBase &targetInfo) {
  auto allocation = [&targetInfo](Operation *op) -> unsigned {
    if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      auto srcTy = cvtOp.getSrc().getType();
      auto dstTy = cvtOp.getType();
      if (!cvtNeedsSharedMemory(srcTy, dstTy))
        return 0;
      // In cuda we always swizzle
      auto elems = getNumScratchElemsSwizzledCvt(srcTy, dstTy, targetInfo);
      return elems * getBitwidth(srcTy) / 8;
    }
    return defaultAllocationAnalysisScratchSizeFn(op);
  };
  return allocation;
}
} // namespace mlir::triton::nvidia_gpu

namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>>
createAllocateSharedMemoryNvPass(int32_t computeCapability,
                                 int32_t ptxVersion) {
  return std::make_unique<AllocateSharedMemoryNv>(computeCapability,
                                                  ptxVersion);
}
} // namespace mlir::triton
