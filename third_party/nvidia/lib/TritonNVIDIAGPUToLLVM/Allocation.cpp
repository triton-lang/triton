#include <memory>

#include "Allocation.h"
#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Dialect/TritonInstrument/IR/ConSanConstants.h"

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

std::function<unsigned(Operation *)>
getNvidiaAllocationAnalysisScratchSizeFn(TargetInfoBase &targetInfo) {
  auto allocation = [&targetInfo](Operation *op) -> unsigned {
    if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      return getConvertLayoutScratchBufferInfo(cvtOp,
                                               targetInfo.supportLdMatrix(),
                                               targetInfo.supportStMatrix())
          .size;
    }
    if (auto ws = dyn_cast<triton::gpu::WarpSpecializeOp>(op)) {
      unsigned captureSize = defaultAllocationAnalysisScratchSizeFn(op);
      // ConSan adds captures after allocation; reserve space pre-computed by
      // the common TritonInstrumentPrepareConSanCaptures pass.
      if (auto extra = ws->getAttrOfType<IntegerAttr>(
              mlir::triton::instrument::kConSanExtraCaptureBytesAttr))
        captureSize += extra.getInt();
      return captureSize;
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
