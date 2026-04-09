#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_PREPARECONSANCAPTURES
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

namespace ttg = mlir::triton::gpu;
namespace ttag = mlir::triton::amdgpu;
namespace tti = mlir::triton::instrument;

// Pre-reserve LDS space for the WarpSpecialize captures that the
// ConcurrencySanitizer pass will add.
struct PrepareConSanCaptures
    : public mlir::triton::impl::PrepareConSanCapturesBase<
          PrepareConSanCaptures> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    bool hasSharedBufs = false;
    bool hasBarriers = false;
    bool needsTdm = false;
    bool needsAsyncCp = false;

    mod.walk([&](Operation *op) {
      if (isa<ttg::LocalAllocOp>(op))
        hasSharedBufs = true;
      if (isa<ttag::InitBarrierOp>(op))
        hasBarriers = true;
      if (isa<ttag::AsyncTDMCopyGlobalToLocalOp,
              ttag::AsyncTDMCopyLocalToGlobalOp, ttag::AsyncTDMWait,
              ttag::AsyncTDMIntrinsicWait>(op))
        needsTdm = true;
      if (isa<ttag::AsyncWaitOp, ttg::AsyncCopyGlobalToLocalOp,
              ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp>(op))
        needsAsyncCp = true;
    });

    int M = hasSharedBufs ? 1 : 0;
    int K = (needsTdm ? 1 : 0) + (needsAsyncCp ? 1 : 0);

    int totalCaptures = tti::estimateConSanCaptureCount(M, hasBarriers, K);
    int extraBytes = totalCaptures * tti::kCaptureSizeBytes;

    auto i32Ty = IntegerType::get(mod.getContext(), 32);
    mod.walk([&](ttg::WarpSpecializeOp ws) {
      ws->setAttr("consan.extra_capture_bytes",
                  IntegerAttr::get(i32Ty, extraBytes));
    });
  }
};

} // namespace
