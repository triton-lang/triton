#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

struct WarpIdOpPattern : public ConvertOpToLLVMPattern<WarpIdOp> {
  explicit WarpIdOpPattern(LLVMTypeConverter &typeConverter,
                           const TargetInfoBase &targetInfo,
                           PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<WarpIdOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(WarpIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    if (triton::gpu::lookupNumWarps(op) == 1) {
      // If there is only one warp, the warp ID is always 0.
      rewriter.replaceOp(op, b.i32_val(0));
      return success();
    }

    // If this is inside a warp specialize op, compute the relative thread ID
    // within the warp group.
    Value tid = NVVM::ThreadIdXOp::create(rewriter, loc, i32_ty);
    if (std::optional<int> startId =
            getWarpGroupStartThreadId(rewriter.getInsertionBlock()))
      tid = LLVM::SubOp::create(rewriter, loc, tid, b.i32_val(*startId));

    Value warpId = b.udiv(tid, b.i32_val(32));
    // This indicates to PTXAS that the result and its derived values are
    // uniform across the warp. For example, if a branch condition derives from
    // this value, it can be proven to be non-divergent.
    warpId = shuffleCommon(loc, rewriter, warpId, b.i32_val(0),
                           NVVM::ShflKind::idx, b.i32_val(0x1f));
    rewriter.replaceOp(op, warpId);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateWarpIdOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<WarpIdOpPattern>(typeConverter, targetInfo, benefit);
}
