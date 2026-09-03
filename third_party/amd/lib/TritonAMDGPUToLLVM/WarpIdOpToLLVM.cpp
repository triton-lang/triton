#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

class WarpIdOpPattern : public ConvertOpToLLVMPattern<WarpIdOp> {
public:
  WarpIdOpPattern(LLVMTypeConverter &typeConverter,
                  const AMD::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<WarpIdOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(WarpIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // These are runtime constant values so insert ops at the beginning of the
    // function to help LLVM uniformity analysis, unless we are in a warp
    // specialized partition region where we need to keep ops in their
    // respective regions.
    std::optional<int> startWarpId = getWarpGroupStartWarpId(op->getBlock());
    if (!startWarpId) {
      auto funcOp = op->getParentOfType<FunctionOpInterface>();
      rewriter.setInsertionPoint(
          &funcOp.getFunctionBody().getBlocks().front().front());
    }

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value warpId;

    if (targetInfo.supportsWaveId()) {
      warpId = ROCDL::WaveId::create(rewriter, loc, i32_ty);
    } else {
      int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
      Value warpSizeVal = b.i32_val(threadsPerWarp);
      Value tid = getThreadId(rewriter, loc);
      warpId = b.udiv(tid, warpSizeVal);

      // `warpId` is derived from the thread id, so LLVM's uniformity analysis
      // conservatively treats it as divergent even though every lane in a wave
      // computes the same value. `v_readfirstlane` moves it into an SGPR and
      // lets LLVM propagate uniformity to dependent ops, selecting SALU/SGPRs
      // instead of VALU/VGPRs.
      auto call =
          ROCDL::ReadfirstlaneOp::create(rewriter, loc, {i32_ty}, warpId);
      warpId = call.getRes();
    }

    if (startWarpId) {
      warpId = b.sub(warpId, b.i32_val(*startWarpId));
    }

    rewriter.replaceOp(op, warpId);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};
} // namespace

void mlir::triton::AMD::populateWarpIdOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<WarpIdOpPattern>(typeConverter, targetInfo, benefit);
}
