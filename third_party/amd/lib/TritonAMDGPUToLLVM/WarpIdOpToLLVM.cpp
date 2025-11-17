#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "third_party/amd/include/TritonAMDGPUToLLVM/TargetUtils.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using mlir::triton::AMD::ISAFamily;

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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto isaFamily = targetInfo.getISAFamily();
    int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
    Value warpSizeVal = b.i32_val(threadsPerWarp);
    Value tid = getThreadId(rewriter, loc);
    Value warpId = b.udiv(tid, warpSizeVal);
    if (ISAFamily::CDNA3 == isaFamily || ISAFamily::CDNA4 == isaFamily) {
      auto call =
          ROCDL::ReadfirstlaneOp::create(rewriter, loc, {i32_ty}, warpId);
      warpId = call.getRes();
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
