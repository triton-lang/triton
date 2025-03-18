#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/NvidiaPatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
class ReadCounterOpConversion
    : public ConvertOpToLLVMPattern<proton::gpu::ReadCounterOp> {
public:
  explicit ReadCounterOpConversion(LLVMTypeConverter &typeConverter,
                                   const proton::NVIDIA::TargetInfo &targetInfo,
                                   PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<proton::gpu::ReadCounterOp>(typeConverter,
                                                                 benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(proton::gpu::ReadCounterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value clock = targetInfo.clock(rewriter, op.getLoc(), false);
    rewriter.replaceOp(op, {clock});
    return success();
  }

private:
  const proton::NVIDIA::TargetInfo &targetInfo;
};
} // namespace

namespace mlir::triton::proton::NVIDIA {
void populateReadCounterOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         const TargetInfo &targetInfo,
                                         PatternBenefit benefit) {
  patterns.add<ReadCounterOpConversion>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::proton::NVIDIA
