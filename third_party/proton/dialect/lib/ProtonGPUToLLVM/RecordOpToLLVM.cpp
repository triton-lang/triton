#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/RecordOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {

struct RecordOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::RecordOp> {
  explicit RecordOpConversion(LLVMTypeConverter &typeConverter,
                              const proton::gpu::TargetInfoBase &targetInfo,
                              PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::RecordOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::RecordOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::proton::gpu::populateRecordOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<RecordOpConversion>(typeConverter, targetInfo, benefit);
}
