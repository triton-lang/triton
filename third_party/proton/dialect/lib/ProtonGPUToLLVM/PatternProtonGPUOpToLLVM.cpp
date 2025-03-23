#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/GlobalScratchAllocOpToLLVM.h"
#include "third_party/proton/dialect/include/Dialect/ProtonGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton {
namespace proton::gpu {

namespace {
struct CircularStoreOpConversion
    : public ConvertOpToLLVMPattern<
          mlir::triton::proton::gpu::CircularStoreOp> {
  explicit CircularStoreOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<
            mlir::triton::proton::gpu::CircularStoreOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::CircularStoreOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct InitBufferIndexOpConversion
    : public ConvertOpToLLVMPattern<
          mlir::triton::proton::gpu::InitBufferIndexOp> {
  explicit InitBufferIndexOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<
            mlir::triton::proton::gpu::InitBufferIndexOp>(typeConverter,
                                                          benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::InitBufferIndexOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto zero = b.i32_val(0);
    rewriter.replaceOp(op, zero.getDefiningOp());
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct ReadCounterOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::ReadCounterOp> {
  explicit ReadCounterOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::ReadCounterOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::ReadCounterOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto zero = b.i32_val(0);
    rewriter.replaceOp(op, zero.getDefiningOp());
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct FinalizeOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::FinalizeOp> {
  explicit FinalizeOpConversion(LLVMTypeConverter &typeConverter,
                                const proton::gpu::TargetInfoBase &targetInfo,
                                PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::FinalizeOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::FinalizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

} // namespace

void populateProtonGPUOpPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit) {
  populateGlobalScratchAllocOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                            benefit);
  patterns.add<CircularStoreOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<InitBufferIndexOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<ReadCounterOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<FinalizeOpConversion>(typeConverter, targetInfo, benefit);
}

} // namespace proton::gpu
} // namespace mlir::triton
