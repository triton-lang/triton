#include "Patterns.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct RegAllocOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::nvidia_gpu::RegAllocOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::RegAllocOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::RegAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    rewriter.replaceOpWithNewOp<triton::nvgpu::RegAllocOp>(
        op, adaptor.getRegCount());
    return success();
  }
};

struct RegDeallocOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::nvidia_gpu::RegDeallocOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::RegDeallocOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::RegDeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    rewriter.replaceOpWithNewOp<triton::nvgpu::RegDeallocOp>(
        op, adaptor.getRegCount());
    return success();
  }
};
} // namespace

void mlir::triton::populateRegReallocOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    const ModuleAllocation &allocation, PatternBenefit benefit) {
  patterns.add<RegAllocOpConversion>(typeConverter, benefit);
  patterns.add<RegDeallocOpConversion>(typeConverter, benefit);
  return;
}
