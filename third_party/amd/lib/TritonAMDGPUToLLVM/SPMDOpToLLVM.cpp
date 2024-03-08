#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct GetProgramIdOpConversion
    : public ConvertOpToLLVMPattern<triton::GetProgramIdOp> {
  using ConvertOpToLLVMPattern<triton::GetProgramIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                    mlir::gpu::Dimension::y,
                                                    mlir::gpu::Dimension::z};
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);

    Value blockId =
        rewriter.create<::mlir::gpu::BlockIdOp>(loc, dims[op.getAxisAsInt()]);
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, i32_ty, blockId);
    return success();
  }
};

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                    mlir::gpu::Dimension::y,
                                                    mlir::gpu::Dimension::z};
    Location loc = op->getLoc();
    assert(op.getAxis() < 3);
    Value blockId =
        rewriter.create<::mlir::gpu::GridDimOp>(loc, dims[op.getAxis()]);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i32_ty, blockId);
    return success();
  }
};

} // namespace

void AMD::populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      PatternBenefit benefit) {
  patterns.add<GetProgramIdOpConversion>(typeConverter, benefit);
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
}
