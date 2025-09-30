#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;

namespace {

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                    mlir::gpu::Dimension::y,
                                                    mlir::gpu::Dimension::z};
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);
    Value blockId =
        rewriter.create<::mlir::gpu::GridDimOp>(loc, dims[op.getAxisAsInt()]);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i32_ty, blockId);
    return success();
  }
};

struct CondBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::CondBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::CondBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterCondBarBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterCondBarBlock);
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, adaptor.getPred(), trueBlock,
                                    afterCondBarBlock);

    // conditional barrier
    rewriter.setInsertionPointToStart(trueBlock);
    rewriter.create<ROCDL::SBarrierOp>(loc);
    rewriter.create<LLVM::BrOp>(loc, afterCondBarBlock);
    rewriter.eraseOp(op);
    return success();
  }
};

struct WaveIdOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::WaveIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::WaveIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<ModuleOp>();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value id = getThreadId(rewriter, loc);
    int waveSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    Value waveId = LLVM::createLLVMIntrinsicCallOp(
                       rewriter, loc, "llvm.amdgcn.readfirstlane", {i32_ty},
                       {b.udiv(id, b.i32_val(waveSize))})
                       ->getResult(0);
    rewriter.replaceOp(op, waveId);
    return success();
  }
};

} // namespace

void mlir::triton::AMD::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion, WaveIdOpConversion>(typeConverter,
                                                               benefit);
  patterns.add<CondBarrierOpConversion>(typeConverter, benefit);
}
