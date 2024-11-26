#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;

namespace {

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
    assert(op.getAxisAsInt() < 3);
    Value blockId =
        rewriter.create<::mlir::gpu::GridDimOp>(loc, dims[op.getAxisAsInt()]);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i32_ty, blockId);
    return success();
  }
};

struct CondBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::condBarrierOp> {
  using ConvertOpToLLVMPattern<
      triton::amdgpu::condBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::condBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto i32ty = rewriter.getIntegerType(32);
    auto workIDX = rewriter.create<ROCDL::ThreadIdXOp>(loc, i32ty);
    auto constZero = rewriter.create<LLVM::ConstantOp>(
        loc, i32ty, IntegerAttr::get(i32ty, 0));
    auto constWarpSize = rewriter.create<LLVM::ConstantOp>(
        loc, i32ty, IntegerAttr::get(i32ty, 256));
    auto warpIDX = rewriter.create<LLVM::SDivOp>(loc, workIDX, constWarpSize);
    auto warpLow = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne,
                                                 warpIDX, constZero);
    auto pred = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                              warpLow, op->getOperand(0));

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterCondBarBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterCondBarBlock);
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, trueBlock, afterCondBarBlock);

    // conditional barrier
    rewriter.setInsertionPointToStart(trueBlock);
    rewriter.create<ROCDL::SBarrierOp>(loc);
    rewriter.create<LLVM::BrOp>(loc, afterCondBarBlock);

    rewriter.setInsertionPointToStart(afterCondBarBlock);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::AMD::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<CondBarrierOpConversion>(typeConverter, benefit);
}
