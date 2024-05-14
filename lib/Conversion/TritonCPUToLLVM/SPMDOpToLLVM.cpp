#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonCPUToLLVM/PatternTritonCPUOpToLLVM.h"
#include "triton/Conversion/TritonCPUToLLVM/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

struct GetProgramIdOpConversion
    : public ConvertOpToLLVMPattern<triton::GetProgramIdOp> {
  explicit GetProgramIdOpConversion(LLVMTypeConverter &typeConverter,
                                    const CPUTargetInfo &targetInfo,
                                    PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::GetProgramIdOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value programId = targetInfo.programId(
        rewriter, op->getLoc(), op->getParentOfType<LLVM::LLVMFuncOp>(),
        op.getAxisAsInt());
    rewriter.replaceOp(op, programId);
    return success();
  }

private:
  const CPUTargetInfo &targetInfo;
};

} // namespace

void mlir::triton::cpu::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const CPUTargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<GetProgramIdOpConversion>(typeConverter, targetInfo, benefit);
}
