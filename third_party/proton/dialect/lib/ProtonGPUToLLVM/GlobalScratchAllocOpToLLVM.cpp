#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/PatternProtonOpToLLVM.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/Utility.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "third_party/proton/dialect/include/Dialect/ProtonGPU/IR/Dialect.h"

namespace {
using namespace mlir;
using namespace mlir::triton;

struct GlobalScratchAllocOpConversion
    : public ConvertOpToLLVMPattern<proton::gpu::GlobalScratchAllocOp> {
  explicit GlobalScratchAllocOpConversion(LLVMTypeConverter &typeConverter,
                                          const TargetInfoBase &targetInfo,
                                          PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<proton::gpu::GlobalScratchAllocOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(proton::gpu::GlobalScratchAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.eraseOp(op);
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto ctx = funcOp->getContext();
    auto loc = funcOp.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto gmemBase = funcOp.getArgument(funcOp.getNumArguments() - 1);
    // TODO: implement this offset value
    auto opOffset = 0;
    auto llvmPointerType = LLVM::LLVMPointerType::get(ctx);
    rewriter.create<LLVM::GEPOp>(loc, llvmPointerType, llvmPointerType,
                                 gmemBase, b.i32_val(opOffset));
    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::proton::populateGlobalScratchAllocOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<GlobalScratchAllocOpConversion>(typeConverter, targetInfo,
                                               benefit);
}
