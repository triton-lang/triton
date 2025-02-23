#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/PatternProtonOpToLLVM.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "third_party/proton/dialect/include/Dialect/ProtonGPU/IR/Dialect.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/Utility.h"

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
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto loc = func.getLoc();
    auto ctx = func->getContext();
    auto globalPtrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto &region = func.getBody();
    region.addArgument(globalPtrTy, loc);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value bufferOffset = b.i32_val(0);
    Value ptr =
        triton::proton::gpu::getGlobalScratchPtr(loc, rewriter, func, bufferOffset);

    rewriter.replaceOp(op, ptr);    
    return success();
  }
protected:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::proton::populateGlobalScratchAllocOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<GlobalScratchAllocOpConversion>(typeConverter, targetInfo, benefit);
}
