#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace {

struct FuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  FuncOpConversion(LLVMTypeConverter &converter,
                   const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prevent LLVM's inliner to inline this function
    auto amendedFuncOp = amendFuncOp(funcOp, rewriter, targetInfo);
    FailureOr<LLVM::LLVMFuncOp> maybeNewFuncOp =
        mlir::convertFuncOpToLLVMFuncOp(amendedFuncOp, rewriter,
                                        *getTypeConverter());
    if (failed(maybeNewFuncOp)) {
      return failure();
    }

    auto ctx = funcOp->getContext();
    LLVM::LLVMFuncOp newFuncOp = *maybeNewFuncOp;
    handleArgPtrDatatype(funcOp, newFuncOp);

    if (triton::isKernel(funcOp)) {
      newFuncOp.setLinkage(LLVM::Linkage::External);
    } else {
      SmallVector<Attribute> passthroughAttrs = {
          rewriter.getStringAttr("noinline")};
      if (funcOp->hasAttrOfType<UnitAttr>("always_use_warp_shuffle"))
        passthroughAttrs.push_back(rewriter.getStringAttr("convergent"));
      newFuncOp.setPassthroughAttr(ArrayAttr::get(ctx, passthroughAttrs));
      newFuncOp.setLinkage(LLVM::Linkage::Internal);
      if (Attribute numWarps = funcOp->getAttr(triton::gpu::AttrNumWarpsName))
        newFuncOp->setAttr("ws_num_warps", numWarps);
    }

    rewriter.eraseOp(funcOp);
    rewriter.eraseOp(amendedFuncOp);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::AMD::populateFuncOpConversionPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<FuncOpConversion>(typeConverter, targetInfo, benefit);
}
