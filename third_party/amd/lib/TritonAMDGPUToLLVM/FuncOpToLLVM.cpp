#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {

struct FuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  FuncOpConversion(LLVMTypeConverter &converter,
                   const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  static void handleArgPtrDatatype(triton::FuncOp funcOp,
                                   LLVM::LLVMFuncOp &llvmFuncOp) {

    // The convertion from triton::PointerType to LLVM::LLVMPointerType losts
    // the pointee datatype information.
    // This function add back the pointee datatype information to arg attribute.
    FunctionType fty = funcOp.getFunctionType();
    for (unsigned i = 0; i < fty.getNumInputs(); ++i) {
      auto argType = fty.getInput(i);
      if (auto argPtrType = dyn_cast<triton::PointerType>(argType)) {
        auto argDType = argPtrType.getPointeeType();
        llvmFuncOp.setArgAttr(i, "tt.pointee_type",
                              mlir::TypeAttr::get(argDType));
      }
    }
  }

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
      // Set attribute `noinline` to prevent inlining.
      newFuncOp.setPassthroughAttr(
          ArrayAttr::get(ctx, rewriter.getStringAttr("noinline")));
      newFuncOp.setLinkage(LLVM::Linkage::Internal);
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
