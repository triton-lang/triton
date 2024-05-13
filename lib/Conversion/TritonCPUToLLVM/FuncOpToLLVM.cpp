#include "mlir/Support/LogicalResult.h"
#include "triton/Conversion/TritonCPUToLLVM/PatternTritonCPUOpToLLVM.h"
#include "triton/Conversion/TritonCPUToLLVM/Utility.h"

namespace mlir {
FailureOr<LLVM::LLVMFuncOp>
convertFuncOpToLLVMFuncOp(FunctionOpInterface funcOp,
                          ConversionPatternRewriter &rewriter,
                          const LLVMTypeConverter &converter);
}

namespace {

using namespace mlir;
using namespace mlir::triton;

struct FuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  FuncOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!LLVM::isKernel(funcOp)) {
      llvm_unreachable("Not implemented");
    }

    LLVM::LLVMFuncOp newFuncOp =
        *mlir::convertFuncOpToLLVMFuncOp(funcOp, rewriter, *getTypeConverter());
    if (!newFuncOp) {
      return failure();
    }

    auto ctx = funcOp->getContext();
    if (LLVM::isKernel(funcOp)) {
      // Set an attribute to indicate this function is a kernel entry.
      newFuncOp->setAttr("cpu.kernel",
                         rewriter.getIntegerAttr(type::u1Ty(ctx), 1));
    } else {
      llvm_unreachable("Not implemented");
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

} // namespace

void mlir::triton::cpu::populateFuncOpConversionPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<FuncOpConversion>(typeConverter, benefit);
}
