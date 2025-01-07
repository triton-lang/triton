#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace mlir {
namespace triton {

namespace gpu {
namespace {
struct UndefOpConversion : public ConvertOpToLLVMPattern<UndefOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(UndefOp op, UndefOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = getTypeConverter()->convertType(op.getType());
    if (!type)
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, type);
    return success();
  }
};
} // namespace
} // namespace gpu

void populateUndefOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  patterns.insert<gpu::UndefOpConversion>(typeConverter);
}

} // namespace triton
} // namespace mlir
