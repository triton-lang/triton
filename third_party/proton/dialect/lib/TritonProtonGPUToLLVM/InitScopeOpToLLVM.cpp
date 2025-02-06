#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "third_party/proton/dialect/include/TritonProtonToLLVM/PatternTritonProtonOpToLLVM.h"

namespace {

struct InitScopeOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::InitScopeOp> {
  explicit InitScopeOpConversion(LLVMTypeConverter &typeConverter,
                                 PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::InitScopeOp>(
            typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::InitScopeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // TODO: Use an allocate scope id pass to attach each scope id op an
    // attribute
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto i0 = b.i32_val(0);
    rewriter.replaceOp(op, i0);
    return success();
  }
};

} // namespace

void mlir::triton::proton::populateInitScopeOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<InitScopeOpConversion>(typeConverter, benefit);
}
