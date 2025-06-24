#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {

using namespace mlir;

struct PreserveOpConversion
    : public ConvertOpToLLVMPattern<triton::PreserveOp> {
  explicit PreserveOpConversion(LLVMTypeConverter &typeConverter,
                                const TargetInfoBase &targetInfo,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::PreserveOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::PreserveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto val = adaptor.getValue();

    // Convert to inline assembly that acts like a black box - prevents
    // optimization from eliminating the value while being a no-op at runtime
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto funcType = LLVM::LLVMFunctionType::get(voidType, {val.getType()});

    rewriter.create<LLVM::InlineAsmOp>(
        loc, voidType, ValueRange{val},
        /*asm_string=*/StringAttr::get(rewriter.getContext(), ""),
        /*constraints=*/StringAttr::get(rewriter.getContext(), "r,~{memory}"),
        /*has_side_effects=*/UnitAttr::get(rewriter.getContext()),
        /*is_align_stack=*/nullptr,
        /*tail_call_kind=*/LLVM::TailCallKindAttr(),
        /*asm_dialect=*/
        LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                  LLVM::AsmDialect::AD_ATT),
        /*operand_attrs=*/ArrayAttr());

    rewriter.eraseOp(op);

    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populatePreserveOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<PreserveOpConversion>(typeConverter, targetInfo, benefit);
}
