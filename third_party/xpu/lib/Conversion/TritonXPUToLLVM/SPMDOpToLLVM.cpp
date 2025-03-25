//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "xpu/lib/Conversion/TritonXPUToLLVM/PatternTritonXPUOpToLLVM.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct XPUGetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Value retVal = rewriter.create<mlir::LLVM::XPU::LoadParamOp>(
        loc, type::i32Ty(ctx), i32_val(1));

    rewriter.replaceOp(op, {retVal});
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<XPUGetNumProgramsOpConversion>(typeConverter, benefit);
}
