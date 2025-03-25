//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "xpu/lib/Conversion/TritonXPUToLLVM/PatternTritonXPUOpToLLVM.h"

namespace {

template <typename SourceOp, typename DestOp>
struct ElementwiseOpConversion
    : public mlir::triton::gpu::ElementwiseOpConversionBase<
          SourceOp, ElementwiseOpConversion<SourceOp, DestOp>> {

  using Base = mlir::triton::gpu::ElementwiseOpConversionBase<
      SourceOp, ElementwiseOpConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  SmallVector<DestOp>
  createDestOps(SourceOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter, Type elemTy,
                mlir::triton::gpu::MultipleOperandsRange operands,
                Location loc) const {
    return {rewriter.create<DestOp>(loc, elemTy, operands[0],
                                    adaptor.getAttributes().getValue())};
  }
};

template <typename TritonOp>
struct OpToExternCallConversion
    : public triton::gpu::ElementwiseOpConversionBase<
          TritonOp, OpToExternCallConversion<TritonOp>> {
  using Base = triton::gpu::ElementwiseOpConversionBase<
      TritonOp, OpToExternCallConversion<TritonOp>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit OpToExternCallConversion(LLVMTypeConverter &typeConverter,
                                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                                    StringRef externFuncName,
                                    PatternBenefit benefit)
      : Base::ElementwiseOpConversionBase(typeConverter, axisAnalysisPass,
                                          benefit),
        funcName(externFuncName) {}

  SmallVector<Value> createDestOps(TritonOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy,
                                   triton::gpu::MultipleOperandsRange operands,
                                   Location loc) const {
    Type funcType = triton::gpu::getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        triton::gpu::appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
    return {
        rewriter.create<LLVM::CallOp>(loc, funcOp, operands[0]).getResult()};
  }

private:
  StringRef funcName;
};

} // namespace

void mlir::triton::xpu::populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfo &targetInfo,
    PatternBenefit benefit) {

  patterns.add<OpToExternCallConversion<triton::PreciseSqrtOp>>(
      typeConverter, axisInfoAnalysis, "_ZN3xpu10__fsqrt_rnEf", benefit);

#define POPULATE_UNARY_OP(SRC_OP, DST_OP)                                      \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit);
  POPULATE_UNARY_OP(arith::NegFOp, LLVM::FNegOp)
  POPULATE_UNARY_OP(arith::ExtFOp, LLVM::FPExtOp)
  POPULATE_UNARY_OP(arith::TruncFOp, LLVM::FPTruncOp)
  POPULATE_UNARY_OP(arith::SIToFPOp, LLVM::SIToFPOp)
  POPULATE_UNARY_OP(arith::FPToSIOp, LLVM::FPToSIOp)
  POPULATE_UNARY_OP(math::ExpOp, LLVM::Exp2Op)
  POPULATE_UNARY_OP(math::LogOp, LLVM::Log2Op)
#undef POPULATE_UNARY_OP

#define POPULATE_BINARY_OP(SRC_OP, DST_OP)                                     \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit);
  POPULATE_BINARY_OP(arith::AddFOp, LLVM::FAddOp)        // addf
  POPULATE_BINARY_OP(arith::SubFOp, LLVM::FSubOp)        // subf
  POPULATE_BINARY_OP(arith::MulFOp, LLVM::FMulOp)        // mulf
  POPULATE_BINARY_OP(arith::DivFOp, LLVM::FDivOp)        // divf
  POPULATE_BINARY_OP(arith::MaximumFOp, LLVM::MaximumOp) // maximum
  POPULATE_BINARY_OP(arith::MinimumFOp, LLVM::MinimumOp) // minimum
  POPULATE_BINARY_OP(triton::PreciseDivFOp, LLVM::FDivOp)
#undef POPULATE_BINARY_OP
}
