//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "xpu/lib/Conversion/TritonXPUToLLVM/PatternTritonXPUOpToLLVM.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using ::mlir::triton::gpu::getTotalElemsPerThread;

struct XPUExtractOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::ExtractOp> {

  XPUExtractOpConversion(LLVMTypeConverter &converter,
                         const xpu::TargetInfo &targetInfo,
                         ModuleAxisInfoAnalysis &axisAnalysisPass,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::ExtractOp>(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::xpu::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // original values
    auto index = op.getIndex();

    // adaptor values
    auto llTensor = adaptor.getTensor();

    // Get the LLVM values
    auto llTensors = unpackLLElements(loc, llTensor, rewriter);

    // TODO[dyq]: necessary?
    // assert(index >= 0 && index < llTensors.size() &&
    //        "Get Invalid Index For triton::xpu::ExtractOp");

    // Modifition Logic
    rewriter.replaceOp(op, {llTensors[index]});
    return success();
  };
};

struct XPUExtractSliceOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::ExtractSliceOp> {

  XPUExtractSliceOpConversion(LLVMTypeConverter &converter,
                              const xpu::TargetInfo &targetInfo,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::ExtractSliceOp>(converter,
                                                            benefit) {}

  LogicalResult
  matchAndRewrite(triton::xpu::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();
    auto llTensor = adaptor.getTensor();
    auto llTensors = unpackLLElements(loc, llTensor, rewriter);

    auto resType = op.getResult().getType();
    auto rankedTy = mlir::dyn_cast<RankedTensorType>(resType);
    unsigned elems = getTotalElemsPerThread(resType);
    SmallVector<Value> retVals(elems);
    for (unsigned i = 0; i < elems; ++i) {
      retVals[i] = llTensors[i];
    }
    Type llvmResultStructTy = getTypeConverter()->convertType(resType);
    Value resultStruct = packLLElements(loc, getTypeConverter(), retVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  };
};

struct XPUGetThreadIdOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::GetThreadIdOp> {

  XPUGetThreadIdOpConversion(LLVMTypeConverter &converter,
                             const xpu::TargetInfo &targetInfo,
                             ModuleAxisInfoAnalysis &axisAnalysisPass,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::GetThreadIdOp>(converter, benefit) {
  }

  LogicalResult
  matchAndRewrite(triton::xpu::GetThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    auto threadType = op.getThreadType();
    auto resType = op.getResult().getType();
    auto rankedTy = mlir::dyn_cast<RankedTensorType>(resType);
    unsigned elems = getTotalElemsPerThread(resType);

    SmallVector<Value> retVals(elems);
    Value clusterNum = mlir::LLVM::XPU::getGridDim(rewriter, loc);
    Value coreNum = mlir::LLVM::XPU::getBlockDim(rewriter, loc);
    Value clusterId = mlir::LLVM::XPU::getBlockId(rewriter, loc);
    Value coreId = getThreadId(rewriter, loc);
    Value threadId;
    switch (threadType) {
    case 0: {
      // tid = core_id() * cluster_num() + cluster_id()
      threadId = add(mul(coreId, clusterNum), clusterId);
      break;
    }
    case 1: {
      // tid = core_num() * cluster_id() + core_id()
      threadId = add(mul(coreNum, clusterId), coreId);
      break;
    }
    default:
      llvm_unreachable("Unknown threadId Type");
    }

    for (unsigned i = 0; i < elems; ++i) {
      retVals[i] = threadId;
    }

    Type llvmResultStructTy = getTypeConverter()->convertType(resType);
    Value resultStruct = packLLElements(loc, getTypeConverter(), retVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  };
};

struct XPUGetClusterIdOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::GetClusterIdOp> {

  XPUGetClusterIdOpConversion(LLVMTypeConverter &converter,
                              const xpu::TargetInfo &targetInfo,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::GetClusterIdOp>(converter,
                                                            benefit) {}

  LogicalResult
  matchAndRewrite(triton::xpu::GetClusterIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Value retVal = rewriter.create<mlir::LLVM::XPU::LoadParamOp>(
        loc, type::i32Ty(ctx), i32_val(0));

    rewriter.replaceOp(op, {retVal});
    return success();
  };
};

struct XPUGetCoreIdOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::GetCoreIdOp> {

  XPUGetCoreIdOpConversion(LLVMTypeConverter &converter,
                           const xpu::TargetInfo &targetInfo,
                           ModuleAxisInfoAnalysis &axisAnalysisPass,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::GetCoreIdOp>(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::xpu::GetCoreIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Value retVal =
        rewriter.create<mlir::LLVM::XPU::CoreIdOp>(loc, type::i32Ty(ctx));

    rewriter.replaceOp(op, {retVal});
    return success();
  };
};

struct XPUGetNumClusterOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::GetNumClusterOp> {

  XPUGetNumClusterOpConversion(LLVMTypeConverter &converter,
                               const xpu::TargetInfo &targetInfo,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::GetNumClusterOp>(converter,
                                                             benefit) {}

  LogicalResult
  matchAndRewrite(triton::xpu::GetNumClusterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Value retVal = rewriter.create<mlir::LLVM::XPU::LoadParamOp>(
        loc, type::i32Ty(ctx), i32_val(1));

    rewriter.replaceOp(op, {retVal});
    return success();
  };
};

} // namespace

void mlir::triton::xpu::populateTTXPUUtilityOpToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns.add<XPUExtractOpConversion>(typeConverter, targetInfo,
                                       axisInfoAnalysis, benefit);
  patterns.add<XPUExtractSliceOpConversion>(typeConverter, targetInfo,
                                            axisInfoAnalysis, benefit);
  patterns.add<XPUGetCoreIdOpConversion>(typeConverter, targetInfo,
                                         axisInfoAnalysis, benefit);
  patterns.add<XPUGetThreadIdOpConversion>(typeConverter, targetInfo,
                                           axisInfoAnalysis, benefit);
  patterns.add<XPUGetClusterIdOpConversion>(typeConverter, targetInfo,
                                            axisInfoAnalysis, benefit);
  patterns.add<XPUGetNumClusterOpConversion>(typeConverter, targetInfo,
                                             axisInfoAnalysis, benefit);
}
