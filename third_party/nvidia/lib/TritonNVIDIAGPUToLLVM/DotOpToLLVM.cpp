#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::isPermutationMatrixLayout;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::toLinearLayout;

LogicalResult convertMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                         const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, bool isTuring);

LogicalResult convertMMADotScaled(triton::DotScaledOp op,
                                  triton::DotScaledOp::Adaptor adaptor,
                                  const LLVMTypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter);

LogicalResult convertWGMMA(triton::nvidia_gpu::WarpGroupDotOp op,
                           triton::nvidia_gpu::WarpGroupDotOp::Adaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread);

namespace {
struct ScaledDotOpConversion
    : public ConvertOpToLLVMPattern<triton::DotScaledOp> {
  using ConvertOpToLLVMPattern<triton::DotScaledOp>::ConvertOpToLLVMPattern;

  ScaledDotOpConversion(LLVMTypeConverter &converter, int,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DotScaledOp>(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, triton::DotScaledOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rty = cast<RankedTensorType>(op.getResult().getType());
    if (!isPermutationMatrixLayout(
            toLinearLayout(rty.getShape(), rty.getEncoding())))
      return rewriter.notifyMatchFailure(
          op, "ScaledDotOp result encoding must have a permutation-matrix "
              "linear layout");
    return convertMMADotScaled(op, adaptor, getTypeConverter(), rewriter);
  }
};

struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  DotOpConversion(LLVMTypeConverter &converter, int, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DotOp>(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // D = A * B + C
    auto dType = op.getResult().getType();
    auto dEncoding = dType.getEncoding();

    if (!isPermutationMatrixLayout(toLinearLayout(dType.getShape(), dEncoding)))
      return rewriter.notifyMatchFailure(
          op,
          "DotOp result encoding must have a permutation-matrix linear layout");

    NvidiaMmaEncodingAttr mmaLayout =
        dyn_cast<NvidiaMmaEncodingAttr>(dEncoding);
    if (mmaLayout) {
      if (mmaLayout.getVersionMajor() == 2) {
        return convertMMA(op, adaptor, getTypeConverter(), rewriter,
                          mmaLayout.isTuring());
      }

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotOp to LLVM.");
    }

    if (isa<BlockedEncodingAttr>(dEncoding))
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};

struct WarpGroupDotOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::WarpGroupDotOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::WarpGroupDotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::WarpGroupDotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    return convertWGMMA(op, adaptor, getTypeConverter(), rewriter,
                        getThreadId(rewriter, loc));
  }
};

struct WarpGroupDotWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::WarpGroupDotWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::WarpGroupDotWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::WarpGroupDotWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pendings = op.getPendings();
    Location loc = op.getLoc();
    ValueRange inputs = adaptor.getInputs();
    if (inputs.size() == 1) {
      rewriter.replaceOpWithNewOp<triton::nvgpu::WGMMAWaitGroupOp>(
          op, inputs.front(), pendings);
      return success();
    }
    SmallVector<Type> types;
    // Pack the inputs into a single struct.
    for (Type type : inputs.getTypes()) {
      auto structType = dyn_cast<LLVM::LLVMStructType>(type);
      if (!structType)
        return failure();
      llvm::append_range(types, structType.getBody());
    }
    auto packedType =
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
    Value packed = LLVM::UndefOp::create(rewriter, loc, packedType);
    unsigned outputStructIndex = 0;
    for (Value input : inputs) {
      for (auto [i, type] : llvm::enumerate(
               cast<LLVM::LLVMStructType>(input.getType()).getBody())) {
        Value value = LLVM::ExtractValueOp::create(rewriter, loc, input, i);
        packed = LLVM::InsertValueOp::create(rewriter, loc, packedType, packed,
                                             value, outputStructIndex++);
      }
    }
    Value packedOutput = triton::nvgpu::WGMMAWaitGroupOp::create(
        rewriter, loc, packed, pendings);
    // Unpack the output into the original struct types.
    SmallVector<Value> outputs;
    outputStructIndex = 0;
    for (Type type : inputs.getTypes()) {
      auto structType = cast<LLVM::LLVMStructType>(type);
      Value unpacked = LLVM::UndefOp::create(rewriter, loc, structType);
      for (auto [i, type] : llvm::enumerate(structType.getBody())) {
        Value value = LLVM::ExtractValueOp::create(rewriter, loc, packedOutput,
                                                   outputStructIndex++);
        unpacked = LLVM::InsertValueOp::create(rewriter, loc, structType,
                                               unpacked, value, i);
      }
      outputs.push_back(unpacked);
    }
    rewriter.replaceOp(op, outputs);
    return success();
  }
};
} // namespace

void mlir::triton::NVIDIA::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int computeCapability, PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, computeCapability, benefit);
  patterns.add<WarpGroupDotOpConversion>(typeConverter, benefit);
  patterns.add<WarpGroupDotWaitOpConversion>(typeConverter, benefit);
  patterns.add<ScaledDotOpConversion>(typeConverter, computeCapability,
                                      benefit);
}
