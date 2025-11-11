#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using namespace mlir;

using ::mlir::triton::AMD::TargetInfo;
using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;

namespace mlir::triton::AMD {
LogicalResult convertAMDFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                               const LLVMTypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter);

LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter,
                          const TargetInfo &targetInfo);

LogicalResult convertScaledMFMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter,
                                const TargetInfo &targetInfo);

LogicalResult convertWMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);

LogicalResult convertScaledWMMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter);
} // namespace mlir::triton::AMD

namespace {
struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  DotOpConversion(LLVMTypeConverter &typeConverter,
                  const AMD::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value D = op.getResult();

    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();
    if (isa<AMDMfmaEncodingAttr>(dEncoding)) {
      return AMD::convertMFMA(op, adaptor, getTypeConverter(), rewriter,
                              targetInfo);
    }
    if (isa<AMDWmmaEncodingAttr>(dEncoding)) {
      return AMD::convertWMMA(op, adaptor, getTypeConverter(), rewriter);
    }

    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding()))
      return AMD::convertAMDFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }

  const AMD::TargetInfo &targetInfo;
};

struct ScaledDotOpConversion
    : public ConvertOpToLLVMPattern<triton::DotScaledOp> {

  ScaledDotOpConversion(LLVMTypeConverter &typeConverter,
                        const AMD::TargetInfo &targetInfo,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value D = op.getResult();

    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();

    if (isa<AMDMfmaEncodingAttr>(dEncoding)) {
      return AMD::convertScaledMFMA(op, adaptor, getTypeConverter(), rewriter,
                                    targetInfo);
    }
    if (isa<AMDWmmaEncodingAttr>(dEncoding)) {
      return AMD::convertScaledWMMA(op, adaptor, getTypeConverter(), rewriter);
    }

    llvm::report_fatal_error(
        "Unsupported DotScaleOp found when converting TritonGPU to LLVM.");
  }

  const AMD::TargetInfo &targetInfo;
};
} // namespace

namespace mlir::triton::AMD {
void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 const TargetInfo &targetInfo,
                                 PatternBenefit benefit) {
  patterns.add<DotOpConversion, ScaledDotOpConversion>(typeConverter,
                                                       targetInfo, benefit);
}
} // namespace mlir::triton::AMD
