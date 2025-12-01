#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using namespace mlir;

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::LinearEncodingAttr;

namespace mlir::triton::AMD {
LogicalResult convertAMDFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                               const LLVMTypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter);

LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter, int mfmaVersion);

LogicalResult convertScaledMFMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter,
                                int mfmaVersion);

LogicalResult convertWMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);

LogicalResult convertScaledWMMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter);
} // namespace mlir::triton::AMD

namespace {
class DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

public:
  DotOpConversion(LLVMTypeConverter &typeConverter, int mfmaVersion,
                  PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        mfmaVersion(mfmaVersion) {}

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value D = op.getResult();

    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();
    if (isa<LinearEncodingAttr>(dEncoding)) {
      return AMD::convertMFMA(op, adaptor, getTypeConverter(), rewriter,
                              mfmaVersion);
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

private:
  int mfmaVersion;
};

class ScaledDotOpConversion
    : public ConvertOpToLLVMPattern<triton::DotScaledOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

public:
  ScaledDotOpConversion(LLVMTypeConverter &typeConverter, int mfmaVersion,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        mfmaVersion(mfmaVersion) {}

  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value D = op.getResult();

    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();

    if (isa<LinearEncodingAttr>(dEncoding)) {
      return AMD::convertScaledMFMA(op, adaptor, getTypeConverter(), rewriter,
                                    mfmaVersion);
    }
    if (isa<AMDWmmaEncodingAttr>(dEncoding)) {
      return AMD::convertScaledWMMA(op, adaptor, getTypeConverter(), rewriter);
    }

    llvm::report_fatal_error(
        "Unsupported DotScaleOp found when converting TritonGPU to LLVM.");
  }

private:
  int mfmaVersion;
};
} // namespace

namespace mlir::triton::AMD {
void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 int mfmaVersion, PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, mfmaVersion, benefit);
  patterns.add<ScaledDotOpConversion>(typeConverter, mfmaVersion, benefit);
}
} // namespace mlir::triton::AMD
