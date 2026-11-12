#include "TargetInfo.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using namespace mlir;

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::isPermutationMatrixLayout;
using ::mlir::triton::gpu::toLinearLayout;

namespace mlir::triton::AMD {
LogicalResult convertAMDFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                               const LLVMTypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter);

LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);

LogicalResult convertScaledMFMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter);

LogicalResult convertWMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const TargetInfo &targetInfo,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);

LogicalResult convertScaledWMMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const TargetInfo &targetInfo,
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
    // D = A * B + C
    auto dType = op.getD().getType();
    auto dEncoding = dType.getEncoding();

    // WMMA is the only path that has been validated against non-permutation
    // (e.g. swizzled-warp) encodings, so handle it up front and then enforce
    // the permutation-matrix invariant for all other paths.
    if (isa<AMDWmmaEncodingAttr>(dEncoding))
      return AMD::convertWMMA(op, adaptor, targetInfo, getTypeConverter(),
                              rewriter);

    if (!isPermutationMatrixLayout(toLinearLayout(dType.getShape(), dEncoding)))
      return rewriter.notifyMatchFailure(op,
                                         "Non-WMMA DotOp result encoding must "
                                         "have a permutation-matrix linear "
                                         "layout");

    if (isa<AMDMfmaEncodingAttr>(dEncoding)) {
      return AMD::convertMFMA(op, adaptor, getTypeConverter(), rewriter);
    }

    if (isa<BlockedEncodingAttr>(dEncoding))
      return AMD::convertAMDFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }

private:
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
    auto dType = op.getD().getType();
    auto dEncoding = dType.getEncoding();

    if (isa<AMDWmmaEncodingAttr>(dEncoding))
      return AMD::convertScaledWMMA(op, adaptor, targetInfo, getTypeConverter(),
                                    rewriter);

    if (!isPermutationMatrixLayout(toLinearLayout(dType.getShape(), dEncoding)))
      return rewriter.notifyMatchFailure(
          op, "non-WMMA dot encoding must have a permutation-matrix linear "
              "layout");

    if (isa<AMDMfmaEncodingAttr>(dEncoding))
      return AMD::convertScaledMFMA(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotScaleOp found when converting TritonGPU to LLVM.");
  }

private:
  const AMD::TargetInfo &targetInfo;
};
} // namespace

namespace mlir::triton::AMD {
void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfo &targetInfo,
                                 PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<ScaledDotOpConversion>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::AMD
