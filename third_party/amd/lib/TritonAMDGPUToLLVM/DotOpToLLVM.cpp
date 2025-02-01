#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

using namespace mlir;

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;

namespace mlir::triton::AMD {
LogicalResult convertAMDFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                               const LLVMTypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter);

LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);

LogicalResult convertWMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);
} // namespace mlir::triton::AMD

namespace {
struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value D = op.getResult();

    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();
    if (isa<AMDMfmaEncodingAttr>(dEncoding)) {
      return AMD::convertMFMA(op, adaptor, getTypeConverter(), rewriter);
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
};
} // namespace

namespace mlir::triton::AMD {
void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns, int numWarps,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
