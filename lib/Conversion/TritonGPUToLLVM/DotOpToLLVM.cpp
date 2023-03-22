#include "DotOpToLLVM.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MmaEncodingAttr;

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonGPUToLLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter);

LogicalResult convertMMA884(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonGPUToLLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter);

LogicalResult convertMMA16816(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                              TritonGPUToLLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter);

struct DotOpConversion : public ConvertTritonGPUOpToLLVMPattern<triton::DotOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::DotOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShape = A.getType().cast<RankedTensorType>().getShape();
    size_t reduceAxis = 1;
    unsigned K = AShape[reduceAxis];
    bool isOuter = K == 1;

    MmaEncodingAttr mmaLayout = D.getType()
                                    .cast<RankedTensorType>()
                                    .getEncoding()
                                    .dyn_cast<MmaEncodingAttr>();
    if (!isOuter && mmaLayout && supportMMA(op, mmaLayout.getVersionMajor())) {
      if (mmaLayout.isVolta())
        return convertMMA884(op, adaptor, getTypeConverter(), rewriter);
      if (mmaLayout.isAmpere())
        return convertMMA16816(op, adaptor, getTypeConverter(), rewriter);

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotOp to LLVM.");
    }

    if (D.getType()
            .cast<RankedTensorType>()
            .getEncoding()
            .isa<BlockedEncodingAttr>())
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};

void populateDotOpToLLVMPatterns(TritonGPUToLLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns, int numWarps,
                                 AxisInfoAnalysis &axisInfoAnalysis,
                                 const Allocation *allocation, Value smem,
                                 PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, allocation, smem, benefit);
}
