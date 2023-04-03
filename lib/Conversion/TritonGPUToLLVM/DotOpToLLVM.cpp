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
#ifdef USE_ROCM
      if (mmaLayout.isMI200()) {
        return convertMFMA(op, adaptor, rewriter);
      }
#endif
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

private:
// TODO move to separate file
  LogicalResult convertMFMA(triton::DotOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto mmaLayout = op.getResult()
                         .getType()
                         .cast<RankedTensorType>()
                         .getEncoding()
                         .cast<MmaEncodingAttr>();

    Value A = op.getA();
    Value B = op.getB();
    Value C = op.getC();

    DotOpMFMAConversionHelper helper(A.getType(), mmaLayout,
                                     getThreadId(rewriter, loc), rewriter,
                                     getTypeConverter(), loc);

    auto ATensorTy = A.getType().cast<RankedTensorType>();
    auto BTensorTy = B.getType().cast<RankedTensorType>();

    assert(ATensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
           BTensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
           "Both $a and %b should be DotOperand layout.");

    Value loadedA, loadedB, loadedC;
    loadedA = adaptor.getA();
    loadedB = adaptor.getB();
    loadedC = helper.loadC(op.getC(), adaptor.getC());

    return helper.convertDot(A, B, C, op.getD(), loadedA, loadedB, loadedC, op,
                             adaptor);
  }
};

void populateDotOpToLLVMPatterns(TritonGPUToLLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns, int numWarps,
                                 AxisInfoAnalysis &axisInfoAnalysis,
                                 const Allocation *allocation, Value smem,
                                 PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, allocation, smem, benefit);
}
