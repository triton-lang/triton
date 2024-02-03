#include "DotOpToLLVM.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu_rocm::DotOperandEncodingAttr;
using ::mlir::triton::gpu_rocm::getShapePerCTA;
using ::mlir::triton::gpu_rocm::MmaEncodingAttr;

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonGPUToLLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter);

LogicalResult convertMMA884(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonGPUToLLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter);

LogicalResult convertMMA1688(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                             TritonGPUToLLVMTypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter);

LogicalResult convertMMA16816(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                              TritonGPUToLLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter);

#if 1
LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          TritonGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);
#endif
LogicalResult convertWGMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                           TritonGPUToLLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread);

LogicalResult convertAsyncWGMMA(triton::nvidia_gpu::DotAsyncOp op,
                                triton::nvidia_gpu::DotAsyncOp::Adaptor adaptor,
                                TritonGPUToLLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter,
                                Value thread);

struct DotOpConversion : public ConvertTritonGPUOpToLLVMPattern<triton::DotOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::DotOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    MmaEncodingAttr mmaLayout = D.getType()
                                    .cast<RankedTensorType>()
                                    .getEncoding()
                                    .dyn_cast<MmaEncodingAttr>();
    if (!isOuter && mmaLayout && supportMMA(op, mmaLayout.getVersionMajor())) {
      if (mmaLayout.isVolta())
        return convertMMA884(op, adaptor, getTypeConverter(), rewriter);
      if (mmaLayout.isTuring())
        return convertMMA1688(op, adaptor, getTypeConverter(), rewriter);
      if (mmaLayout.isAmpere())
        return convertMMA16816(op, adaptor, getTypeConverter(), rewriter);
      if (mmaLayout.isHopper())
        return convertWGMMA(op, adaptor, getTypeConverter(), rewriter,
                            getThreadId(rewriter, loc));

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotOp to LLVM.");
    }

#if 1
    MfmaEncodingAttr mfmaLayout = D.getType()
                                      .cast<RankedTensorType>()
                                      .getEncoding()
                                      .dyn_cast<MfmaEncodingAttr>();
    if (!isOuter && mfmaLayout)) {
      return convertMFMA(op, adaptor, getTypeConverter(), rewriter);
    }
#endif

    if (D.getType()
            .cast<RankedTensorType>()
            .getEncoding()
            .isa<BlockedEncodingAttr>())
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};

struct DotAsyncOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::nvidia_gpu::DotAsyncOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::DotAsyncOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::DotAsyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    MmaEncodingAttr mmaLayout = D.getType()
                                    .cast<RankedTensorType>()
                                    .getEncoding()
                                    .dyn_cast<MmaEncodingAttr>();
    if (!isOuter && mmaLayout &&
        supportMMA(op.getOperand(0), mmaLayout.getVersionMajor())) {
      if (mmaLayout.isHopper()) {
        return convertAsyncWGMMA(op, adaptor, getTypeConverter(), rewriter,
                                 getThreadId(rewriter, loc));
      }

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotAsyncOp to LLVM.");
    }

    llvm::report_fatal_error(
        "Unsupported DotAsyncOp found when converting TritonGPU to LLVM.");
  }
};

struct DotWaitOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::nvidia_gpu::DotWaitOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::DotWaitOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::DotWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pendings = op.getPendings();
    rewriter.create<triton::nvgpu::WGMMAWaitGroupOp>(op.getLoc(), pendings);

    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

void populateDotOpToLLVMPatterns(TritonGPUToLLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns, int numWarps,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 ModuleAllocation &allocation,
                                 PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, allocation, benefit);
  patterns.add<DotAsyncOpConversion>(typeConverter, allocation, benefit);
  patterns.add<DotWaitOpConversion>(typeConverter, allocation, benefit);
}
