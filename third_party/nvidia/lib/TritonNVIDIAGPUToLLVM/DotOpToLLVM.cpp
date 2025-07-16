#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

LogicalResult convertMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                         const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, bool isTuring,
                         bool isHopperF64);

LogicalResult convertWGMMA(triton::nvidia_gpu::WarpGroupDotOp op,
                           triton::nvidia_gpu::WarpGroupDotOp::Adaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread);
namespace {
struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  DotOpConversion(LLVMTypeConverter &converter, int computeCapability,
                  PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DotOp>(converter, benefit),
        computeCapability(computeCapability) {}

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

    NvidiaMmaEncodingAttr mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(
        cast<RankedTensorType>(D.getType()).getEncoding());
    if (!isOuter && mmaLayout && supportMMA(op, mmaLayout.getVersionMajor())) {
      if (mmaLayout.getVersionMajor() == 2) {
        bool isHopperF64 =
            computeCapability == 90 &&
            cast<RankedTensorType>(A.getType()).getElementType().isF64();
        return convertMMA(op, adaptor, getTypeConverter(), rewriter,
                          mmaLayout.isTuring(), isHopperF64);
      }

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotOp to LLVM.");
    }

    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding()))
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }

private:
  int computeCapability;
};

struct WarpGroupDotOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::WarpGroupDotOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::WarpGroupDotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::WarpGroupDotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // D = A * B + C
    Value A = op.getA();
    TypedValue<RankedTensorType> D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    auto mmaLayout = cast<NvidiaMmaEncodingAttr>(D.getType().getEncoding());
    if (!isOuter && supportMMA(op.getOperand(0), mmaLayout.getVersionMajor())) {
      return convertWGMMA(op, adaptor, getTypeConverter(), rewriter,
                          getThreadId(rewriter, loc));
    }

    return op.emitError(
        "Unsupported WarpGroupDotOp found when converting TritonGPU to LLVM.");
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
    rewriter.create<NVVM::WgmmaWaitGroupSyncOp>(loc, pendings);
    if (adaptor.getInputs().empty()) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, adaptor.getInputs());
    }
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
}
