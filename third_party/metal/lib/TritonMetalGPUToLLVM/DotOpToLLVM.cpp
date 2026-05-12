#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton::gpu;

namespace mlir::triton::metal {
LogicalResult convertMetalFMADot(triton::DotOp op,
                                 triton::DotOp::Adaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter);
LogicalResult convertSimdgroupMatmul(
    triton::DotOp op, triton::DotOp::Adaptor adaptor,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    const DenseMap<int, std::array<Operation *, 2>> &dotAllocOps,
    const mlir::triton::metal::TargetInfo &targetInfo);
} // namespace mlir::triton::metal

namespace {
struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  explicit DotOpConversion(
      LLVMTypeConverter &typeConverter,
      const DenseMap<int, std::array<Operation *, 2>> &dotAllocOps,
      const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DotOp>(typeConverter, benefit),
        dotAllocOps(dotAllocOps), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // D = A * B + C
    Value D = op.getResult();
    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();

    if (isa<BlockedEncodingAttr>(dEncoding))
      return metal::convertSimdgroupMatmul(op, adaptor, getTypeConverter(),
                                           rewriter, dotAllocOps, targetInfo);

    // if (isa<BlockedEncodingAttr>(
    //         cast<RankedTensorType>(D.getType()).getEncoding()))
    //   return metal::convertMetalFMADot(op, adaptor, getTypeConverter(),
    //                                    rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }

protected:
  const DenseMap<int, std::array<Operation *, 2>> &dotAllocOps;
  const mlir::triton::metal::TargetInfo &targetInfo;
};

} // namespace

namespace mlir::triton::metal {
void populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis,
    const DenseMap<int, std::array<Operation *, 2>> &dotAllocOps,
    const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, dotAllocOps, targetInfo,
                                benefit);
}
} // namespace mlir::triton::metal
