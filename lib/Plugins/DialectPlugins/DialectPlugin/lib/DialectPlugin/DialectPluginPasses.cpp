#include "DialectPlugin/DialectPluginPasses.h"
#include "DialectPlugin/DialectPluginDialect.h"
#include "DialectPlugin/DialectPluginOps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::NVIDIA;

namespace mlir::triton::plugin {
#define GEN_PASS_DEF_DIALECTPLUGINMAGICOP
#include "DialectPlugin/DialectPluginPasses.h.inc"
} // namespace mlir::triton::plugin

namespace {
class PluginLLVMConversionTarget : public ConversionTarget {
public:
  explicit PluginLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addIllegalDialect<mlir::triton::plugin::DialectPluginDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct PluginMagicOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::plugin::MagicOp> {
  PluginMagicOpConversion(LLVMTypeConverter &typeConverter,
                          const TargetInfoBase &targetInfo,
                          PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(mlir::triton::plugin::MagicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto a = op.getInput();
    auto newOp = mlir::LLVM::ZeroOp::create(rewriter, loc, a.getType());
    rewriter.replaceOp(op, newOp);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

namespace mlir::triton::plugin {
void populatePluginGPUOpPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit) {
  patterns.add<PluginMagicOpConversion>(typeConverter, targetInfo);
  return;
}
} // namespace mlir::triton::plugin

struct ConvertPluginGPUToLLVMPass
    : public mlir::triton::plugin::impl::DialectPluginMagicOpBase<
          ConvertPluginGPUToLLVMPass> {
  explicit ConvertPluginGPUToLLVMPass(int32_t computeCapability,
                                      int32_t ptxVersion) {
    this->computeCapability = computeCapability;
    this->ptxVersion = ptxVersion;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();
    auto tritonTargetInfo =
        mlir::triton::NVIDIA::TargetInfo(computeCapability, ptxVersion);
    mlir::LowerToLLVMOptions option(context);
    TritonGPUToLLVMTypeConverter typeConverter(context, option,
                                               tritonTargetInfo);
    mlir::triton::plugin::populatePluginGPUOpPatterns(
        typeConverter, patterns, tritonTargetInfo, 1);
    auto convTarget = PluginLLVMConversionTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

namespace mlir::triton::plugin {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertPluginGPUToLLVMPass(int32_t computeCapability,
                                 int32_t ptxVersion) {
  return std::make_unique<ConvertPluginGPUToLLVMPass>(computeCapability,
                                                      ptxVersion);
}
} // namespace mlir::triton::plugin
