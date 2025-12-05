#include "DialectPlugin/DialectPluginDialect.h"
#include "DialectPlugin/DialectPluginOps.h"
#include "DialectPlugin/DialectPluginPasses.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h"


using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::NVIDIA;

namespace mlir::triton::dialectplugin {
#define GEN_PASS_DEF_DIALECTPLUGINMAGICOP
#include "DialectPlugin/DialectPluginPasses.h.inc"
}

namespace {
class PluginLLVMConversionTarget : public ConversionTarget {
public:
  explicit PluginLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addIllegalDialect<mlir::triton::dialectplugin::DialectPluginDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct PluginMagicOpConversion
: public ConvertOpToLLVMPattern<mlir::triton::dialectplugin::MagicOp> {
  PluginMagicOpConversion(LLVMTypeConverter &typeConverter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::dialectplugin::MagicOp op, OpAdaptor adaptor,
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

namespace mlir::triton::dialectplugin {
void populatePluginGPUOpPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit) {
  patterns.add<PluginMagicOpConversion>(typeConverter, targetInfo);
return;
}
} // namespace mlir::triton::dialectplugin

struct ConvertPluginGPUToLLVMPass
    : public mlir::triton::dialectplugin::impl::DialectPluginMagicOpBase<
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
    mlir::triton::dialectplugin::populatePluginGPUOpPatterns(
        typeConverter, patterns, tritonTargetInfo, 1);
    auto convTarget = PluginLLVMConversionTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};


namespace mlir::triton::dialectplugin {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertPluginGPUToLLVMPass(int32_t computeCapability,
                                       int32_t ptxVersion) {
  return std::make_unique<ConvertPluginGPUToLLVMPass>(computeCapability,
                                                        ptxVersion);
  }
} // namespace mlir::triton::dialectplugin
