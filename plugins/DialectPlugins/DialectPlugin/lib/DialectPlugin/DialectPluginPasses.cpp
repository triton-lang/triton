#include "DialectPlugin/DialectPluginPasses.h"
#include "DialectPlugin/DialectPluginDialect.h"
#include "DialectPlugin/DialectPluginOps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::plugin;

namespace mlir::triton::plugin {
#define GEN_PASS_DEF_DIALECTPLUGINMAGICOP
#include "DialectPlugin/DialectPluginPasses.h.inc"


PluginTypeConverter::PluginTypeConverter(mlir::MLIRContext *context,
                                               int numWarps, int threadsPerWarp,
                                               int numCTAs)
    : context(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp),
      numCTAs(numCTAs) {
  addConversion([](mlir::Type type) { return type; });
}
}

namespace {

class PluginConversionTarget : public ConversionTarget {
public:
  explicit PluginConversionTarget(MLIRContext &ctx, PluginTypeConverter &typeConverter)
      : ConversionTarget(ctx) {
    addLegalDialect<triton::TritonDialect>();
    addLegalDialect<triton::gpu::TritonGPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addLegalOp<mlir::triton::gpu::ConvertLayoutOp>();
  }
};


struct PluginMagicOpConversion : OpConversionPattern<mlir::triton::plugin::MagicOp> {
  using OpConversionPattern<mlir::triton::plugin::MagicOp >::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::triton::plugin::MagicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto typeConverter = getTypeConverter<PluginTypeConverter>();
    int numWarps = typeConverter->getNumWarps();
    int numCTAs = typeConverter->getNumCTAs();
    int threadsPerWarp = typeConverter->getThreadsPerWarp();
    rewriter.replaceOp(op, input);
    return success();
  }
};

} // namespace

struct ConvertPluginGPUToTritonGPUPass
    : public mlir::triton::plugin::impl::DialectPluginMagicOpBase<
          ConvertPluginGPUToTritonGPUPass> {
  explicit ConvertPluginGPUToTritonGPUPass(int32_t num_warps, int32_t threadsPerWarp, int32_t numCTAs) {}
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();
    PluginTypeConverter typeConverter(context, num_warps, threadsPerWarp,
                                         numCTAs);
    PluginConversionTarget convTarget(*context, typeConverter);
    patterns.add<PluginMagicOpConversion>(typeConverter, context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

namespace mlir::triton::plugin {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertPluginGPUToTritonGPUPass(int32_t num_warps, int32_t threadsPerWarp, int32_t numCTAs) {

  return std::make_unique<ConvertPluginGPUToTritonGPUPass>(num_warps, threadsPerWarp, numCTAs);
}
} // namespace mlir::triton::plugin
