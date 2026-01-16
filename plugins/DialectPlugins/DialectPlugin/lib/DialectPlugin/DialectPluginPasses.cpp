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


namespace mlir::triton::plugin {
#define GEN_PASS_DEF_DIALECTPLUGINMAGICOP
#include "DialectPlugin/DialectPluginPasses.h.inc"
} // namespace mlir::triton::plugin

namespace {
class PluginTypeConverter : public TypeConverter {
  public:
  explicit PluginTypeConverter(MLIRContext *context,
                                               int numWarps, int threadsPerWarp,
                                               int numCTAs,
                                               bool enableSourceRemat){
      addConversion([](Type type) { return type; });
    }
};

class PluginConversionTarget : public ConversionTarget {
public:
  explicit PluginConversionTarget(MLIRContext &ctx, PluginTypeConverter &typeConverter)
      : ConversionTarget(ctx) {
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
    rewriter.replaceOp(op, input);
    return success();
  }
};

} // namespace

struct ConvertPluginGPUToLLVMPass
    : public mlir::triton::plugin::impl::DialectPluginMagicOpBase<
          ConvertPluginGPUToLLVMPass> {
  explicit ConvertPluginGPUToLLVMPass(int32_t num_warps) {
    this->num_warps = num_warps;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();
    int numWarps = 1; //mlir::triton::gpu::lookupNumWarps(mod);
    int threadsPerWarp = 1; // mlir::triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    int numCTAs = 1; //mlir::triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    PluginTypeConverter typeConverter(context, numWarps, threadsPerWarp,
                                         numCTAs, /*enableSourceRemat=*/false);
    PluginConversionTarget convTarget(*context, typeConverter);
    patterns.add<PluginMagicOpConversion>(typeConverter, context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

namespace mlir::triton::plugin {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertPluginGPUToLLVMPass(int32_t num_warps) {

  return std::make_unique<ConvertPluginGPUToLLVMPass>(num_warps);
}
} // namespace mlir::triton::plugin
