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
using namespace mlir::triton::NVIDIA;
using namespace mlir::triton::plugin;

namespace mlir::triton::plugin {
#define GEN_PASS_DEF_DIALECTPLUGINMAGICOP
#define GEN_PASS_DEF_DIALECTPLUGINFMAGICOP
#include "DialectPlugin/DialectPluginPasses.h.inc"
} // namespace mlir::triton::plugin

namespace {
class PluginLLVMConversionTarget : public ConversionTarget {
public:
  explicit PluginLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<::mlir::gpu::GPUDialect>();
    addLegalDialect<::mlir::arith::ArithDialect>();
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

  // Let's just do something kind of silly for the example to show what is
  // possible. Take the input to the magic op and add to the thread id since
  // Triton doesn't directly expose the thread id this is how a plugin writer
  // could get it and do something with it
  LogicalResult
  matchAndRewrite(mlir::triton::plugin::MagicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto a = op.getInput();
    Value tid = ::mlir::gpu::ThreadIdOp::create(rewriter, loc,
                                                ::mlir::gpu::Dimension::x);
    Value threadId = arith::IndexCastOp::create(rewriter, loc, i32_ty, tid);
    auto newOp = b.add(a, threadId);
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
    mlir::triton::plugin::populatePluginGPUOpPatterns(typeConverter, patterns,
                                                      tritonTargetInfo, 1);
    auto convTarget = PluginLLVMConversionTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

PluginTypeConverter::PluginTypeConverter(mlir::MLIRContext *context,
                                         int numWarps, int threadsPerWarp,
                                         int numCTAs)
    : context(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp),
      numCTAs(numCTAs) {
  addConversion([](mlir::Type type) { return type; });
}

namespace {

class PluginConversionTarget : public ConversionTarget {
public:
  explicit PluginConversionTarget(MLIRContext &ctx,
                                  PluginTypeConverter &typeConverter)
      : ConversionTarget(ctx) {
    addLegalDialect<triton::TritonDialect>();
    addLegalDialect<triton::gpu::TritonGPUDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addLegalOp<mlir::triton::gpu::ConvertLayoutOp>();
  }
};

struct PluginFMagicOpConversion
    : OpConversionPattern<mlir::triton::plugin::FMagicOp> {
  using OpConversionPattern<
      mlir::triton::plugin::FMagicOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::triton::plugin::FMagicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto typeConverter = getTypeConverter<PluginTypeConverter>();
    int numWarps = typeConverter->getNumWarps();
    int numCTAs = typeConverter->getNumCTAs();
    int threadsPerWarp = typeConverter->getThreadsPerWarp();
    int numElements = cast<RankedTensorType>(input.getType()).getShape()[0];

    int numElementPerThread = numElements / (numWarps * threadsPerWarp);
    // Increment all elements of the float RankedTensorType input by 1.0 (assume
    // f32)
    auto loc = op.getLoc();
    auto inputTy = cast<RankedTensorType>(input.getType());
    auto inputElementTy = inputTy.getElementType();

    SmallVector<Attribute> oneValues;
    oneValues.reserve(inputTy.getNumElements());
    for (int i = 0; i < inputTy.getNumElements(); ++i) {
      oneValues.push_back(FloatAttr::get(inputElementTy, numElements));
    }
    auto oneAttr = DenseElementsAttr::get(inputTy, oneValues);
    Value oneTensor =
        arith::ConstantOp::create(rewriter, loc, inputTy, oneAttr);

    // Perform elementwise addition
    Value incremented = arith::AddFOp::create(rewriter, loc, input, oneTensor);

    llvm::errs() << "numElements: " << numElements << "\n";
    llvm::errs() << "numWarps: " << numWarps << "\n";
    llvm::errs() << "numElementPerThread: " << numElementPerThread << "\n";
    llvm::errs() << "totalThreads: " << (numWarps * threadsPerWarp) << "\n";
    rewriter.replaceOp(op, incremented);
    return success();
  }
};

} // namespace

struct ConvertPluginGPUToTritonGPUPass
    : public mlir::triton::plugin::impl::DialectPluginFMagicOpBase<
          ConvertPluginGPUToTritonGPUPass> {
  explicit ConvertPluginGPUToTritonGPUPass(int32_t num_warps,
                                           int32_t threadsPerWarp,
                                           int32_t numCTAs) {}
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();
    PluginTypeConverter typeConverter(context, num_warps, threadsPerWarp,
                                      numCTAs);
    PluginConversionTarget convTarget(*context, typeConverter);

    patterns.add<PluginFMagicOpConversion>(typeConverter, context);
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
std::unique_ptr<OperationPass<ModuleOp>>
createConvertPluginGPUToTritonGPUPass(int32_t num_warps, int32_t threadsPerWarp,
                                      int32_t numCTAs) {
  return std::make_unique<ConvertPluginGPUToTritonGPUPass>(
      num_warps, threadsPerWarp, numCTAs);
}
} // namespace mlir::triton::plugin
