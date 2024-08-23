#include "AMDGPUToLLVM/AMDGPUToLLVMPass.h"
#include "Dialect/AMDGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
// clang-format off
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
// clang-format on

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
namespace tta = mlir::triton::amdgpu;

#define GEN_PASS_CLASSES
#include "AMDGPUToLLVM/Passes.h.inc"

namespace mlir::triton::AMD {

class AMDDialectLLVMConversionTarget : public ConversionTarget {
public:
  explicit AMDDialectLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<ROCDL::ROCDLDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addLegalDialect<triton::TritonDialect>();
    addLegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

void populateAMDGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                  PatternBenefit benefit) {
  // TODO: Add some actual patterns to lower
}
} // namespace mlir::triton::AMD

class ConvertAMDGPUToLLVM
    : public ConvertAMDGPUToLLVMBase<ConvertAMDGPUToLLVM> {

public:
  explicit ConvertAMDGPUToLLVM() {}
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    RewritePatternSet patterns(context);
    mlir::LowerToLLVMOptions option(context);

    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    constexpr int benefit = 1;
    AMD::populateAMDGPUToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis,
                                      benefit);

    AMD::AMDDialectLLVMConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createConvertAMDGPUToLLVMPass() {
  return std::make_unique<::ConvertAMDGPUToLLVM>();
}
} // namespace triton
} // namespace mlir
