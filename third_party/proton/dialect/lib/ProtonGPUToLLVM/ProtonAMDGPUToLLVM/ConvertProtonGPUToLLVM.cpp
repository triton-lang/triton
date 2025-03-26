#include "Conversion/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/TargetInfo.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton::proton::gpu {
#define GEN_PASS_DEF_CONVERTPROTONAMDGPUTOLLVM
#include "proton/dialect/include/Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h.inc"
} // namespace triton::proton::gpu
} // namespace mlir

namespace {

class ProtonLLVMConversionTarget : public ConversionTarget {
public:
  explicit ProtonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addIllegalDialect<mlir::triton::proton::gpu::ProtonGPUDialect>();
    addIllegalDialect<mlir::triton::proton::ProtonDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertProtonAMDGPUToLLVM
    : public mlir::triton::proton::gpu::impl::ConvertProtonAMDGPUToLLVMBase<
          ConvertProtonAMDGPUToLLVM> {
  explicit ConvertProtonAMDGPUToLLVM(std::string arch) { this->arch = arch; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();
    auto tritonTargetInfo = mlir::triton::AMD::TargetInfo(arch);
    auto protonTargetInfo =
        mlir::triton::proton::gpu::AMD::TargetInfo(tritonTargetInfo);
    mlir::LowerToLLVMOptions option(context);
    TritonGPUToLLVMTypeConverter typeConverter(context, option,
                                               tritonTargetInfo);
    mlir::triton::proton::gpu::populateProtonGPUOpPatterns(
        typeConverter, patterns, protonTargetInfo, 1);
    auto convTarget = ProtonLLVMConversionTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {

namespace triton::proton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertProtonAMDGPUToLLVMPass(std::string arch) {
  return std::make_unique<ConvertProtonAMDGPUToLLVM>(arch);
}

} // namespace gpu

} // namespace triton::proton

} // namespace mlir
