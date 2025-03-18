#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton::proton {
#define GEN_PASS_DEF_CONVERTPROTONNVIDIAGPUTOLLVM
#include "proton/dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h.inc"
} // namespace triton::proton
} // namespace mlir

namespace {

struct ConvertProtonNvidiaGPUToLLVM
    : public mlir::triton::proton::impl::ConvertProtonNvidiaGPUToLLVMBase<
          ConvertProtonNvidiaGPUToLLVM> {
  explicit ConvertProtonNvidiaGPUToLLVM(int32_t computeCapability,
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
    auto protonTargetInfo =
        mlir::triton::proton::NVIDIA::TargetInfo(tritonTargetInfo);
    mlir::LowerToLLVMOptions option(context);
    TritonGPUToLLVMTypeConverter typeConverter(context, option,
                                               tritonTargetInfo);
    mlir::triton::proton::populateProtonGPUOpPatterns(typeConverter, patterns,
                                                      protonTargetInfo, 1);
    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {

namespace triton::proton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertProtonNvidiaGPUToLLVMPass(int32_t computeCapability,
                                       int32_t ptxVersion) {
  return std::make_unique<ConvertProtonNvidiaGPUToLLVM>(computeCapability,
                                                        ptxVersion);
}

} // namespace gpu

} // namespace triton::proton

} // namespace mlir
