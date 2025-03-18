#include "Conversion/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/NvidiaPatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.h"
#include "Dialect/Proton/IR/Dialect.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::proton::gpu;

namespace mlir {
namespace triton::proton {
#define GEN_PASS_DEF_CONVERTPROTONNVIDIAGPUTOLLVM
#include "proton/dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h.inc"
} // namespace triton::proton
} // namespace mlir

namespace {

class ProtonLLVMConversionTarget : public ConversionTarget {
public:
  explicit ProtonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalDialect<triton::TritonDialect>();
    addLegalDialect<mlir::triton::gpu::TritonGPUDialect>();
    addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();
    addLegalDialect<mlir::gpu::GPUDialect>();
    addLegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<triton::proton::ProtonDialect>();
    addIllegalDialect<triton::proton::gpu::ProtonGPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

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
    ModuleOp mod = getOperation();
    auto tritonTargetInfo =
        mlir::triton::NVIDIA::TargetInfo(computeCapability, ptxVersion);
    auto protonTargetInfo =
        mlir::triton::proton::NVIDIA::TargetInfo(tritonTargetInfo);

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option,
                                               tritonTargetInfo);

    RewritePatternSet patterns(context);
    int benefit = patternBenefitDefault;

    populateProtonGPUOpPatterns(typeConverter, patterns, protonTargetInfo,
                                benefit);
    populateReadCounterOpToLLVMPatterns(typeConverter, patterns,
                                        protonTargetInfo, benefit);

    ProtonLLVMConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    return;
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
