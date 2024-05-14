#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonCPUToLLVM/Passes.h"
#include "triton/Conversion/TritonCPUToLLVM/PatternTritonCPUOpToLLVM.h"
#include "triton/Conversion/TritonCPUToLLVM/TypeConverter.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONCPUTOLLVM
#include "triton/Conversion/TritonCPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<index::IndexDialect>();
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::cpu::TritonCPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertTritonCPUToLLVM
    : public triton::impl::ConvertTritonCPUToLLVMBase<ConvertTritonCPUToLLVM> {
  using ConvertTritonCPUToLLVMBase<
      ConvertTritonCPUToLLVM>::ConvertTritonCPUToLLVMBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::cpu::TritonCPUDialect, LLVM::LLVMDialect>();
  }

  ConvertTritonCPUToLLVM() : ConvertTritonCPUToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    // Lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      TritonCPUToLLVMTypeConverter typeConverter(context, option);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      mlir::triton::cpu::populateFuncOpConversionPattern(
          typeConverter, funcPatterns,
          mlir::triton::cpu::patternBenefitDefault);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    RewritePatternSet patterns(context);
    mlir::triton::cpu::CPUTargetInfo targetInfo;
    int benefit =
        mlir::triton::cpu::patternBenefitPrioritizeOverLLVMConversions;
    mlir::triton::cpu::populateControlFlowOpToLLVMPattern(typeConverter,
                                                          patterns, benefit);
    mlir::triton::cpu::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                                    targetInfo, benefit);
    mlir::triton::cpu::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                   targetInfo, benefit);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonCPUToLLVMPass() {
  return std::make_unique<ConvertTritonCPUToLLVM>();
}

} // namespace triton
} // namespace mlir
