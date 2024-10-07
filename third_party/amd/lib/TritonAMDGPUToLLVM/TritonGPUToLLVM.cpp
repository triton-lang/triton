#include "TritonAMDGPUToLLVM/Passes.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONAMDGPUTOLLVM
#include "TritonAMDGPUToLLVM/Passes.h.inc"
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
    addLegalDialect<ROCDL::ROCDLDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<ROCDL::ROCDLDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addLegalOp<triton::amdgpu::InstructionSchedHint>();
  }
};

struct ConvertTritonAMDGPUToLLVM
    : public triton::impl::ConvertTritonAMDGPUToLLVMBase<
          ConvertTritonAMDGPUToLLVM> {
  explicit ConvertTritonAMDGPUToLLVM(StringRef targetArch, bool ftz,
                                     bool enable_buffer_ops) {
    this->arch = targetArch.str();
    this->ftz = ftz;
    this->enable_buffer_ops = enable_buffer_ops;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, NVVM::NVVMDialect,
                    mlir::ROCDL::ROCDLDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    AMD::TargetInfo targetInfo(this->arch.getValue());
    if (targetInfo.getISAFamily() == AMD::ISAFamily::Unknown) {
      mod.emitError("unsupported target: '") << this->arch.getValue() << "'";
      return signalPassFailure();
    }

    // Given how young is the buffer intrinsic support in Triton, we shield it
    // behind this variable. Once we smooth all the edges, we should get rid of
    // this
    bool enableBufferIntrinsics = this->enable_buffer_ops;

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);

    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    TritonLLVMConversionTarget convTarget(*context);

    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Hack: WSMaterialization may have changed the effective number of warps,
    // in a way that isn't reflected in triton_gpu.num-warps.  If so, we have to
    // respect that here.
    if (Attribute attr = mod->getAttr("triton_gpu.num-warp-groups-per-cta")) {
      numWarps *= cast<IntegerAttr>(attr).getInt();
    }

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(mod);
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    // Collect all the assumed expressions
    DenseSet<Value> assumptions;
    mod.walk([&](LLVM::AssumeOp op) { assumptions.insert(op->getOperand(0)); });

    // Lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      mlir::triton::populateFuncOpConversionPattern(typeConverter, funcPatterns,
                                                    numWarps, targetInfo,
                                                    patternBenefitDefault);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(typeConverter);

    // Convert call and ret ops
    {
      mlir::LowerToLLVMOptions option(context);
      TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    // Emit logics to get threadId/blockIds/linearized clusterCTAId etc. and
    // cache the values. The reason to do it here is that cluster_ctaid is
    // currently implemented via inline asm, and thus cannot be CSEed.
    // clusterCTAId will be emitted only when numCTAs is larger than 1, and
    // other values will be DCEed if not used hereafter.
    OpBuilder::InsertPoint indexInsertPoint;

    RewritePatternSet patterns(context);
    int commonBenefit = patternBenefitPrioritizeOverLLVMConversions;
    // Make benefit for AMD specific patterns higher so they apply before common
    // patterns
    int AMDBenefit = commonBenefit + 1;
    auto populatePatterns1 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, numWarps, axisInfoAnalysis,
                   allocation, benefit);
    };

    auto populatePatterns5 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, benefit);
    };

    auto populatePatterns6 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, numWarps, axisInfoAnalysis,
                   allocation, targetInfo, benefit);
    };

    auto populatePatterns7 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, targetInfo, benefit);
    };

    AMD::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, numWarps,
                                               axisInfoAnalysis, AMDBenefit);
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, commonBenefit);
    AMD::populateDotOpToLLVMPatterns(typeConverter, patterns, numWarps,
                                     axisInfoAnalysis, AMDBenefit);
    AMD::populateElementwiseOpToLLVMPatterns(typeConverter, patterns, ftz,
                                             axisInfoAnalysis, allocation,
                                             targetInfo, AMDBenefit);
    AMD::populateLoadStoreOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, numWarps, axisInfoAnalysis,
        assumptions, enableBufferIntrinsics, AMDBenefit);
    populatePatterns7(mlir::triton::populateReduceOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns7(mlir::triton::populateScanOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns5(mlir::triton::populateViewOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns7(mlir::triton::populateHistogramOpToLLVMPatterns,
                      commonBenefit);
    mlir::triton::populateMemoryOpToLLVMPattern(typeConverter, targetInfo,
                                                patterns, commonBenefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, commonBenefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, commonBenefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, commonBenefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, commonBenefit);
    AMD::populateSPMDOpToLLVMPattern(typeConverter, patterns, AMDBenefit);

    mlir::triton::AMD::populateTritonAMDGPUToLLVMPatterns(typeConverter,
                                                          patterns, AMDBenefit);

    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

    // Native lowering patterns
    mlir::populateGpuToROCDLConversionPatterns(typeConverter, patterns,
                                               mlir::gpu::amd::HIP);

    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, commonBenefit);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    auto global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::kSharedMemorySpace));
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonAMDGPUToLLVMPass(StringRef targetArch, bool ftz,
                                    bool enable_buffer_ops) {
  return std::make_unique<ConvertTritonAMDGPUToLLVM>(targetArch, ftz,
                                                     enable_buffer_ops);
}

} // namespace triton
} // namespace mlir
