#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Utility.h"

#include "Allocation.h"
#include "Dialect/NVGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/Transforms/ConSanTargetHooks.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONGPUTOLLVM
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton::NVIDIA;

namespace {

class NvidiaLLVMConversionTarget : public ConversionTarget {
public:
  explicit NvidiaLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    // This base is shared by scoped partial conversions, so only list IR that
    // is valid throughout every phase of this pass.
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public NvidiaLLVMConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : NvidiaLLVMConversionTarget(ctx) {
    // CF is lowered after the axis-info-dependent patterns have finished.
    addLegalDialect<cf::ControlFlowDialect>();
    // The custom NVGPU dialect is LLVM-level IR lowered by a subsequent pass.
    addLegalDialect<triton::nvgpu::NVGPUDialect>();

    // Leave extension dialects (e.g., proton) unclassified so partial
    // conversion can preserve them for their own downstream lowering passes.
    addIllegalDialect<triton::TritonDialect, triton::gpu::TritonGPUDialect,
                      triton::nvidia_gpu::TritonNvidiaGPUDialect,
                      triton::instrument::TritonInstrumentDialect,
                      mlir::gpu::GPUDialect>();

    // Warp specialization and warp ID are lowered by subsequent passes.
    addLegalOp<triton::gpu::WarpIdOp, triton::gpu::WarpSpecializeOp,
               triton::gpu::WarpYieldOp,
               triton::gpu::WarpSpecializePartitionsOp,
               triton::gpu::WarpReturnOp>();
  }
};

void createSharedMemoryGlobal(ModuleOp mod, LLVMTypeConverter &typeConverter) {
  OpBuilder builder(mod.getBodyRegion());
  Type elemTy = typeConverter.convertType(builder.getIntegerType(8));
  // A zero-sized array with external linkage represents dynamic shared memory.
  // Request 16-byte alignment because 4xi32 is the widest supported access.
  auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
  LLVM::GlobalOp::create(
      builder, mod.getLoc(), arrayTy, /*isConstant=*/false,
      LLVM::Linkage::External, "global_smem", /*value=*/Attribute(),
      /*alignment=*/16, static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
}

struct ConvertTritonGPUToLLVM
    : public triton::impl::ConvertTritonGPUToLLVMBase<ConvertTritonGPUToLLVM> {
  using ConvertTritonGPUToLLVMBase::ConvertTritonGPUToLLVMBase;

  ConvertTritonGPUToLLVM(int32_t computeCapability)
      : ConvertTritonGPUToLLVMBase({computeCapability}) {}
  ConvertTritonGPUToLLVM(int32_t computeCapability, int32_t ptxVersion)
      : ConvertTritonGPUToLLVMBase({computeCapability, ptxVersion}) {}
  ConvertTritonGPUToLLVM(int32_t computeCapability, int32_t ptxVersion,
                         bool enableConcurrencySanitizer)
      : ConvertTritonGPUToLLVMBase(
            {computeCapability, ptxVersion, enableConcurrencySanitizer}) {}

  void runOnOperation() override;

private:
  LogicalResult prepareModule(ModuleOp mod, TargetInfo &targetInfo);
  LogicalResult lowerFunctions(ModuleOp mod, LLVMTypeConverter &typeConverter,
                               TargetInfo &targetInfo);
  void populateConversionPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                  TargetInfo &targetInfo);
  LogicalResult lowerTritonGPUOps(ModuleOp mod,
                                  LLVMTypeConverter &typeConverter,
                                  TargetInfo &targetInfo);
  LogicalResult lowerControlFlow(ModuleOp mod,
                                 LLVMTypeConverter &typeConverter);
  void finalizeModule(ModuleOp mod);
};

void ConvertTritonGPUToLLVM::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp mod = getOperation();
  TargetInfo targetInfo(computeCapability, ptxVersion);

  // These analyses and transformations require high-level shared-memory ops,
  // so they must all run before dialect conversion starts.
  if (failed(prepareModule(mod, targetInfo))) {
    signalPassFailure();
    return;
  }

  mlir::LowerToLLVMOptions option(context);
  option.overrideIndexBitwidth(32);
  TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);

  if (failed(lowerFunctions(mod, typeConverter, targetInfo))) {
    signalPassFailure();
    return;
  }

  // The shared-memory global must exist before call and return conversion so
  // those patterns can resolve each function's shared-memory base address.
  createSharedMemoryGlobal(mod, typeConverter);

  // CF was kept while ModuleAxisInfoAnalysis was in use.
  // Lower it after all axis-info-dependent patterns have finished.
  if (failed(lowerTritonGPUOps(mod, typeConverter, targetInfo)) ||
      failed(lowerControlFlow(mod, typeConverter))) {
    signalPassFailure();
    return;
  }

  finalizeModule(mod);
}

LogicalResult ConvertTritonGPUToLLVM::prepareModule(ModuleOp mod,
                                                    TargetInfo &targetInfo) {
  ModuleAllocation allocation(
      mod, mlir::triton::nvidia_gpu::getNvidiaAllocationAnalysisScratchSizeFn(
               targetInfo));
  mlir::triton::nvidia_gpu::runClusterBarrierInsertion(allocation,
                                                       computeCapability);
  if (failed(mlir::triton::nvidia_gpu::runCrossCTAMBarrierInitSyncInsertion(
          allocation, computeCapability)))
    return failure();

  ModuleMembarAnalysis membarPass(allocation, canSkipBarSync);
  membarPass.run();
  mlir::PassManager waitPm(mod.getContext());
  waitPm.addPass(ttng::createTritonNvidiaGPUTMemBarrierInsertionPass());
  if (failed(waitPm.run(mod)))
    return failure();

  if (enableConcurrencySanitizer) {
    auto hooks = mlir::triton::instrument::createConSanHooks("nvidia");
    if (!hooks) {
      mod.emitError("no ConSan hooks registered for nvidia");
      return failure();
    }
    if (failed(mlir::triton::instrument::runConcurrencySanitizer(mod, *hooks)))
      return failure();

    // Normalize instrumentation-generated IR before allocation and lowering.
    mlir::PassManager cleanupPm(mod.getContext());
    cleanupPm.addPass(mlir::triton::gluon::createGluonCanonicalize());
    cleanupPm.addPass(mlir::createCSEPass());
    if (failed(cleanupPm.run(mod)))
      return failure();
  }

  mlir::triton::nvidia_gpu::runClusterBarrierMbarAllocator(mod);
  mod.walk([&](triton::gpu::GlobalScratchAllocOp) -> WalkResult {
    mlir::triton::gpu::runGlobalScratchMemoryAllocation(mod);
    return WalkResult::interrupt();
  });

  return success();
}

LogicalResult ConvertTritonGPUToLLVM::lowerFunctions(
    ModuleOp mod, LLVMTypeConverter &typeConverter, TargetInfo &targetInfo) {
  NvidiaLLVMConversionTarget target(*mod.getContext());
  target.addIllegalOp<triton::FuncOp>();
  RewritePatternSet patterns(mod.getContext());
  mlir::triton::populateFuncOpConversionPattern(
      typeConverter, patterns, targetInfo, patternBenefitDefault);
  return applyPartialConversion(mod, target, std::move(patterns));
}

void ConvertTritonGPUToLLVM::populateConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, TargetInfo &targetInfo) {
  const int benefit = patternBenefitPrioritizeOverLLVMConversions;
  mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(
      typeConverter, targetInfo, patterns, benefit);
  mlir::triton::NVIDIA::populateTensorMemorySubviewOpToLLVMPattern(
      typeConverter, patterns, patternBenefitNvidiaTensorCoreSubviewPattern);
  mlir::triton::NVIDIA::populateTMAToLLVMPatterns(typeConverter, targetInfo,
                                                  patterns, benefit);
  populateDotOpToLLVMPatterns(typeConverter, patterns, computeCapability,
                              benefit);
  populateElementwiseOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis,
                                      computeCapability, targetInfo, benefit);
  populateClampFOpToLLVMPattern(typeConverter, patterns, axisInfoAnalysis,
                                computeCapability,
                                patternBenefitClampOptimizedPattern);
  populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo,
                                    computeCapability, patterns,
                                    axisInfoAnalysis, benefit);
  mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
  mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                             targetInfo, benefit);
  mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
  populateBarrierOpToLLVMPatterns(typeConverter, patterns, benefit, targetInfo);
  populateClusterOpsToLLVMPatterns(typeConverter, patterns, benefit,
                                   targetInfo);
  mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                                  targetInfo, benefit);
  mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                             targetInfo, benefit);
  mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                   targetInfo, benefit);
  mlir::triton::NVIDIA::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                    benefit);
  mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                            benefit);
  // TODO(thomas): this should probably be done in a separate step to not
  // interfere with our own lowering of arith ops. Add arith/math's patterns
  // to help convert scalar expression to LLVM.
  mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);
  mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);
  mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns, benefit);
  mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);
  mlir::triton::NVIDIA::populateMemoryOpToLLVMPatterns(
      typeConverter, targetInfo, patterns, benefit);
  mlir::triton::NVIDIA::populateTensorMemoryOpToLLVMPattern(typeConverter,
                                                            patterns, benefit);
  mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                 patterns, benefit);
  mlir::triton::NVIDIA::populateTCGen5MMAOpToLLVMPattern(
      typeConverter, patterns, benefit, targetInfo);
  mlir::triton::NVIDIA::populateFp4ToFpToLLVMPatterns(typeConverter, patterns,
                                                      benefit);
  mlir::triton::populateInstrumentationToLLVMPatterns(typeConverter, patterns,
                                                      targetInfo);
  mlir::triton::populateFpSanToLLVMPatterns(typeConverter, patterns);
  mlir::triton::populateGSanToLLVMPatterns(typeConverter, patterns,
                                           axisInfoAnalysis, targetInfo);
}

LogicalResult ConvertTritonGPUToLLVM::lowerTritonGPUOps(
    ModuleOp mod, LLVMTypeConverter &typeConverter, TargetInfo &targetInfo) {
  ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
  RewritePatternSet patterns(mod.getContext());
  populateConversionPatterns(typeConverter, patterns, axisInfoAnalysis,
                             targetInfo);
  TritonLLVMConversionTarget target(*mod.getContext());
  return applyPartialConversion(mod, target, std::move(patterns));
}

LogicalResult
ConvertTritonGPUToLLVM::lowerControlFlow(ModuleOp mod,
                                         LLVMTypeConverter &typeConverter) {
  MLIRContext *context = mod.getContext();
  NvidiaLLVMConversionTarget target(*context);
  target.addIllegalDialect<cf::ControlFlowDialect>();
  RewritePatternSet patterns(context);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  return applyPartialConversion(mod, target, std::move(patterns));
}

void ConvertTritonGPUToLLVM::finalizeModule(ModuleOp mod) {
  // Fold CTAId when there is only one CTA.
  if (triton::gpu::TritonGPUDialect::getNumCTAs(mod) == 1) {
    mod.walk([](triton::nvgpu::ClusterCTAIdOp id) {
      OpBuilder builder(id);
      Value zero = LLVM::createConstantI32(id->getLoc(), builder, 0);
      id.replaceAllUsesWith(zero);
    });
  }

  fixUpLoopAnnotation(mod);
  // Ensure warp-group code is isolated from above.
  makeAllWarpGroupsIsolatedFromAbove(mod);
}

} // anonymous namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass() {
  return std::make_unique<ConvertTritonGPUToLLVM>();
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability);
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability,
                                 int32_t ptxVersion) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability,
                                                  ptxVersion);
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability, int32_t ptxVersion,
                                 bool enableConcurrencySanitizer) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability, ptxVersion,
                                                  enableConcurrencySanitizer);
}

bool NVIDIA::canSkipBarSync(Operation *before, Operation *after,
                            bool /*beforeIsRead*/, bool /*afterIsRead*/,
                            Allocation * /*allocation*/) {
  // wait_barrier will never run ahead of the load it's waiting on
  if (isa<ttng::TMALoadLikeOpInterface>(before) &&
      isa<ttng::WaitBarrierOp>(after))
    return true;

  // An expect cannot run ahead of the wait
  if (auto wait = dyn_cast<ttng::WaitBarrierOp>(after)) {
    if (before->getParentRegion() == after->getParentRegion() &&
        isa<ttng::BarrierExpectOp, ttng::ArriveBarrierOp,
            ttng::AsyncCopyMbarrierArriveOp, ttng::TCGen5CommitOp>(before)) {
      auto signal = cast<triton::gpu::MBarrierOpInterface>(before);
      if (signal.getBarrier() == wait.getAlloc())
        return true;
    }
  }

  // Identical same-width commutative atomics can be freely reordered.
  auto beforeAtomic = dyn_cast<triton::gpu::LocalAtomicScatterRMWOp>(before);
  auto afterAtomic = dyn_cast<triton::gpu::LocalAtomicScatterRMWOp>(after);
  return beforeAtomic && afterAtomic && beforeAtomic.isCommutative() &&
         afterAtomic.isCommutative() &&
         beforeAtomic.getAtomicRmwOp() == afterAtomic.getAtomicRmwOp() &&
         beforeAtomic.getDst().getType().getElementType() ==
             afterAtomic.getDst().getType().getElementType();
}

} // namespace mlir::triton
