#include "TritonAMDGPUToLLVM/Passes.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/TypeConverter.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONAMDGPUTOLLVM
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

// Returns the [min, max] range of waves per EU that the LLVM AMDGPU backend
// accepts for an "amdgpu-waves-per-eu" request, given the kernel's static LDS
// usage and its flat workgroup size range. LLVM drops requests outside this
// range with only a warning, so Triton reports them as out of resources.
//
// This mirrors AMDGPUSubtarget::getEffectiveWavesPerEU and
// AMDGPUSubtarget::getOccupancyWithWorkGroupSizes in
// llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp, together with the IsaInfo
// helpers they rely on. Keep them in sync.
std::pair<unsigned, unsigned>
computeAchievableWavesPerEU(const AMD::TargetInfo &targetInfo,
                            unsigned waveSize, unsigned maxFlatWorkGroupSize,
                            unsigned ldsBytes) {
  using triton::amdgpu::ISAFamily;
  namespace AMDGPU = llvm::AMDGPU;

  AMDGPU::GPUKind kind = AMDGPU::parseArchAMDGCN(targetInfo.getBaseArch());
  ISAFamily family = targetInfo.getISAFamily();
  bool cuMode = targetInfo.isCuMode();
  bool isGFX10Plus = family == ISAFamily::RDNA3 ||
                     family == ISAFamily::RDNA4m ||
                     family == ISAFamily::RDNA4 || family == ISAFamily::CDNA5;
  // gfx1250 always runs a workgroup on all four SIMDs.
  bool fullSIMDMode = family == ISAFamily::CDNA5 || !cuMode;

  const unsigned numSIMDs = AMDGPU::getNumWorkGroupSIMDs(fullSIMDMode);
  const unsigned maxWavesPerEU = AMDGPU::getMaxWavesPerEU(kind);
  const unsigned localMemorySize =
      AMDGPU::getLocalMemorySize(kind, fullSIMDMode);
  const unsigned ldsGranule = std::max(AMDGPU::getLDSAllocGranule(kind), 1u);

  // Triton requests "amdgpu-flat-work-group-size"="1,<maxFlatWorkGroupSize>".
  // Out-of-range requests fall back to the default range, whose minimum is
  // also the minimum flat workgroup size.
  const unsigned minWGSize = AMDGPU::getMinFlatWorkGroupSize();
  const unsigned maxWGSize =
      std::min(maxFlatWorkGroupSize, AMDGPU::getMaxFlatWorkGroupSize());

  auto getWavesPerWorkGroup = [&](unsigned wgSize) {
    return static_cast<unsigned>(llvm::divideCeil(wgSize, waveSize));
  };
  auto getMaxWorkGroupsPerCU = [&](unsigned wgSize) {
    unsigned maxWaves = maxWavesPerEU * numSIMDs;
    unsigned wavesPerWG = getWavesPerWorkGroup(wgSize);
    // Single-wave workgroups don't consume barrier resources.
    if (wavesPerWG == 1)
      return maxWaves;
    unsigned maxBarriers = isGFX10Plus && !cuMode ? 32 : 16;
    return std::min(maxWaves / wavesPerWG, maxBarriers);
  };

  // getOccupancyWithWorkGroupSizes(ldsBytes, {minWGSize, maxWGSize}).second.
  // The adjustment LLVM applies to the minimum waves per CU is omitted because
  // getEffectiveWavesPerEU only uses the maximum.
  unsigned maxOccupancy = 1;
  ldsBytes = llvm::alignTo(ldsBytes, ldsGranule);
  if (unsigned maxWGsLDS = localMemorySize / std::max(ldsBytes, 1u)) {
    auto propsFromWGSize = [&](unsigned wgSize) {
      unsigned wavesPerWG = getWavesPerWorkGroup(wgSize);
      unsigned wgsPerCU = std::min(getMaxWorkGroupsPerCU(wgSize), maxWGsLDS);
      return std::make_tuple(wavesPerWG, wgsPerCU, wavesPerWG * wgsPerCU);
    };
    auto [minWavesPerWG, maxWGsPerCU, maxWavesPerCU] =
        propsFromWGSize(minWGSize);
    auto [maxWavesPerWG, minWGsPerCU, minWavesPerCU] =
        propsFromWGSize(maxWGSize);
    if (minWavesPerCU >= maxWavesPerCU) {
      maxWavesPerCU = minWavesPerCU;
    } else {
      // Look for a potential larger group size than the minimum which
      // increases the concurrent number of waves on the CU for the same number
      // of concurrent workgroups on the CU.
      const unsigned waveSlotsPerCU = maxWavesPerEU * numSIMDs;
      unsigned leftoverSlots = waveSlotsPerCU - maxWGsPerCU * minWavesPerWG;
      if (unsigned leftoverSlotsPerWG = leftoverSlots / maxWGsPerCU) {
        maxWavesPerCU += maxWGsPerCU * std::min(leftoverSlotsPerWG,
                                                ((maxWGSize - 1) / waveSize) +
                                                    1 - minWavesPerWG);
      }
    }
    maxOccupancy = std::clamp(
        static_cast<unsigned>(llvm::divideCeil(maxWavesPerCU, numSIMDs)), 1u,
        maxWavesPerEU);
  }

  // getEffectiveWavesPerEU: the default range LLVM accepts requests within.
  unsigned minOccupancy =
      std::min(static_cast<unsigned>(
                   llvm::divideCeil(getWavesPerWorkGroup(maxWGSize), numSIMDs)),
               maxOccupancy);
  return {minOccupancy, maxOccupancy};
}

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
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
    addIllegalDialect<triton::instrument::TritonInstrumentDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addIllegalOp<triton::amdgpu::BufferLoadToLocalOp>();
    // Warp specialization is lowered later.
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

struct ConvertTritonAMDGPUToLLVM
    : public triton::impl::ConvertTritonAMDGPUToLLVMBase<
          ConvertTritonAMDGPUToLLVM> {
  explicit ConvertTritonAMDGPUToLLVM(StringRef gfxArch, bool ftz, bool cuMode) {
    this->gfxArch = gfxArch.str();
    this->ftz = ftz;
    this->cuMode = cuMode;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, NVVM::NVVMDialect, mlir::ROCDL::ROCDLDialect,
                mlir::triton::amdgpu::TritonAMDGPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    AMD::TargetInfo targetInfo(this->gfxArch.getValue(),
                               this->cuMode.getValue());
    if (targetInfo.getISAFamily() == triton::amdgpu::ISAFamily::Unknown) {
      mod.emitError("unsupported target: '") << this->gfxArch.getValue() << "'";
      return signalPassFailure();
    }

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);

    TritonAMDGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    TritonLLVMConversionTarget convTarget(*context);

    // Lower functions
    {
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      mlir::triton::AMD::populateFuncOpConversionPattern(
          typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
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
    recordAchievableWavesPerEU(targetInfo);

    // Convert call and ret ops
    {
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    // Emit logics to get threadId/blockIds/linearized clusterCTAId etc. and
    // cache the values. The reason to do it here is that cluster_ctaid is
    // currently implemented via inline asm, and thus cannot be CSEed.
    // clusterCTAId will be emitted only when numCTAs is larger than 1, and
    // other values will be DCEed if not used hereafter.

    RewritePatternSet patterns(context);
    int commonBenefit = patternBenefitPrioritizeOverLLVMConversions;
    // Make benefit for AMD specific patterns higher so they apply before common
    // patterns
    int AMDBenefit = commonBenefit + 1;
    auto populatePatterns5 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, benefit);
    };

    auto populatePatterns7 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, targetInfo, benefit);
    };

    AMD::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, AMDBenefit);
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, commonBenefit);
    AMD::populateDotOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                     AMDBenefit);
    AMD::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, ftz, axisInfoAnalysis, targetInfo, AMDBenefit);
    AMD::populateFpCastOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo, AMDBenefit);
    AMD::populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo, patterns,
                                           axisInfoAnalysis, AMDBenefit);
    AMD::populateMaskedOpsToLLVMPatterns(patterns, targetInfo);
    AMD::populateBarrierOpToLLVMPatterns(typeConverter, patterns, AMDBenefit);
    AMD::populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns,
                                            AMDBenefit);

    populatePatterns7(mlir::triton::populateReduceOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns7(mlir::triton::populateScanOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns5(mlir::triton::populateViewOpToLLVMPatterns,
                      commonBenefit);
    AMD::populateHistogramOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                           AMDBenefit);
    populatePatterns7(mlir::triton::populateHistogramOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns7(mlir::triton::populateGatherOpToLLVMPatterns,
                      commonBenefit);

    AMD::populateMemoryOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                        AMDBenefit);
    mlir::triton::populateMemoryOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, axisInfoAnalysis, commonBenefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, commonBenefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, commonBenefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, commonBenefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, commonBenefit);
    AMD::populateSPMDOpToLLVMPattern(typeConverter, patterns, AMDBenefit);

    mlir::triton::AMD::populateTritonAMDGPUToLLVMPatterns(
        typeConverter, patterns, targetInfo, AMDBenefit);
    mlir::triton::AMD::populateFp4ToFpToLLVMPatterns(typeConverter, patterns,
                                                     targetInfo, AMDBenefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

    mlir::triton::AMD::populateWarpIdOpToLLVMPattern(typeConverter, targetInfo,
                                                     patterns, commonBenefit);

    FailureOr<mlir::amdgpu::Chipset> maybeChipset =
        mlir::amdgpu::Chipset::parse(targetInfo.getBaseArch());
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(&getContext()),
                "Invalid AMDGPU chipset name: " + this->gfxArch);
      return signalPassFailure();
    }
    // Native lowering patterns
    mlir::populateGpuToROCDLConversionPatterns(
        typeConverter, patterns, mlir::gpu::amd::HIP, *maybeChipset);

    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, commonBenefit);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    mlir::triton::populateInstrumentationToLLVMPatterns(typeConverter, patterns,
                                                        targetInfo);
    mlir::triton::populateFpSanToLLVMPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }

    AMD::adjustModeRegister(mod, targetInfo);
    fixUpLoopAnnotation(mod);

    // Ensure warp group code is isolated from above.
    makeAllWarpGroupsIsolatedFromAbove(mod);
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto sharedSize = mod->getAttrOfType<IntegerAttr>("ttg.shared");
    auto arrayTy =
        LLVM::LLVMArrayType::get(elemTy, sharedSize ? sharedSize.getInt() : 0);
    auto global = LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::Internal,
        "global_smem",
        /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
    Block *initBlock = b.createBlock(&global.getInitializerRegion());
    b.setInsertionPointToStart(initBlock);
    Value undef = LLVM::UndefOp::create(b, loc, arrayTy);
    LLVM::ReturnOp::create(b, loc, undef);
  }

  // The static LDS allocation is final once global_smem is sized, so record
  // the waves-per-EU range LLVM will honor. The backend checks the requested
  // waves_per_eu against it.
  void recordAchievableWavesPerEU(const AMD::TargetInfo &targetInfo) {
    ModuleOp mod = getOperation();
    auto sharedSize = mod->getAttrOfType<IntegerAttr>("ttg.shared");
    unsigned ldsBytes = sharedSize ? sharedSize.getInt() : 0;
    int numWarps = triton::gpu::lookupNumWarps(mod);
    if (auto totalNumWarps =
            mod->getAttrOfType<IntegerAttr>("ttg.total-num-warps"))
      numWarps = totalNumWarps.getInt();
    unsigned waveSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    auto [minWavesPerEU, maxWavesPerEU] = computeAchievableWavesPerEU(
        targetInfo, waveSize, numWarps * waveSize, ldsBytes);
    Builder b(mod.getContext());
    mod->setAttr("ttg.min_waves_per_eu", b.getI32IntegerAttr(minWavesPerEU));
    mod->setAttr("ttg.max_waves_per_eu", b.getI32IntegerAttr(maxWavesPerEU));
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonAMDGPUToLLVMPass(StringRef gfxArch, bool ftz, bool cuMode) {
  return std::make_unique<ConvertTritonAMDGPUToLLVM>(gfxArch, ftz, cuMode);
}

} // namespace mlir::triton
