#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_

#include "TDMUtility.h"
#include "TargetInfo.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::triton::AMD {
void populateConvertLayoutOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           const TargetInfo &targetInfo,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);

void populateMemoryOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    const TargetInfo &targetInfo,
                                    PatternBenefit benefit);

void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 PatternBenefit benefit);
void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, bool ftz,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    const TargetInfo &targetInfo, PatternBenefit benefit);

void populateFpCastOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns, bool ftz,
                                    ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                    ModuleAllocation &allocation,
                                    const TargetInfo &targetInfo,
                                    PatternBenefit benefit);

// Manipulates with execution mode register which is per-wavefront one.
// The register controls execution of instructions - e.g., rounding modes,
// exception handling, etc.
void adjustModeRegister(ModuleOp mod, const TargetInfo &targetInfo);

// `tdmMergeGroups` carries the implicit-merge analysis result built once per
// pass via `LLVM::AMD::computeTDMMergeGroups(module)`; the
// AsyncTDMCopyGlobalToLocalOp conversion pattern queries it to decide whether
// to emit a fused intrinsic.  The map MUST outlive the pattern set; pass an
// empty map (e.g. a default-constructed reference) to disable merging.
void populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    const DataFlowSolver *uniformitySolver,
    const llvm::DenseMap<Operation *,
                         std::shared_ptr<mlir::LLVM::AMD::TDMMergeGroupInfo>>
        &tdmMergeGroups,
    PatternBenefit benefit);

void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);
void populateTritonAMDGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        const AMD::TargetInfo &,
                                        PatternBenefit benefit);

void populateFp4ToFpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   const TargetInfo &targetInfo,
                                   PatternBenefit benefit);

void populateMaskedOpsToLLVMPatterns(RewritePatternSet &patterns,
                                     const TargetInfo &targetInfo);

void populateTensorPtrOpsToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit);

void populateHistogramOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       const TargetInfoBase &targetInfo,
                                       PatternBenefit benefit);

void populateBarrierOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit);
void populateWarpIdOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                   const TargetInfo &targetInfo,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit);
void populateFuncOpConversionPattern(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     const TargetInfoBase &targetInfo,
                                     PatternBenefit benefit);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_
