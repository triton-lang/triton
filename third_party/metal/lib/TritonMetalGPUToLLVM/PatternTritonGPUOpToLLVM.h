#ifndef TRITON_THIRD_PARTY_METAL_LIB_TRITONMETALGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_
#define TRITON_THIRD_PARTY_METAL_LIB_TRITONMETALGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir::triton::metal {
void populateFuncOpConversionPattern(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     const TargetInfoBase &targetInfo,
                                     PatternBenefit benefit);

void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfo &targetInfo,
    PatternBenefit benefit);

void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit);

void populateGPUIdxOpsConversionPattern(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        const TargetInfoBase &targetInfo,
                                        PatternBenefit benefit);

void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const metal::TargetInfo &targetInfo,
                                 PatternBenefit benefit);

void populateBarrierOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit);

void populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis,
    const DenseMap<int, std::array<Operation *, 2>> &dotAllocOps,
    const TargetInfo &targetInfo, PatternBenefit benefit);

void populateSimdgroupAsyncCopyOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit);

void populateSimdgroupWaitOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit);

void populateSimdgroupMMAOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit);

void populateSimdgroupStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit);

} // namespace mlir::triton::metal

#endif // TRITON_THIRD_PARTY_METAL_LIB_TRITONMETALGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_