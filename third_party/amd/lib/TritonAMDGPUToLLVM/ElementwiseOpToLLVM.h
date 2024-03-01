#ifndef TRITON_CONVERSION_TRITONAMDGPU_TO_ELEMENTWISE_OP_H
#define TRITON_CONVERSION_TRITONAMDGPU_TO_ELEMENTWISE_OP_H

#include "TargetInfo.h"
#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

namespace AMD {
void populateElementwiseOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    int computeCapability, const TargetInfo &targetInfo,
    PatternBenefit benefit);
}

#endif
