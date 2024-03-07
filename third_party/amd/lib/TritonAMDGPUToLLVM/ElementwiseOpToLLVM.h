#ifndef TRITON_CONVERSION_TRITONAMDGPU_TO_ELEMENTWISE_OP_H
#define TRITON_CONVERSION_TRITONAMDGPU_TO_ELEMENTWISE_OP_H

#include "TargetInfo.h"
#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

namespace AMD {
void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    int computeCapability, const TargetInfo &targetInfo,
    PatternBenefit benefit);
}

#endif
