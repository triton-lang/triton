#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_HISTOGRAM_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_HISTOGRAM_OP_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

void populateHistogramOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit);

#endif
