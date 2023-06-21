#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCE_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCE_OP_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

void populateReduceOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAllocation &allocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit);

#endif
