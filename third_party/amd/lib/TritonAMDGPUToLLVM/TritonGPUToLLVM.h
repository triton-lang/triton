#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

namespace AMD {
void populateTritonGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns, int numWarps,
                                     ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                     ModuleAllocation &allocation,
                                     PatternBenefit benefit);
} // namespace AMD

#endif
