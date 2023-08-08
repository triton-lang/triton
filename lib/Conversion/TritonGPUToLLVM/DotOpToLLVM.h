#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_DOT_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_DOT_OP_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

void populateDotOpToLLVMPatterns(TritonGPUToLLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns, int numWarps,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 ModuleAllocation &allocation,
                                 PatternBenefit benefit);

#endif
