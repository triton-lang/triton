#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_LOAD_STORE_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_LOAD_STORE_OP_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

namespace AMD {
void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       int numWarps,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit);
}

#endif
