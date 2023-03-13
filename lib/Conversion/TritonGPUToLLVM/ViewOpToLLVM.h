#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_VIEW_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_VIEW_OP_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

void populateViewOpToLLVMPatterns(TritonGPUToLLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns, int numWarps,
                                  AxisInfoAnalysis &axisInfoAnalysis,
                                  const Allocation *allocation, Value smem,
                                  PatternBenefit benefit);

#endif
