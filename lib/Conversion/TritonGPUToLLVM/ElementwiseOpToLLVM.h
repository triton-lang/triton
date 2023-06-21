#ifndef TRITON_CONVERSION_TRITONGPU_TO_ELEMENTWISE_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_ELEMENTWISE_OP_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

void populateElementwiseOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

bool isLegalElementwiseOp(Operation *op);

#endif
