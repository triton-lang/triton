#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_CONVERT_LAYOUT_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_CONVERT_LAYOUT_OP_H

#include "TargetInfo.h"
#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;

namespace AMD {
void populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit);
}

#endif
