#ifndef THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPU_TO_LLVM_PATTERNS_AMDGPU_OP_TO_LLVM_H
#define THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPU_TO_LLVM_PATTERNS_AMDGPU_OP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"

using namespace mlir;
// using namespace mlir::triton;

namespace mlir::triton::AMD {

void populateViewSliceOpTritonAMDGPUToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

}

#endif