#ifndef TRITONAMDGPU_TO_LLVM_PATTERNS_AMDGPU_OP_TO_LLVM_H
#define TRITONAMDGPU_TO_LLVM_PATTERNS_AMDGPU_OP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton::AMD {

void populateViewSliceOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                       mlir::RewritePatternSet &patterns,
                                       mlir::PatternBenefit benefit);

}

#endif
