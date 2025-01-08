#ifndef TRITON_CONVERSION_TRITONPROTON_TO_LLVM_PATTERNS_TRITON_PROTON_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONPROTON_TO_LLVM_PATTERNS_TRITON_PROTON_OP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace proton {
void populateRecordOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   const TargetInfoBase &targetInfo,
                                   PatternBenefit benefit);
} // namespace proton
} // namespace triton
} // namespace mlir

#endif
