#ifndef PROTON_GPU_TO_LLVM_PATTERN_PROTON_OP_TO_LLVM_H
#define PROTON_GPU_TO_LLVM_PATTERN_PROTON_OP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton {
class TargetInfoBase;
namespace proton {

void populateRecordOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   const TargetInfoBase &targetInfo,
                                   PatternBenefit benefit);

void populateInitScopeOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      PatternBenefit benefit);

} // namespace proton
} // namespace mlir::triton

#endif // PROTON_TO_LLVM_PATTERN_PROTON_OP_TO_LLVM_H
