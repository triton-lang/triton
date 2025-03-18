#ifndef PROTONGPU_TO_LLVM_PATTERN_PROTONGPUOP_TO_LLVM_H
#define PROTONGPU_TO_LLVM_PATTERN_PROTONGPUOP_TO_LLVM_H

#include "TargetInfoBase.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton {
namespace proton {
void populateProtonGPUOpPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit);
} // namespace proton
} // namespace mlir::triton

#endif // PROTONGPU_TO_LLVM_PATTERN_PROTONGPUOP_TO_LLVM_H
