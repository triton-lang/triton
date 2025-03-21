#ifndef PROTONGPU_TO_LLVM_PATTERN_PROTONGPUOP_TO_LLVM_H
#define PROTONGPU_TO_LLVM_PATTERN_PROTONGPUOP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton {
namespace proton::gpu {
void populateProtonGPUOpPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit);
} // namespace proton::gpu
} // namespace mlir::triton

#endif // PROTONGPU_TO_LLVM_PATTERN_PROTONGPUOP_TO_LLVM_H
