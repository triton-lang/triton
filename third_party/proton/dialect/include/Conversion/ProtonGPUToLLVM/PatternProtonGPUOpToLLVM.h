#ifndef PROTON_GPU_TO_LLVM_PATTERN_PROTONGPU_OP_TO_LLVM_H
#define PROTON_GPU_TO_LLVM_PATTERN_PROTONGPU_OP_TO_LLVM_H

#include "TargetInfoBase.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton {
namespace proton {
namespace gpu {
void populateProtonGPUOpPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit);
} // namespace gpu
} // namespace proton
} // namespace mlir::triton

#endif // PROTON_GPU_TO_LLVM_PATTERN_PROTONGPU_OP_TO_LLVM_H
