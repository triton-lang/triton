#ifndef PROTON_GPU_TO_LLVM_NVIDIA_PATTERN_PROTONGPU_OP_TO_LLVM_H
#define PROTON_GPU_TO_LLVM_NVIDIA_PATTERN_PROTONGPU_OP_TO_LLVM_H

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton {
namespace proton {
namespace NVIDIA {

void populateReadCounterOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         const TargetInfo &targetInfo,
                                         PatternBenefit benefit);
} // namespace NVIDIA
} // namespace proton
} // namespace mlir::triton

#endif // PROTON_GPU_TO_LLVM_NVIDIA_PATTERN_PROTONGPU_OP_TO_LLVM_H
