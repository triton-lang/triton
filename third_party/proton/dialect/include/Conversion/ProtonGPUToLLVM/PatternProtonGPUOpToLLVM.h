#ifndef PROTON_GPU_TO_LLVM_PATTERN_PROTONGPU_OP_TO_LLVM_H
#define PROTON_GPU_TO_LLVM_PATTERN_PROTONGPU_OP_TO_LLVM_H

#include "TargetInfoBase.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton {
namespace proton {
namespace gpu {

// Profiler index is private to each thread, address space is 5
constexpr int indexPtrAddressSpace = 5;

constexpr int patternBenefitDefault = 1;

void populateProtonGPUOpPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit);
} // namespace gpu
} // namespace proton
} // namespace mlir::triton

#endif // PROTON_GPU_TO_LLVM_PATTERN_PROTONGPU_OP_TO_LLVM_H
