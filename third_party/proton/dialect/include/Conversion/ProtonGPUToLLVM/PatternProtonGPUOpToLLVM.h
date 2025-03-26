#ifndef PROTONGPU_TO_LLVM_PATTERN_PROTONGPUOP_TO_LLVM_H
#define PROTONGPU_TO_LLVM_PATTERN_PROTONGPUOP_TO_LLVM_H

#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton {
namespace proton::gpu {

// Profiler index is private to each thread, address space is 5.
// See detail discussion:
// https://llvm.org/docs/NVPTXUsage.html#address-spaces
// https://llvm.org/docs/AMDGPUUsage.html#address-spaces
constexpr int IndexPtrAddrSpace = 5;

void populateProtonGPUOpPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit);
} // namespace proton::gpu
} // namespace mlir::triton

#endif // PROTONGPU_TO_LLVM_PATTERN_PROTONGPUOP_TO_LLVM_H
