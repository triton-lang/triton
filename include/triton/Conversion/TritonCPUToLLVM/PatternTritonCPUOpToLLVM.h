#ifndef TRITON_CONVERSION_TRITONCPU_TO_LLVM_PATTERNS_TRITON_CPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONCPU_TO_LLVM_PATTERNS_TRITON_CPU_OP_TO_LLVM_H

#include "CPUTargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
// Some populate* functions have name collisions with the ones for GPUs.
namespace cpu {

constexpr int patternBenefitDefault = 1;
constexpr int patternBenefitPrioritizeOverLLVMConversions = 10;
constexpr int patternBenefitClampOptimizedPattern = 20;
constexpr int patternBenefitConvertLayoutOptimizedPattern = 20;

void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const cpu::CPUTargetInfo &targetInfo,
                                 PatternBenefit benefit);

void populateControlFlowOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit);

void populateFuncOpConversionPattern(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit);

void populatePrintOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  const CPUTargetInfo &targetInfo,
                                  PatternBenefit benefit);

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif
