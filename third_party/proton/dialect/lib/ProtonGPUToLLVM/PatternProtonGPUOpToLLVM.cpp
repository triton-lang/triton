#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/GlobalScratchAllocOpToLLVM.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/RecordOpToLLVM.h"
#include "third_party/proton/dialect/include/Dialect/ProtonGPU/IR/Dialect.h"

namespace mlir::triton {
namespace proton {
void populateProtonGPUOpPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit) {}

} // namespace proton
} // namespace mlir::triton
