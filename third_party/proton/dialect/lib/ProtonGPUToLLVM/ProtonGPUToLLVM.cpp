#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/PatternProtonOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"

namespace mlir::triton {

namespace proton {
void populateProtonOpPatterns(LLVMTypeConverter &typeConverter,
                              RewritePatternSet &patterns,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit) {
  populateInitScopeOpToLLVMPattern(typeConverter, patterns, benefit);
  populateRecordOpToLLVMPattern(typeConverter, patterns, targetInfo, benefit);
}

} // namespace proton
} // namespace mlir::triton
