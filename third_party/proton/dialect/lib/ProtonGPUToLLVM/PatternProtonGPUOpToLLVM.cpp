#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "third_party/proton/dialect/include/Dialect/ProtonGPU/IR/Dialect.h"

namespace mlir::triton {
namespace proton::gpu {
void populateProtonGPUOpPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit) {
  // TODO(fywkevin): populate all kinds of "TargetInfoBase"-related patterns
  // here.
}

} // namespace proton::gpu
} // namespace mlir::triton
