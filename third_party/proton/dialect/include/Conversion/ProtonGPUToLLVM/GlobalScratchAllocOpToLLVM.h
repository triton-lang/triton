#ifndef PROTONGPU_TO_LLVM_GLOBALSCRATCHALLOCOP_TO_LLVM_H
#define PROTONGPU_TO_LLVM_GLOBALSCRATCHALLOCOP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton {
class TargetInfoBase;
namespace proton {

void populateGlobalScratchAllocOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               const TargetInfoBase &targetInfo,
                                               PatternBenefit benefit);

} // namespace proton
} // namespace mlir::triton

#endif // PROTONGPU_TO_LLVM_GLOBALSCRATCHALLOCOP_TO_LLVM_H
