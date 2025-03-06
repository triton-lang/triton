#ifndef PROTON_GLOBALSCRATCHALLOCOP_TO_LLVM_H
#define PROTON_GLOBALSCRATCHALLOCOP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

// TODO(crobeck): This pattern is a temporary solution to convert the op
// to LLVM IR such that we could still have a path to test the frontend. Need to
// be removed soon.

namespace mlir::triton {
class TargetInfoBase;
namespace proton {

void populateGlobalScratchAllocOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               const TargetInfoBase &targetInfo,
                                               PatternBenefit benefit);

} // namespace proton
} // namespace mlir::triton

#endif // PROTON_GLOBALSCRATCHALLOCOP_TO_LLVM_H
