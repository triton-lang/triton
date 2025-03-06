#ifndef PROTON_RECORDOP_TO_LLVM_H
#define PROTON_RECORDOP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

// TODO(fywkevin): This pattern is a temporary solution to convert the record op
// to LLVM IR such that we could still have a path to test the frontend. Need to
// be removed soon.

namespace mlir::triton {
class TargetInfoBase;
namespace proton {

void populateRecordOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   const TargetInfoBase &targetInfo,
                                   PatternBenefit benefit);

} // namespace proton
} // namespace mlir::triton

#endif // PROTON_RECORDOP_TO_LLVM_H
