#ifndef TRITON_CONVERSION_TRITONCOMMONGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONCOMMONGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"

typedef llvm::DenseMap<mlir::Operation *, mlir::triton::MakeTensorPtrOp>
    TensorPtrMapT;

namespace mlir {
namespace triton {
namespace common {

void populateMemoryOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit);
} // namespace common
} // namespace triton
} // namespace mlir

#endif
