#ifndef TRITON_CONVERSION_TRITONGPUTOLLVM_WARPIFUTILITY_H
#define TRITON_CONVERSION_TRITONGPUTOLLVM_WARPIFUTILITY_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton {
LogicalResult lowerWarpIfOps(ModuleOp module,
                             const LLVMTypeConverter &converter,
                             const TargetInfoBase &target);
} // namespace mlir::triton
#endif
