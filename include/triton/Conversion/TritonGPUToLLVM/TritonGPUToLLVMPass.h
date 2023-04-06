#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_PASS_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_PASS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int computeCapability = 80,
                                 bool isROCM = false);

} // namespace triton

} // namespace mlir

#endif
