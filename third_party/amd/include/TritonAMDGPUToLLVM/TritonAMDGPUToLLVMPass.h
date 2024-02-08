#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_PASS_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_PASS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

enum Target { NVVM, ROCDL, Default = NVVM };

#define GEN_PASS_DECL
#include "TritonAMDGPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonAMDGPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonAMDGPUToLLVMPass(int32_t computeCapability);

} // namespace triton

} // namespace mlir

#endif
