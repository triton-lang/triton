#ifndef TRITON_CONVERSION_TRITONGPUROCM_TO_LLVM_PASS_H
#define TRITON_CONVERSION_TRITONGPUROCM_TO_LLVM_PASS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Target/PTX/TmaMetadata.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

enum Target { NVVM, ROCDL, Default = NVVM };

#define GEN_PASS_DECL
#include "triton/Conversion/TritonGPUROCMToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUROCMToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUROCMToLLVMPass(const ConvertTritonGPUROCMToLLVMOptions &options);

} // namespace triton

} // namespace mlir

#endif
