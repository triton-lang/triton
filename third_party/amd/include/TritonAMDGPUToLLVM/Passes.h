#ifndef TRITONAMDGPU_CONVERSION_PASSES_H
#define TRITONAMDGPU_CONVERSION_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
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
createConvertTritonAMDGPUToLLVMPass(int32_t computeCapability, Target target);
#define GEN_PASS_REGISTRATION
#include "TritonAMDGPUToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
