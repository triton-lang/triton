#ifndef TRITONGPU_CONVERSION_TRITONMTGPUTOLLVM_PASSES_H
#define TRITONGPU_CONVERSION_TRITONMTGPUTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "mthreads/plugin/include/TritonMTGPUToLLVM/Passes.h.inc"

namespace MUSA {} // namespace MUSA

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonMTGPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonMTGPUToLLVMPass(int32_t computeCapability);
std::unique_ptr<OperationPass<ModuleOp>>
createConvertMTGPUBuiltinFuncToLLVMPass();

#define GEN_PASS_REGISTRATION
#include "mthreads/plugin/include/TritonMTGPUToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
