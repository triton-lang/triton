#ifndef PROTONGPU_CONVERSION_PROTONGPUTOLLVM_PASSES_H
#define PROTONGPU_CONVERSION_PROTONGPUTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton::proton {

#define GEN_PASS_DECL
#include "proton/dialect/include/Conversion/ProtonGPUToLLVM/Passes.h.inc"

namespace gpu {
std::unique_ptr<OperationPass<ModuleOp>> createAddProtonKernelArgPass();
std::unique_ptr<OperationPass<ModuleOp>> createAllocateProtonSharedMemoryPass();
std::unique_ptr<OperationPass<ModuleOp>>
createAllocateProtonGlobalScratchBufferPass();

} // namespace gpu

#define GEN_PASS_REGISTRATION
#include "proton/dialect/include/Conversion/ProtonGPUToLLVM/Passes.h.inc"

} // namespace triton::proton

} // namespace mlir

#endif
