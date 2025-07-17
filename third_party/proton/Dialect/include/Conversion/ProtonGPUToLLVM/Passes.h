#ifndef PROTONGPU_TO_LLVM_PASSES_H
#define PROTONGPU_TO_LLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;

namespace triton::proton::gpu {

#define GEN_PASS_DECL
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createAddSchedBarriersPass();

#define GEN_PASS_REGISTRATION
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/Passes.h.inc"

} // namespace triton::proton::gpu

} // namespace mlir

#endif // PROTONGPU_TO_LLVM_PASSES_H
