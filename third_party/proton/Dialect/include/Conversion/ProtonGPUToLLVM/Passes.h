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

#ifdef TRITON_BUILD_AMD_BACKEND
// Implemented in ProtonAMDGPUToLLVM, which is only built with the AMD backend.
std::unique_ptr<OperationPass<ModuleOp>> createAddSchedBarriersPass();
#else
// The tablegen'd registration helpers in Passes.h.inc reference this symbol,
// so a declaration must exist; registrations of this pass are disabled when
// the AMD backend is not built.
inline std::unique_ptr<OperationPass<ModuleOp>> createAddSchedBarriersPass() {
  return nullptr;
}
#endif

#define GEN_PASS_REGISTRATION
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/Passes.h.inc"

} // namespace triton::proton::gpu

} // namespace mlir

#endif // PROTONGPU_TO_LLVM_PASSES_H
