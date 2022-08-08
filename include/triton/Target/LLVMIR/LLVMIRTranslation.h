#ifndef TRITON_TARGET_LLVMIRTRANSLATION_H
#define TRITON_TARGET_LLVMIRTRANSLATION_H
#include <memory>

namespace llvm {
class Module;
class LLVMContext;
} // namespace llvm

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace mlir {
namespace triton {

// Translate mlir LLVM dialect to LLVMIR, return null if failed.
std::unique_ptr<llvm::Module>
TranslateLLVMToLLVMIR(llvm::LLVMContext *llvmContext, mlir::ModuleOp module);

} // namespace triton
} // namespace mlir

#endif // TRITON_TARGET_LLVMIRTRANSLATION_H
