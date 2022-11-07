#ifndef TRITON_TARGET_LLVMIRTRANSLATION_H
#define TRITON_TARGET_LLVMIRTRANSLATION_H
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <vector>

namespace llvm {
class Module;
class LLVMContext;
} // namespace llvm

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace mlir {
namespace triton {

// add external dependent libs
void addExternalLibs(mlir::ModuleOp &module,
                     const std::vector<std::string> &names,
                     const std::vector<std::string> &paths);

// Translate TritonGPU dialect to LLVMIR, return null if failed.
std::unique_ptr<llvm::Module>
translateTritonGPUToLLVMIR(llvm::LLVMContext *llvmContext,
                           mlir::ModuleOp module);

// Translate mlir LLVM dialect to LLVMIR, return null if failed.
std::unique_ptr<llvm::Module>
translateLLVMToLLVMIR(llvm::LLVMContext *llvmContext, mlir::ModuleOp module);

bool linkExternLib(llvm::Module &module, llvm::StringRef path);

} // namespace triton
} // namespace mlir

#endif // TRITON_TARGET_LLVMIRTRANSLATION_H
