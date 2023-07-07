#ifndef TRITON_TARGET_LLVM_IR_LLVM_IR_TRANSLATION_H
#define TRITON_TARGET_LLVM_IR_LLVM_IR_TRANSLATION_H
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>
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
                           mlir::ModuleOp module, int computeCapability,
                           bool isROCM);

// Translate mlir LLVM dialect to LLVMIR, return null if failed.
std::unique_ptr<llvm::Module>
translateLLVMToLLVMIR(llvm::LLVMContext *llvmContext, mlir::ModuleOp module,
                      bool isROCM);

} // namespace triton
} // namespace mlir

#endif // TRITON_TARGET_LLVMIRTRANSLATION_H
