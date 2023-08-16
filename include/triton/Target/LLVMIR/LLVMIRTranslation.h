#ifndef TRITON_TARGET_LLVM_IR_LLVM_IR_TRANSLATION_H
#define TRITON_TARGET_LLVM_IR_LLVM_IR_TRANSLATION_H
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Target/PTX/TmaMetadata.h"
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
                           mlir::triton::gpu::TMAMetadataTy &tmaInfos,
                           Target target);

// Translate mlir LLVM dialect to LLVMIR, return null if failed.
std::unique_ptr<llvm::Module>
translateLLVMToLLVMIR(llvm::LLVMContext *llvmContext, mlir::ModuleOp module,
                      Target target);

bool linkExternLib(llvm::Module &module, llvm::StringRef name,
                   llvm::StringRef path, Target target);

} // namespace triton
} // namespace mlir

#endif // TRITON_TARGET_LLVMIRTRANSLATION_H
