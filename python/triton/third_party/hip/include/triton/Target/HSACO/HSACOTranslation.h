#ifndef TRITON_TARGET_HSACOTRANSLATION_H
#define TRITON_TARGET_HSACOTRANSLATION_H

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace mlir {
class ModuleOp;
}

namespace llvm {
class Module;
class LLVMContext;
} // namespace llvm

namespace mlir {
namespace triton {

// add external libs to modules
void addExternalLibs(mlir::ModuleOp &module,
                     const std::vector<std::string> &names,
                     const std::vector<std::string> &paths);

// Translate Triton dialect to TritonGPU, return null if failed.
void translateTritonToTritonGPUROCM(mlir::ModuleOp &module, int computeCapability,
                                int numWarps, int numStages);

// Translate Triton GPU to mlir LLVM dialect, return null if failed.
void translateTritonGPUROCMToLLVMDialect(mlir::ModuleOp &module,
                                     int computeCapability, bool isROCM);

// Translate mlir LLVM dialect to LLVMIR, return null if failed.
std::unique_ptr<llvm::Module>
translateLLVMDialectToLLVMIR(llvm::LLVMContext *llvmContext,
                             mlir::ModuleOp module, bool isROCM);

// Translate LLVM IR to HSACO code.
std::tuple<std::string, std::string>
translateLLVMIRToHSACO(llvm::Module &module, std::string gfx_arch,
                       std::string gfx_triple, std::string gfx_features);

std::tuple<std::string, std::string>
translateTritonIRToHSACO(mlir::ModuleOp module, std::string gfx_arch,
                         std::string gfx_triple, std::string gfx_features,
                         int numWarps, int numStages,
                         const std::vector<std::string> &names,
                         const std::vector<std::string> &paths);

} // namespace triton
} // namespace mlir

#endif
