#ifndef TRITON_TARGET_MUSATRANSLATION_H
#define TRITON_TARGET_MUSATRANSLATION_H

#include <memory>
#include <string>
#include <tuple>

namespace llvm {
class Module;
} // namespace llvm

namespace mlir::triton {

// Translate TritonGPU IR to MUSA binary code.
std::tuple<std::string, std::string>
translateLLVMIRToMUBIN(llvm::Module &module, const std::string &opt_option,
                       int capability, int version);

} // namespace mlir::triton

#endif
