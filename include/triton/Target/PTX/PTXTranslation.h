#ifndef TRITON_TARGET_PTXTRANSLATION_H
#define TRITON_TARGET_PTXTRANSLATION_H

#include <memory>
#include <string>

namespace llvm {
class Module;
} // namespace llvm

namespace triton {

// Translate TritonGPU IR to PTX code.
std::string translateLLVMIRToPTX(llvm::Module &module, int cc, int version);

} // namespace triton

#endif
