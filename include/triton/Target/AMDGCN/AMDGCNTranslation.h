#ifndef TRITON_TARGET_AMDGCNTRANSLATION_H
#define TRITON_TARGET_AMDGCNTRANSLATION_H

#include <string>
#include <tuple>

namespace llvm {
class Module;
} // namespace llvm

namespace triton {

// Translate LLVM IR to AMDGCN code.
std::tuple<std::string, std::string>
translateLLVMIRToAMDGCN(llvm::Module &module, std::string cc);

} // namespace triton

#endif
