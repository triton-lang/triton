#ifndef TRITON_TARGET_AMDGCNTRANSLATION_H
#define TRITON_TARGET_AMDGCNTRANSLATION_H

#include <memory>
#include <string>

namespace llvm {
class Module;
} // namespace llvm

namespace triton {

// Translate LLVM IR to AMDGCN code.
std::string translateLLVMIRToAMDGCN(llvm::Module &module,
                                    const std::string &_proc);

} // namespace triton

#endif
