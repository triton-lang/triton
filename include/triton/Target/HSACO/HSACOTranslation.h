#ifndef TRITON_TARGET_HSACOTRANSLATION_H
#define TRITON_TARGET_HSACOTRANSLATION_H

#include <memory>
#include <string>
#include <tuple>

namespace llvm {
class Module;
} // namespace llvm

namespace triton {

// Translate TritonGPU IR to HSACO code.
std::tuple<std::string, std::string> translateLLVMIRToHSACO(llvm::Module& module,
                                                            std::string cc);

} // namespace triton

#endif
