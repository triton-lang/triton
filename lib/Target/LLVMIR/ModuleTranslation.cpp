#include "triton/Target/LLVMIR/ModuleTranslation.h"

std::unique_ptr<llvm::Module> mlir::triton::translateModuleToLLVMIR(
    Operation *module, llvm::LLVMContext &llvmContext, llvm::StringRef name) {
  auto llvmMod = mlir::translateModuleToLLVMIR(module, llvmContext, name);

  // Workaround for MLIR not exposing a way to add NoReturn to a function
  // declaration
  auto assertfail = llvmMod->getFunction("__assertfail");
  if (assertfail) {
    assertfail->addFnAttr(llvm::Attribute::NoReturn);
  }
  return llvmMod;
}
