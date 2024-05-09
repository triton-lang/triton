#ifndef TRITON_TARGET_LLVM_IR_MODULE_TRANSLATION_H
#define TRITON_TARGET_LLVM_IR_MODULE_TRANSLATION_H

#include "mlir/Target/LLVMIR/ModuleTranslation.h"

namespace mlir::triton {

std::unique_ptr<llvm::Module>
translateModuleToLLVMIR(mlir::Operation *module, llvm::LLVMContext &llvmContext,
                        llvm::StringRef name = "LLVMDialectModule");

} // namespace mlir::triton

#endif // TRITON_TARGET_LLVM_IR_MODULE_TRANSLATION_H
