#include "triton/codegen/extern_lib.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "triton/codegen/pass.h"

namespace triton {

namespace codegen {

std::unique_ptr<llvm::Module> ExternLib::load(llvm::LLVMContext& ctx) {
  llvm::SMDiagnostic err;
  auto mod = llvm::parseIRFile(this->path_, err, ctx);
  if (!mod) {
    throw std::runtime_error("Failed to load extern lib " + this->name_ +
                             " at " + this->path_);
  }
  return mod;
}

void ExternLib::link(std::unique_ptr<llvm::Module>& llvm,
                     std::unique_ptr<llvm::Module>& mod) {
  // Set triple and data layout to match the target module
  mod->setTargetTriple(llvm->getTargetTriple());
  mod->setDataLayout(llvm->getDataLayout());
  if (llvm::Linker::linkModules(*llvm, std::move(mod))) {
    throw std::runtime_error("Failed to link extern lib " + this->name_ +
                             " at " + this->path_);
  }
}

void LibDevice::opt(llvm::LLVMContext& ctx, std::unique_ptr<llvm::Module>& llvm) {
  // Add nvvm reflect flags to llvm module
  // (i32 4 indicates that the value set here overrides the value in another
  // module we link with. See the LangRef <LangRef.html#module-flags-metadata>
  // for details.)
  llvm::Type* I32 = llvm::Type::getInt32Ty(ctx);
  llvm::Metadata* md_four =
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(I32, 4));
  llvm::Metadata* md_name = llvm::MDString::get(ctx, "nvvm-reflect-ftz");
  llvm::Metadata* md_one =
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(I32, 1));
  llvm::MDNode* reflect = llvm::MDNode::get(ctx, {md_four, md_name, md_one});
  llvm->addModuleFlag(reflect);
}

}  // namespace codegen
}  // namespace triton
