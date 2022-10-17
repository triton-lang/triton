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
  // https://llvm.org/docs/LangRef.html#module-flags-metadata
  // i32 4: Override the other module.
  // i32 1: Emit an error
  // If both modules specify Override, but the values differ, an error
  // will be emitted.
  llvm::Type* I32 = llvm::Type::getInt32Ty(ctx);
  llvm::Metadata* md_four =
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(I32, 4));
  llvm::Metadata* md_name = llvm::MDString::get(ctx, "nvvm-reflect-ftz");
  llvm::Metadata* md_one =
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(I32, 1));
  llvm::MDNode* reflect = llvm::MDNode::get(ctx, {md_four, md_name, md_one});
  llvm->addModuleFlag(reflect);
}

std::unique_ptr<ExternLib> create_extern_lib(const std::string& lib_name,
                                             const std::string& lib_path) {
  if (lib_name == "libdevice") {
    return std::make_unique<LibDevice>(lib_name, lib_path);
  } else {
    throw std::runtime_error("Unknown external library: " + lib_name);
  }
}

}  // namespace codegen
}  // namespace triton
