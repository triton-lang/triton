#include "triton/codegen/extern_lib.h"
#include "triton/codegen/pass.h"

#include "llvm/IR/Type.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

namespace triton {

namespace codegen {

void LibDevice::link(llvm::LLVMContext &ctx,
                     std::unique_ptr<llvm::Module> &llvm) {
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

  llvm::legacy::PassManager pass;
  // Cleanup unused functions caused by reflection
  llvm::PassManagerBuilder builder;
  builder.OptLevel = 3;
  builder.SizeLevel = 0;
  builder.populateModulePassManager(pass);

  pass.run(*llvm);
}

}  // namespace codegen
}  // namespace triton
