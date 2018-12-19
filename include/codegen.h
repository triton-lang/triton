#include "ast.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"

namespace tdl
{

class context {
public:
  context();
  llvm::LLVMContext* handle();

private:
  llvm::LLVMContext handle_;
};

class module {
public:
  module(const std::string &name, context *ctx);
  llvm::Module* handle();
  llvm::IRBuilder<>& builder();

private:
  llvm::Module handle_;
  llvm::IRBuilder<> builder_;
};


}
