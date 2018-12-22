#include <unordered_map>
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
  void value(const ast::node* node, llvm::Value* value);
  llvm::Value *value(const ast::node *node);

private:
  llvm::Module handle_;
  llvm::IRBuilder<> builder_;
  std::unordered_map<const ast::node*, llvm::Value*> values_;
};


}
