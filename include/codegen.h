#include <map>
#include <set>
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
  typedef std::pair<const ast::node*, llvm::BasicBlock*> val_key_t;
  llvm::Value *get_value_recursive(const ast::node* node, llvm::BasicBlock *block);

public:
  module(const std::string &name, context *ctx);
  llvm::Module* handle();
  llvm::IRBuilder<>& builder();
  // Setters
  void set_value(const ast::node *node, llvm::BasicBlock* block, llvm::Value *value);
  void set_value(const ast::node* node, llvm::Value* value);
  // Getters
  llvm::Value *get_value(const ast::node *node, llvm::BasicBlock* block);
  llvm::Value *get_value(const ast::node *node);

private:
  llvm::Module handle_;
  llvm::IRBuilder<> builder_;
  std::map<val_key_t, llvm::Value*> values_;
  std::set<llvm::BasicBlock*> sealed_blocks_;
  std::map<val_key_t, llvm::PHINode*> incomplete_phis_;
};


}
