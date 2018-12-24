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
  typedef std::pair<std::string, llvm::BasicBlock*> val_key_t;
  llvm::PHINode *make_phi(llvm::Type *type, unsigned num_values, llvm::BasicBlock *block);
  llvm::Value *add_phi_operands(const std::string& name, llvm::PHINode *&phi);
  llvm::Value *get_value_recursive(const std::string& name, llvm::BasicBlock *block);

public:
  module(const std::string &name, context *ctx);
  llvm::Module* handle();
  llvm::IRBuilder<>& builder();
  // Setters
  void set_value(const std::string& name, llvm::BasicBlock* block, llvm::Value *value);
  void set_value(const std::string& name, llvm::Value* value);
  // Getters
  llvm::Value *get_value(const std::string& name, llvm::BasicBlock* block);
  llvm::Value *get_value(const std::string& name);
  // Seal block -- no more predecessors will be added
  llvm::Value *seal_block(llvm::BasicBlock *block);

private:
  llvm::Module handle_;
  llvm::IRBuilder<> builder_;
  std::map<val_key_t, llvm::Value*> values_;
  std::set<llvm::BasicBlock*> sealed_blocks_;
  std::map<llvm::BasicBlock*, std::map<std::string, llvm::PHINode*>> incomplete_phis_;
};


}
