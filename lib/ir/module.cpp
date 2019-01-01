#include "ir/module.h"

namespace tdl{
namespace ir{

/* Module */
module::module(const std::string &name, context *ctx)
  : handle_(name.c_str(), *ctx->handle()), builder_(*ctx->handle()) {
  sealed_blocks_.insert(nullptr);
}

Module* module::handle() {
  return &handle_;
}

IRBuilder<>& module::builder() {
  return builder_;
}

void module::set_value(const std::string& name, BasicBlock *block, Value *value){
  values_[val_key_t{name, block}] = value;
}

void module::set_value(const std::string& name, Value* value){
  return set_value(name, builder_.GetInsertBlock(), value);
}

PHINode* module::make_phi(Type *type, unsigned num_values, BasicBlock *block){
  Instruction* instr = block->getFirstNonPHIOrDbg();
  if(instr)
    builder_.SetInsertPoint(instr);
  PHINode *res = builder_.CreatePHI(type, num_values);
  if(instr)
    builder_.SetInsertPoint(block);
  return res;
}

Value *module::add_phi_operands(const std::string& name, PHINode *&phi){
  BasicBlock *block = phi->getParent();
  for(BasicBlock *pred: predecessors(block)){
    Value *value = get_value(name, pred);
    phi->addIncoming(value, pred);
  }
  return phi;
}

Value *module::get_value_recursive(const std::string& name, BasicBlock *block) {
  Value *result;
  if(sealed_blocks_.find(block) == sealed_blocks_.end()){
    Value *pred = get_value(name, *pred_begin(block));
    incomplete_phis_[block][name] = make_phi(pred->getType(), 1, block);
    result = (Value*)incomplete_phis_[block][name];
  }
  else if(pred_size(block) <= 1){
    bool has_pred = pred_size(block);
    result = get_value(name, has_pred?*pred_begin(block):nullptr);
  }
  else{
    Value *pred = get_value(name, *pred_begin(block));
    result = make_phi(pred->getType(), 1, block);
    set_value(name, block, result);
    add_phi_operands(name, (PHINode*&)result);
  }
  set_value(name, block, result);
  return result;
}

Value *module::get_value(const std::string& name, BasicBlock *block) {
  val_key_t key(name, block);
  if(values_.find(key) != values_.end()){
    return values_.at(key);
  }
  return get_value_recursive(name, block);
}

Value *module::get_value(const std::string& name) {
  return get_value(name, builder_.GetInsertBlock());
}

Value *module::seal_block(BasicBlock *block){
  for(auto &x: incomplete_phis_[block])
    add_phi_operands(x.first, x.second);
  sealed_blocks_.insert(block);
}

}
}
