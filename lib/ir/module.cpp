#include "ir/basic_block.h"
#include "ir/module.h"
#include "ir/type.h"
#include "ir/constant.h"
#include "ir/function.h"

namespace tdl{
namespace ir{

/* Module */
module::module(const std::string &name, context &ctx)
  : name_(name), context_(ctx), builder_(ctx) {
  sealed_blocks_.insert(nullptr);
}

ir::builder& module::get_builder() {
  return builder_;
}

ir::context& module::get_context() {
  return context_;
}

void module::set_value(const std::string& name, ir::basic_block *block, ir::value *value){
  values_[val_key_t{name, block}] = value;
}

void module::set_value(const std::string& name, ir::value *value){
  return set_value(name, builder_.get_insert_block(), value);
}

void module::set_type(const std::string& name, ir::basic_block *block, ir::type *type){
  types_[val_key_t{name, block}] = type;
}

void module::set_type(const std::string& name, ir::type *type){
  return set_type(name, builder_.get_insert_block(), type);
}

ir::phi_node* module::make_phi(ir::type *ty, unsigned num_values, ir::basic_block *block){
  basic_block::iterator insert = block->get_first_non_phi();
  if(insert != block->end()){
    builder_.set_insert_point(insert);
  }
  ir::phi_node *res = builder_.create_phi(ty, num_values);
  if(insert != block->end())
    builder_.set_insert_point(block);
  return res;
}

ir::value *module::try_remove_trivial_phis(ir::phi_node *&phi, ir::value** pre_user){
  // find non-self references
  std::set<ir::value*> non_self_ref;
  std::copy_if(phi->ops().begin(), phi->ops().end(), std::inserter(non_self_ref, non_self_ref.begin()),
               [phi](ir::value* op){ return  op != phi && op; });
  // non-trivial
  if(non_self_ref.size() != 1)
    return phi;
  // unique value or self-reference
  ir::value *same = *non_self_ref.begin();
  std::set<ir::user*> users = phi->get_users();
  phi->replace_all_uses_with(same);
  phi->erase_from_parent();
  if(pre_user)
    *pre_user = same;
  for(ir::user* u: users)
  if(auto *uphi = dynamic_cast<ir::phi_node*>(u))
    if(uphi != phi)
      try_remove_trivial_phis(uphi, &same);
  return same;
}


ir::value *module::add_phi_operands(const std::string& name, ir::phi_node *&phi){
  // already initialized
  if(phi->get_num_operands())
    return phi;
  ir::basic_block *block = phi->get_parent();
  for(ir::basic_block *pred: block->get_predecessors()){
    ir::value *value = get_value(name, pred);
    phi->add_incoming(value, pred);
  }
  return try_remove_trivial_phis(phi, nullptr);
}

ir::value *module::get_value_recursive(const std::string& name, ir::basic_block *block) {
  std::cout << "getting value " << name << std::endl;
  ir::value *result;
  auto &preds = block->get_predecessors();
  if(block)
  if(sealed_blocks_.find(block) == sealed_blocks_.end()){
    incomplete_phis_[block][name] = make_phi(get_type(name, block), 1, block);
    result = (ir::value*)incomplete_phis_[block][name];
  }
  else if(preds.size() <= 1){
    bool has_pred = preds.size();
    result = get_value(name, has_pred?preds.front():nullptr);
  }
  else{
    result = make_phi(get_type(name, block), 1, block);
    set_value(name, block, result);
    result = add_phi_operands(name, (ir::phi_node*&)result);
  }
  set_value(name, block, result);
  return result;
}

ir::value *module::get_value(const std::string& name, ir::basic_block *block) {
  ir::basic_block* save_block = builder_.get_insert_block();
  ir::basic_block::iterator save_pt = builder_.get_insert_point();
  val_key_t key(name, block);
  if(values_.find(key) != values_.end()){
    return values_.at(key);
  }
  ir::value *result = get_value_recursive(name, block);
  builder_.set_insert_point(save_block);
  if(save_pt != save_block->end())
    builder_.set_insert_point(save_pt);
  return result;
}

ir::value *module::get_value(const std::string& name) {
  return get_value(name, builder_.get_insert_block());
}

ir::type *module::get_type(const std::string &name, basic_block *block) {
  val_key_t key(name, block);
  if(types_.find(key) != types_.end())
    return types_.at(key);
  assert(block);
  const auto& predecessors = block->get_predecessors();
  if(predecessors.empty())
    return get_type(name, nullptr);
  return get_type(name, predecessors[0]);
}

ir::type *module::get_type(const std::string &name) {
  return types_.at({name, builder_.get_insert_block()});
}

void module::seal_block(ir::basic_block *block){
  for(auto &x: incomplete_phis_[block])
    add_phi_operands(x.first, x.second);
  sealed_blocks_.insert(block);
  incomplete_phis_[block].clear();
}

/* functions */
function *module::get_or_insert_function(const std::string &name, function_type *ty) {
  function *&fn = (function*&)symbols_[name];
  if(fn == nullptr)
    return fn = function::create(ty, global_value::external, name, this);
  return fn;
}


}
}
