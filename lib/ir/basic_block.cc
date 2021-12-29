#include <iostream>
#include <algorithm>
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/type.h"
#include "triton/ir/function.h"

namespace triton {
namespace ir {

class phi_node;


basic_block::basic_block(context &ctx, const std::string &name, function *parent, basic_block* next):
    value(type::get_label_ty(ctx), name), ctx_(ctx), parent_(parent) {
  if(parent_)
    parent_->insert_block(this, next);
}

basic_block* basic_block::create(context &ctx, const std::string &name, function *parent, basic_block* next){
  return new basic_block(ctx, name, parent, next);
}

void basic_block::replace_phi_uses_with(basic_block* before, basic_block* after) {
  for(ir::instruction* i: inst_list_){
    auto* curr_phi = dynamic_cast<ir::phi_node*>(i);
    if(!curr_phi)
      break;
    curr_phi->replace_uses_of_with(before, after);
  }
}

void basic_block::append_instruction(ir::instruction* i){
  i->set_parent(this);
  inst_list_.push_back(i);
}

basic_block* basic_block::split_before(ir::instruction* loc, const std::string& name) {
  basic_block* ret = basic_block::create(ctx_, name, parent_, this);
  // splice instruction list
  auto loc_it = std::find(inst_list_.begin(), inst_list_.end(), loc);
  ret->get_inst_list().splice(ret->get_inst_list().begin(), inst_list_, inst_list_.begin(), loc_it);
  for(ir::instruction* i: ret->get_inst_list())
    i->set_parent(ret);
  // the predecessors of `this` becomes the predecessors of `ret`
  for(ir::basic_block* pred: get_predecessors()){
    auto* term = dynamic_cast<ir::terminator_inst*>(pred->get_inst_list().back());
    assert(term);
    term->replace_uses_of_with(this, ret);
    replace_phi_uses_with(pred, ret);
  }
  ir::branch_inst* br = branch_inst::create(this);
  ret->append_instruction(br);
  return ret;
}

std::vector<basic_block*> basic_block::get_predecessors() const {
  std::vector<basic_block*> ret;
  for(ir::user* u: users_)
    if(auto term = dynamic_cast<ir::terminator_inst*>(u))
      ret.push_back(term->get_parent());
  return ret;
}

std::vector<basic_block*> basic_block::get_successors() const {
  std::vector<basic_block*> ret;
  for(ir::instruction* i: inst_list_)
  for(ir::value* v: i->ops())
    if(auto block = dynamic_cast<ir::basic_block*>(v))
      ret.push_back(block);
  return ret;
}

basic_block::iterator basic_block::get_first_non_phi(){
  auto it = begin();
  for(; it != end(); it++)
  if(!dynamic_cast<phi_node*>(*it)){
    return it;
  }
  return it;
}

}

}
