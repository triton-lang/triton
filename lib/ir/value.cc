#include <cassert>
#include "triton/ir/value.h"
#include "triton/ir/instructions.h"

namespace triton{
namespace ir{

class type;

//===----------------------------------------------------------------------===//
//                               value class
//===----------------------------------------------------------------------===//

value::value(type *ty, const std::string &name): ty_(ty){
  set_name(name);
}

void value::add_use(user *arg) {
  users_.insert(arg);
}

unsigned value::erase_use(user *arg){
  return users_.erase(arg);
}

// TODO: automatic naming scheme + update symbol table
void value::set_name(const std::string &name){
  name_ = name;
}

void value::replace_all_uses_with(value *target){
  throw std::runtime_error("not implemented");
}


//===----------------------------------------------------------------------===//
//                               user class
//===----------------------------------------------------------------------===//
void user::set_operand(unsigned i, value *x) {
  assert(i < ops_.size() && "set_operand() out of range!");
  ops_[i] = x;
  x->add_use(this);
}

value* user::get_operand(unsigned i) const {
  assert(i < ops_.size() && "get_operand() out of range!");
  return ops_[i];
}

unsigned user::get_num_operands() const {
  return num_ops_;
}

unsigned user::get_num_hidden() const {
  return num_hidden_;
}

void user::replace_all_uses_with(value *target) {
  for(auto it = users_.begin(); it != users_.end(); it++){
    (*it)->replace_uses_of_with(this, target);
  }
}

void user::replace_uses_of_with(value *before, value *after) {
  for(size_t i = 0; i < ops_.size(); i++)
    if(ops_[i] == before)
      ops_[i] = after;
  after->add_use(this);
  before->erase_use(this);
}

}
}
