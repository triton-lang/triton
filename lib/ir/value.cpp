#include "ir/value.h"
#include <iostream>
#include <cassert>

namespace tdl{
namespace ir{

class type;

//===----------------------------------------------------------------------===//
//                               value class
//===----------------------------------------------------------------------===//

value::value(type *ty, const std::string &name): ty_(ty){
  set_name(name);
}

void value::add_use(use arg) {
  uses_.push_back(arg);
}

// TODO: automatic naming scheme + update symbol table
void value::set_name(const std::string &name){
  name_ = name;
}

void value::replace_all_uses_with(value *target){
  throw std::runtime_error("not implemented");
}


//===----------------------------------------------------------------------===//
//                               use class
//===----------------------------------------------------------------------===//
void use::set(value *val){
  val_ = val;
  val_->add_use(*this);
}

value *use::operator=(value *rhs){
  set(rhs);
  return rhs;
}

const use &use::operator=(const use &rhs){
  set(rhs.val_);
  return rhs;
}

//===----------------------------------------------------------------------===//
//                               user class
//===----------------------------------------------------------------------===//
void user::set_operand(unsigned i, value *x) {
  assert(i < ops_.size() && "set_operand() out of range!");
  ops_[i] = x;
}

value* user::get_operand(unsigned i) {
  assert(i < ops_.size() && "get_operand() out of range!");
  return ops_[i];
}

unsigned user::get_num_operands() const {
  return ops_.size();
}

void user::replace_all_uses_with(value *target) {
  for(use &u: uses_)
  if(auto *usr = dynamic_cast<user*>(u.get())){
    std::cout << "replacing " << this << " by " << target << " in " << usr << std::endl;
    usr->replace_uses_of_with(this, target);
  }
}

void user::replace_uses_of_with(value *before, value *after) {
  for(use &u: ops_)
    if(u.get() == before)
      u.set(after);
}

}
}
