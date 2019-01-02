#include "ir/value.h"
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

// TODO: automatic naming scheme + update symbol table
void value::set_name(const std::string &name){
  name_ = name;
}


//===----------------------------------------------------------------------===//
//                               use class
//===----------------------------------------------------------------------===//
void use::set(value *val){
  val_ = val;
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
void user::set_operand(unsigned i, value *x){
  assert(i < ops_.size() && "set_operand() out of range!");
  ops_[i] = x;
}

value* user::get_operand(unsigned i){
  assert(i < ops_.size() && "get_operand() out of range!");
  return ops_[i];
}

unsigned user::get_num_operands(){
  return ops_.size();
}

}
}
