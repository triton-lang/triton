#include <algorithm>
#include <iostream>
#include "triton/ir/basic_block.h"
#include "triton/ir/module.h"
#include "triton/ir/type.h"
#include "triton/ir/constant.h"
#include "triton/ir/function.h"

namespace triton{
namespace ir{

/* Module */
module::module(const std::string &name, builder &builder)
  : name_(name), builder_(builder) {
  sealed_blocks_.insert(nullptr);
}

ir::builder& module::get_builder() {
  return builder_;
}

const std::string& module::get_name() {
  return name_;
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
