#include <algorithm>
#include "triton/ir/function.h"
#include "triton/ir/type.h"
#include "triton/ir/module.h"

namespace triton{
namespace ir{


/* Argument */

argument::argument(type *ty, const std::string &name, function *parent, unsigned arg_no)
  : value(ty, name), parent_(parent), arg_no_(arg_no) { }

argument *argument::create(type *ty, const std::string &name,
                          function *parent, unsigned arg_no) {
  return new argument(ty, name, parent, arg_no);
}

function* argument::get_parent() const {
  return parent_;
}

unsigned argument::get_arg_no() const {
  return arg_no_;
}

void argument::accept(visitor *v) {
  v->visit_argument(this);
}


/* function */
function::function(function_type *ty, linkage_types_t linkage,
                   const std::string &name, module *parent)
    : global_object(ty, 0, linkage, name), parent_(parent), fn_ty_(ty), is_kernel_(false) {
  unsigned num_params = fn_ty_->get_num_params();
  if(parent)
    parent->push_function(this);
  // skip if no parameter
  if(num_params == 0)
    return;
  // create arguments
  args_.resize(num_params);
  for(unsigned i = 0; i < num_params; i++){
    type *param_ty = fn_ty_->get_param_ty(i);
    args_[i] = argument::create(param_ty, "", this, i);
  }
}

/* basic block */
void function::insert_block(basic_block *block, basic_block *next) {
  auto it = std::find(blocks_.begin(), blocks_.end(), next);
  blocks_.insert(it, block);
}


function *function::create(function_type *ty, linkage_types_t linkage,
                           const std::string &name, module *mod) {
  return new function(ty, linkage, name, mod);
}


}
}

