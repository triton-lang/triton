#include "ir/function.h"
#include "ir/type.h"

namespace tdl{
namespace ir{


/* Argument */

argument::argument(type *ty, const std::string &name, function *parent, unsigned arg_no)
  : value(ty, name), parent_(parent), arg_no_(arg_no) { }

argument *argument::create(type *ty, const std::string &name,
                          function *parent, unsigned arg_no) {
  return new argument(ty, name, parent, arg_no);
}

/* function */
function::function(function_type *ty, linkage_types_t linkage,
                   const std::string &name, module *parent)
    : global_object(ty, 0, linkage, name), parent_(parent) {
  // create arguments
  function_type *fn_ty = get_function_ty();
  unsigned num_params = fn_ty->get_num_params();
  if(num_params > 0) {
    args_.resize(num_params);
    for(unsigned i = 0; i < num_params; i++){
      type *param_ty = fn_ty->get_param_ty(i);
      args_.push_back(argument::create(param_ty, "", this, i));
    }
  }
}


function *function::create(function_type *ty, linkage_types_t linkage,
                           const std::string &name, module *mod){
  return new function(ty, linkage, name, mod);
}


function_type* function::get_function_ty() const
{ return static_cast<function_type*>(get_type()); }


}
}

