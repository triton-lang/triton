#include "triton/lang/module.h"
#include "triton/ir/module.h"


namespace triton{

namespace lang{

/* Translation unit */
ir::value* translation_unit::codegen(ir::module *mod) const{
  mod->add_new_scope();
  decls_.codegen(mod);
  return nullptr;
}

}

}
