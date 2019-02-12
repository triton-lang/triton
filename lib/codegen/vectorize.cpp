#include "codegen/vectorize.h"
#include "codegen/tune.h"
#include "ir/module.h"
#include "ir/function.h"
#include "ir/basic_block.h"
#include "ir/instructions.h"

namespace tdl {

namespace codegen{

void vectorize::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list())
    if(dynamic_cast<ir::copy_to_shared_inst*>(i)){
      builder.set_insert_point(i);
      ir::value *x = i->get_operand(0);
      ir::instruction *rx = (ir::instruction*)builder.create_vectorize(x);
      x->replace_all_uses_with(rx);
      rx->set_operand(0, x);
      params_->copy(rx, x);
    }
}

}
}
