#include "codegen/shared_copy.h"
#include "ir/module.h"
#include "ir/function.h"
#include "ir/basic_block.h"
#include "ir/instructions.h"

namespace tdl {

namespace codegen{

void place_shared_copy::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list())
    if(dynamic_cast<ir::matmul_inst*>(i)){
      builder.set_insert_point(i);
      ir::value *x = i->get_operand(0);
      ir::value *y = i->get_operand(1);
      ir::instruction *rx = (ir::instruction*)builder.create_copy_to_shared(x);
      ir::instruction *ry = (ir::instruction*)builder.create_copy_to_shared(y);
      x->replace_all_uses_with(rx);
      y->replace_all_uses_with(ry);
      rx->set_operand(0, x);
      ry->set_operand(0, y);
    }
}

}
}
