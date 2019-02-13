#include <algorithm>
#include "codegen/shared_copy.h"
#include "ir/module.h"
#include "ir/function.h"
#include "ir/basic_block.h"
#include "ir/instructions.h"

namespace tdl {

namespace codegen{

void place_shared_copy::add_copies(ir::value *x, ir::builder &builder) {
  if(auto *phi = dynamic_cast<ir::phi_node*>(x)) {
    for(auto *op: phi->ops())
      add_copies(op, builder);
  }
  else {
    if(auto *i = dynamic_cast<ir::instruction*>(x)){
      ir::basic_block* block = i->get_parent();
      auto it = std::find(block->begin(), block->end(), i);
      builder.set_insert_point(++it);
    }
    ir::instruction *rx = (ir::instruction*)builder.create_copy_to_shared(x);
    x->replace_all_uses_with(rx);
    rx->set_operand(0, x);
  }
}

void place_shared_copy::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list())
    if(dynamic_cast<ir::matmul_inst*>(i)){
      add_copies(i->get_operand(0), builder);
      add_copies(i->get_operand(1), builder);
    }
}

}
}
