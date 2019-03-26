#include <algorithm>
#include "triton/codegen/shared_copy.h"
#include "triton/codegen/buffer_info.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"

namespace triton {

namespace codegen{

void place_shared_copy::add_copy(ir::value *x, ir::builder &builder) {
  if(auto *i = dynamic_cast<ir::instruction*>(x)){
    ir::basic_block* block = i->get_parent();
    auto it = std::find(block->begin(), block->end(), i);
    builder.set_insert_point(++it);
  }
  ir::instruction *rx = (ir::instruction*)builder.create_copy_to_shared(x);
  x->replace_all_uses_with(rx);
  rx->set_operand(0, x);
}

void place_shared_copy::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list())
  if(info_->is_shared(i) && !info_->is_double(i))
    add_copy(i, builder);

  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list())
    if(auto* cts = dynamic_cast<ir::copy_to_shared_inst*>(i))
      info_->replace(cts->get_operand(0), cts);
}

}
}
