#include <algorithm>
#include "codegen/shared_copy.h"
#include "codegen/buffer_info.h"
#include "ir/module.h"
#include "ir/function.h"
#include "ir/basic_block.h"
#include "ir/instructions.h"

namespace tdl {

namespace codegen{

void place_shared_copy::add_copy(ir::value *x, ir::builder &builder) {
  if(auto *i = dynamic_cast<ir::instruction*>(x)){
    ir::basic_block* block = i->get_parent();
    std::cout << "adding copy: " << x->get_name() << " " << block->get_name() << std::endl;
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
}

}
}
