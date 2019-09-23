#include <algorithm>
#include <iostream>
#include "triton/codegen/transform/cts.h"
#include "triton/codegen/instructions.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/type.h"

namespace triton {
namespace codegen{
namespace transform{

// run pass on module
void add_copy(ir::instruction *parent, ir::value *x, ir::builder &builder) {
  auto *i = dynamic_cast<ir::instruction*>(x);
  // not an instruction
  if(!i) {
    builder.set_insert_point(parent);
    ir::value *cts = builder.create_copy_to_shared(x);
    parent->replace_uses_of_with(x, cts);
    return;
  }
  // phi node
  if(auto* phi = dynamic_cast<ir::phi_node*>(x)) {
    for(unsigned i = 0; i < phi->get_num_incoming(); ++i)
      add_copy(phi, phi->get_incoming_value(i), builder);
    return;
  }
  ir::value_id_t id = i->get_id();
  // already in shared memory
  if(storage_info.at(id).first == SHARED)
    return;
  // copy
  builder.set_insert_point_after(i);
  ir::value *cts = builder.create_copy_to_shared(x);
  parent->replace_uses_of_with(x, cts);
}

void cts::run(ir::module &mod) {
  // Add shared copies
  ir::builder &builder = mod.get_builder();
  for(ir::function *fn: mod.get_function_list()){
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *i: block->get_inst_list()){
      auto storage = storage_info.at(i->get_id());
      // copy to shared operands when necessary
      for(size_t k = 0; k < storage.second.size(); k++)
        if(storage.second[k] == SHARED)
          add_copy(i, i->get_operand(k), builder);
    }
  }
}


}
}
}
