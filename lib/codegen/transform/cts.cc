#include "triton/codegen/transform/cts.h"
#include "triton/codegen/instructions.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"

namespace triton {
namespace codegen{
namespace transform{

inline bool is_shared(ir::value *v) {
  auto *i = dynamic_cast<ir::instruction*>(v);
  if(!i)
    return false;
  return storage_info.at(i->get_id()).first == codegen::SHARED;
}

// run pass on module
void add_copy(ir::instruction *parent, ir::value *x, ir::builder &builder, bool to_shared) {
  auto *i = dynamic_cast<ir::instruction*>(x);
  // not an instruction
  if(!i) {
    builder.set_insert_point(parent);
    ir::value *copy;
    if(to_shared)
      copy = builder.create_copy_to_shared(x);
    else
      copy = builder.create_copy_from_shared(x);
    parent->replace_uses_of_with(x, copy);
    return;
  }
  // phi node
  if(auto* phi = dynamic_cast<ir::phi_node*>(x)) {
    for(unsigned i = 0; i < phi->get_num_incoming(); ++i)
      add_copy(phi, phi->get_incoming_value(i), builder, to_shared);
    return;
  }
  ir::value_id_t id = i->get_id();
  // already in shared memory
  if(to_shared && storage_info.at(id).first == SHARED)
    return;
  // copy
  builder.set_insert_point_after(i);
  ir::value *copy;
  if(to_shared)
    copy = builder.create_copy_to_shared(x);
  else
    copy = builder.create_copy_from_shared(x);
  parent->replace_uses_of_with(x, copy);
}

void cts::run(ir::module &mod) {
  // Add shared copies
  ir::builder &builder = mod.get_builder();
  for(ir::function *fn: mod.get_function_list()){
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *i: block->get_inst_list()){
      auto storage = storage_info.at(i->get_id());
      // copy to shared operands
      for(size_t k = 0; k < storage.second.size(); k++)
        if(storage.second[k] == SHARED)
          add_copy(i, i->get_operand(k), builder, true);
      // copy from shared operands
      for(size_t k = 0; k < storage.second.size(); k++)
        if(storage.second[k] == DISTRIBUTED &&
           is_shared(i->get_operand(k))){
          add_copy(i, i->get_operand(k), builder, false);
        }
    }
  }
}


}
}
}
