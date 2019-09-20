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
namespace analysis{

// run pass on module
bool cts::is_loop_latch(ir::phi_node *phi, ir::instruction *terminator){
  if(phi->get_parent() != terminator->get_parent())
    return false;
  if(auto *br = dynamic_cast<ir::cond_branch_inst*>(terminator))
    return br->get_true_dest() == phi->get_parent()
           || br->get_false_dest() == phi->get_parent();
  else if(dynamic_cast<ir::uncond_branch_inst*>(terminator))
    return false;
  else
    throw std::runtime_error("unreachable");
}



inline bool get_is_shared(ir::value* v) {
  if(dynamic_cast<ir::atomic_cas_inst*>(v))
    return true;
  if(dynamic_cast<ir::trans_inst*>(v))
    return true;
  if(dynamic_cast<ir::copy_to_shared_inst*>(v))
    return true;
  if(dynamic_cast<ir::reduce_inst*>(v))
    return true;
  if(auto *x = dynamic_cast<ir::phi_node*>(v)){
    bool res = true;
    for(unsigned inc = 0; inc < x->get_num_incoming(); inc++)
      res = res && get_is_shared(x->get_incoming_value(inc));
    return res;
  }
  return false;
}

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
  shared_.clear();
  refs_.clear();
  double_.clear();

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

  // Find which buffers are shared
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list())
    if(get_is_shared(i))
      shared_.insert(i);

  // double-buffering
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list()) {
    if(!i->get_type()->is_tile_ty())
      continue;
    // handle phi
    if(auto *phi = dynamic_cast<ir::phi_node*>(i))
    if(is_shared(phi)){
      // determine if the value is in shared memory
      bool is_double = false;
      for(unsigned n = 0; n < phi->get_num_incoming(); n++){
        ir::basic_block *inc_block = phi->get_incoming_block(n);
        ir::instruction *terminator = inc_block->get_inst_list().back();
        is_double = is_double || is_loop_latch(phi, terminator);
      }
      // add to double-buffered
      if(is_double)
        double_.insert(phi);
      // set references of input
      for(unsigned n = 0; n < phi->get_num_incoming(); n++){
        ir::value *inc_val = phi->get_incoming_value(n);
        refs_[inc_val] = phi;
      }
    }
  }
}

// query double-buffered status
bool cts::is_double(ir::value *x)
{ return double_.find(x) != double_.end(); }

// query shared status
bool cts::is_shared(ir::value *x)
{ return shared_.find(x) != shared_.end(); }

// get reference if any
ir::value *cts::get_reference(ir::value *x)
{ return refs_[x]; }



}
}
}
