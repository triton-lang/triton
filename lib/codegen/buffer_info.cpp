#include "triton/codegen/buffer_info.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/type.h"

namespace tdl {

namespace codegen{


// run pass on module
bool buffer_info_pass::is_loop_latch(ir::phi_node *phi, ir::value *terminator){
  if(auto *br = dynamic_cast<ir::cond_branch_inst*>(terminator))
    return br->get_true_dest() == phi->get_parent()
           || br->get_false_dest() == phi->get_parent();
  else if(auto *br = dynamic_cast<ir::uncond_branch_inst*>(terminator))
    return br->get_dest() == phi->get_parent();
  else
    throw std::runtime_error("unreachable");
}

void buffer_info_pass::run(ir::module &mod) {
  // Find which buffers are shared
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list())
    if(dynamic_cast<ir::matmul_inst*>(i)){
      shared_.insert(i->get_operand(0));
      shared_.insert(i->get_operand(1));
    }

  // Handles phi nodes
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
        ir::value *terminator = inc_block->get_inst_list().back();
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

  for(auto &ref: refs_)
    shared_.insert(ref.first);
}

// query double-buffered status
bool buffer_info_pass::is_double(ir::value *x)
{ return double_.find(x) != double_.end(); }

// query shared status
bool buffer_info_pass::is_shared(ir::value *x)
{ return shared_.find(x) != shared_.end(); }

// get reference if any
ir::value *buffer_info_pass::get_reference(ir::value *x)
{ return refs_[x]; }



}
}
