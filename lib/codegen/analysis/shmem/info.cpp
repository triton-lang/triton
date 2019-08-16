#include <algorithm>
#include "triton/codegen/analysis/shmem/info.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/type.h"

namespace triton {

namespace codegen{
namespace analysis{
namespace shmem{

// run pass on module
bool info::is_loop_latch(ir::phi_node *phi, ir::instruction *terminator){
  if(phi->get_parent() != terminator->get_parent())
    return false;
  if(auto *br = dynamic_cast<ir::cond_branch_inst*>(terminator))
    return br->get_true_dest() == phi->get_parent()
           || br->get_false_dest() == phi->get_parent();
  else if(auto *br = dynamic_cast<ir::uncond_branch_inst*>(terminator))
    return false;
  else
    throw std::runtime_error("unreachable");
}

void info::replace(ir::value* before, ir::value *after) {
  shared_.erase(before);
  shared_.insert(after);
  if(refs_.find(before) != refs_.end()){
    ir::value* v = refs_.at(before);
    refs_.erase(before);
    refs_.insert({after, v});
  }
}

inline bool get_is_shared(ir::value* v) {
  if(auto x = dynamic_cast<ir::atomic_cas_inst*>(v))
    return true;
  if(auto x = dynamic_cast<ir::trans_inst*>(v))
    return true;
  if(auto x = dynamic_cast<ir::copy_to_shared_inst*>(v))
    return true;
  if(auto x = dynamic_cast<ir::reduce_inst*>(v))
    return true;
  if(auto x = dynamic_cast<ir::phi_node*>(v)){
    bool res = true;
    for(unsigned inc = 0; inc < x->get_num_incoming(); inc++)
      res = res && get_is_shared(x->get_incoming_value(inc));
    return res;
  }
  return false;
}

void add_copy(ir::value *x, ir::builder &builder) {
  if(auto phi = dynamic_cast<ir::phi_node*>(x)){
    for(unsigned i = 0; i < phi->get_num_incoming(); ++i)
      add_copy(phi->get_incoming_value(i), builder);
  }
  else {
    if(get_is_shared(x))
      return;
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

void info::run(ir::module &mod) {
  // Add shared copies
  for(ir::function *fn: mod.get_function_list()){
    ir::builder builder(mod.get_context());
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *i: block->get_inst_list()){
      if(dynamic_cast<ir::dot_inst*>(i))
      if(i->get_operand(1)->get_type()->get_tile_shapes()[1]->get_value() != 1){
        add_copy(i->get_operand(0), builder);
        add_copy(i->get_operand(1), builder);
      }
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
bool info::is_double(ir::value *x)
{ return double_.find(x) != double_.end(); }

// query shared status
bool info::is_shared(ir::value *x)
{ return shared_.find(x) != shared_.end(); }

// get reference if any
ir::value *info::get_reference(ir::value *x)
{ return refs_[x]; }



}
}
}
}
