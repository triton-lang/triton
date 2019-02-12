#include "codegen/buffer_info.h"
#include "ir/module.h"
#include "ir/function.h"
#include "ir/basic_block.h"
#include "ir/instructions.h"
#include "ir/type.h"

namespace tdl {

namespace codegen{


// run pass on module
void buffer_info_pass::run(ir::module &mod) {
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list()) {
    if(!i->get_type()->is_tile_ty())
      continue;
    // handle phi
    if(auto *phi = dynamic_cast<ir::phi_node*>(i)){
      // determine if the value is in shared memory
      bool is_shared = true;
      bool is_double = false;
      for(unsigned n = 0; n < phi->get_num_incoming(); n++){
        ir::value *inc_val = phi->get_incoming_value(n);
        ir::value *inc_block = phi->get_incoming_block(n);
        is_shared = is_shared &&  dynamic_cast<ir::copy_to_shared_inst*>(inc_val);
        is_double = is_double || inc_block == phi->get_parent();
      }
      // add to shared
      if(is_shared)
        shared_.insert(phi);
      // add to double-buffered
      if(is_double)
        double_.insert(phi);
      // set references of input
      for(unsigned n = 0; n < phi->get_num_incoming(); n++){
        ir::value *inc_val = phi->get_incoming_value(n);
        assert(refs_[inc_val] == nullptr);
        refs_[inc_val] = phi;
      }
    }
    // handle shared copy
    if(auto *copy = dynamic_cast<ir::copy_to_shared_inst*>(i))
      shared_.insert(copy);
  }
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
