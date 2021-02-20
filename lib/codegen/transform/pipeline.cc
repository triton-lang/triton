#include <iostream>
#include <algorithm>
#include "triton/codegen/transform/pipeline.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/utils.h"

namespace triton {
namespace codegen{
namespace transform{


void recursive_deps(ir::value* v, ir::basic_block* block, std::vector<ir::instruction*>& ret){
 ir::instruction* i = dynamic_cast<ir::instruction*>(v);
 if(!i || i->get_parent() != block);
  ret.push_back(i);
 if(i->get_id()==ir::INST_PHI)
   return;
 for(ir::user* u: i->get_users())
   recursive_deps(u, block, ret);
}

void pipeline::run(ir::module &mod) {
  // Crude conservative heuristics for pre-fetching.
  // A load instruction can be pipelined if:
  //   - the pointer is a phi node that references a value
  //     in its basic block (i.e., pointer induction variable)
  //   - it is not used as an incoming value for a phi-node in the same block
  std::vector<std::pair<ir::load_inst*, ir::phi_node*>> to_pipeline;
  ir::for_each_instruction(mod, [&](ir::instruction *i){
    if(auto* load = dynamic_cast<ir::load_inst*>(i)){
      ir::phi_node* ptr = dynamic_cast<ir::phi_node*>(load->get_pointer_operand());
      if(!ptr) return;
      std::vector<ir::instruction*> users;
      recursive_deps(load, load->get_parent(), users);
      auto it = std::find_if(users.begin(), users.end(), [&](ir::instruction* i) {
          return i->get_id() == ir::INST_PHI;
      });
      if(it != users.end())
        return;
      to_pipeline.push_back({load, ptr});
    }});

  for(auto x: to_pipeline)
    std::cout << x.second->get_name() << std::endl;

  // do the pipelining
  std::vector<ir::phi_node*> new_loads;
  ir::builder &builder = mod.get_builder();
  for(auto info: to_pipeline){
    ir::load_inst* load = info.first;
    ir::phi_node* ptr   = info.second;
    ir::basic_block* block = load->get_parent();
    ir::basic_block* header = block->get_predecessors()[0];
    auto* block_br = dynamic_cast<ir::cond_branch_inst*>(block->get_inst_list().back());
    auto* header_br = dynamic_cast<ir::cond_branch_inst*>(header->get_inst_list().back());
    assert(block_br);
    assert(header_br);
    ir::type* ty = load->get_type();
    // pre-fetch first iteration
    builder.set_insert_point(header->get_inst_list().back());
    ir::value* first_ptr = ptr->get_value_for_block(header);
    ir::value* first_mask = builder.create_splat(header_br->get_cond(), ty->get_tile_shapes());
    ir::value* false_value;
    if(auto* masked_load = dynamic_cast<ir::masked_load_inst*>(load)){
      first_mask = builder.create_and(first_mask, masked_load->get_mask_operand());
      false_value = masked_load->get_false_value_operand();
    }
    else
      false_value = builder.create_splat(ir::undef_value::get(ty->get_scalar_ty()), ty->get_tile_shapes());
    ir::value* first_load = builder.create_masked_load(first_ptr, first_mask, false_value);
    // pre-fetch next iteration
    builder.set_insert_point(block->get_inst_list().back());
    ir::value* next_ptr = ptr->get_value_for_block(block);
    ir::value* next_mask = builder.create_splat(block_br->get_cond(), ty->get_tile_shapes());
    if(auto* masked_load = dynamic_cast<ir::masked_load_inst*>(load))
      next_mask = builder.create_and(next_mask, masked_load->get_mask_operand());
    ir::value* next_load = builder.create_masked_load(next_ptr, next_mask, false_value);
    // phi node
    builder.set_insert_point(block->get_first_non_phi());
    ir::phi_node* new_load = builder.create_phi(ty, 2);
    new_load->add_incoming(first_load, header);
    new_load->add_incoming(next_load, block);
    load->replace_all_uses_with(new_load);
    new_loads.push_back(new_load);
  }
}

}
}
}
