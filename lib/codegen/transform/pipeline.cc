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
 if(!i || i->get_parent() != block)
   return;
 if(i->get_id()==ir::INST_PHI)
   return;
 ret.push_back(i);
 for(ir::user* u: i->get_users())
   recursive_deps(u, block, ret);
}

void pipeline::run(ir::module &mod) {
  // *Very* conservative heuristics for pre-fetching.
  // A load instruction can be pipelined if:
  //   - the pointer is a phi node that references a value
  //     in its basic block (i.e., pointer induction variable)
  //   - the load has only  a single use in a dot instruction
  // As more use cases become apparent, this pass will be improved
  std::vector<std::pair<ir::load_inst*, ir::phi_node*>> to_pipeline;
  ir::for_each_instruction(mod, [&](ir::instruction *i){
    if(auto* load = dynamic_cast<ir::load_inst*>(i)){
      ir::phi_node* ptr = dynamic_cast<ir::phi_node*>(load->get_pointer_operand());
      auto users = load->get_users();
      if(ptr && ptr->get_incoming_block(1) == ptr->get_parent()
         && users.size() == 1 && dynamic_cast<ir::dot_inst*>(*users.begin()))
        to_pipeline.push_back({load, ptr});
    }});
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


  // try to move dot_inst after loads
  // for better overlap of io and compute
  struct move_config_t{
    std::vector<ir::instruction*> insts;
    ir::load_inst* dst;
  };
  std::map<ir::basic_block*, move_config_t> to_move;

  if(has_copy_async_){
    for(ir::function* fn: mod.get_function_list())
    for(ir::basic_block* bb: fn->blocks())
    for(ir::instruction* inst: bb->get_inst_list()){
      if(auto* i = dynamic_cast<ir::dot_inst*>(inst))
        recursive_deps(i, bb, to_move[bb].insts);
      if(auto* i = dynamic_cast<ir::load_inst*>(inst))
        to_move[bb].dst = i;
    }

    for(auto& x: to_move){
      builder.set_insert_point_after(x.second.dst);
      for(ir::instruction* i: x.second.insts){
        x.first->erase(i);
        builder.insert(i);
      }
    }
  }


}

}
}
}
