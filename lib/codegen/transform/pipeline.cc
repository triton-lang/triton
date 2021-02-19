#include <iostream>
#include "triton/codegen/transform/pipeline.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/utils.h"

namespace triton {
namespace codegen{
namespace transform{

ir::value* _recursive_clone(ir::value *root, std::map<ir::value*, ir::value*>& clones, ir::basic_block* phi_block, const std::string& prefix, ir::builder& builder) {
  if(clones.find(root) != clones.end())
    return clones.at(root);
  auto* inst = dynamic_cast<ir::instruction*>(root);
  // do not clone if not an instruction
  if(!inst)
    return clones.insert({root, root}).first->second;
  // if phi node, return incoming value for provided block
  auto* phi = dynamic_cast<ir::phi_node*>(inst);
  if(phi){
    ir::value* ret = phi->get_value_for_block(phi_block);
    return  clones.insert({root, ret}).first->second;
  }
  // clone instruction by recursively cloning all operands
  ir::instruction* ret = builder.insert(inst->clone());
  builder.set_insert_point(ret);
  for(ir::value *op: inst->ops()){
    ir::value* new_op = _recursive_clone(op, clones, phi_block, prefix, builder);
    ret->replace_uses_of_with(op, new_op);
  }
  return clones.insert({root, ret}).first->second;
}

ir::value* recursive_clone(ir::value *root, ir::basic_block* phi_block,  const std::string& prefix, ir::builder& builder) {
 std::map<ir::value*, ir::value*> tmp;
 return _recursive_clone(root, tmp, phi_block, prefix, builder);
}

void recursive_deps(ir::value* v, std::vector<ir::instruction*>& ret){
 ir::instruction* i = dynamic_cast<ir::instruction*>(v);
 if(!i || i->get_id()==ir::INST_PHI)
   return;
 ret.push_back(i);
 for(ir::user* u: i->get_users())
   recursive_deps(u, ret);
}

void pipeline::run(ir::module &mod) {
  struct pipe_info_t{
    ir::basic_block* header;
    ir::basic_block* block;
    ir::load_inst* load;
    ir::value* mask;
    ir::value* false_value;
    ir::phi_node* ptr;
    ir::cond_branch_inst* back_edge;
  };
  std::vector<pipe_info_t> to_pipeline;

  ir::for_each_instruction(mod, [&](ir::instruction *i){
    if(auto* load = dynamic_cast<ir::load_inst*>(i)){
      ir::value* ptr = load->get_pointer_operand();
      ir::value* mask = nullptr;
      ir::value* false_value = nullptr;
      if(auto* masked_load = dynamic_cast<ir::masked_load_inst*>(load)){
        mask = masked_load->get_mask_operand();
        false_value = masked_load->get_false_value_operand();
      }
      // simple detection of pointer induction variable
      pipe_info_t info;
      info.load = load;
      info.mask = mask;
      info.false_value = false_value;
      if(auto* phi = dynamic_cast<ir::phi_node*>(ptr))
      for(size_t n = 0; n < phi->get_num_incoming(); n++){
        ir::basic_block* block = phi->get_incoming_block(n);
        ir::instruction* term = block->get_inst_list().back();
        if(block == phi->get_parent()){
          info.header = block->get_predecessors()[0];
          info.block = block;
          info.ptr = phi;
          if(auto* br = dynamic_cast<ir::cond_branch_inst*>(term))
            info.back_edge = br;
        }
      }
      if(info.back_edge)
        to_pipeline.push_back(info);
    }
  });

  // do the pipelining
  std::vector<ir::phi_node*> new_loads;
  ir::builder &builder = mod.get_builder();
  for(auto info: to_pipeline){
    ir::value* cond = info.back_edge->get_cond();
    ir::value* false_value = info.false_value;
    ir::type* ty = info.load->get_type();
    // for first pre-fetching
    builder.set_insert_point(info.header->get_inst_list().back());
    ir::value* first_ptr = info.ptr->get_value_for_block(info.header);
    ir::value* first_mask = recursive_clone(cond, info.header, "first_", builder);
    if(info.mask) first_mask = builder.create_and(first_mask, info.mask);
    if(!false_value) false_value = builder.create_splat(ir::undef_value::get(ty->get_scalar_ty()), ty->get_tile_shapes());
    builder.set_insert_point(info.header->get_inst_list().back());
    first_mask = builder.create_splat(first_mask, ty->get_tile_shapes());
    ir::value* first_load = builder.create_masked_load(first_ptr, first_mask, false_value);
    // for next pre-fetching
    builder.set_insert_point(info.block->get_inst_list().back());
    ir::value* next_ptr = info.ptr->get_value_for_block(info.block);
    ir::value* next_mask = info.mask ? builder.create_and(cond, info.mask) : cond;
    next_mask = builder.create_splat(next_mask, ty->get_tile_shapes());
    ir::value* next_load = builder.create_masked_load(next_ptr, next_mask, false_value);
    // phi node
    builder.set_insert_point(info.block->get_first_non_phi());
    ir::phi_node* new_load = builder.create_phi(ty, 2);
    new_load->add_incoming(first_load, info.header);
    new_load->add_incoming(next_load, info.block);
    info.load->replace_all_uses_with(new_load);
    new_loads.push_back(new_load);
  }

  // try to move dot_inst after loads
  // for better overlap of io and compute
  struct move_config_t{
    std::vector<ir::instruction*> insts;
    ir::load_inst* dst;
  };
  std::map<ir::basic_block*, move_config_t> to_move;

  for(ir::function* fn: mod.get_function_list())
  for(ir::basic_block* bb: fn->blocks())
  for(ir::instruction* inst: bb->get_inst_list()){
    if(auto* i = dynamic_cast<ir::dot_inst*>(inst))
      recursive_deps(i, to_move[bb].insts);
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
