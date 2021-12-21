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

void get_induction_vars(ir::value* cond, std::set<ir::phi_node*>& phis) {
  auto instr = dynamic_cast<ir::instruction*>(cond);
  for (auto op : instr->ops()) {
    if (auto phi_op = dynamic_cast<ir::phi_node*>(op)) {
      phis.insert(phi_op);
      return;
    }
    if (dynamic_cast<ir::instruction*>(op))
      get_induction_vars(op, phis);
  }
}

/// assume incoming block is 1
ir::value* rematerialize_vals(ir::builder& builder, ir::basic_block* block, ir::value* v,
                              std::map<ir::phi_node*, ir::value*>& prev_phi_vals) {
  ir::instruction* i = dynamic_cast<ir::instruction*>(v);
  if(!i || i->get_parent() != block)
    return v;
  if(ir::phi_node* phi = dynamic_cast<ir::phi_node*>(v)) {
    if (prev_phi_vals.find(phi) == prev_phi_vals.end())
      throw std::runtime_error("Don't have that phi node\n");
    return prev_phi_vals.at(phi);
  }

  std::vector<ir::value*> new_ops;
  for(ir::value* op: i->ops()){
    new_ops.push_back(rematerialize_vals(builder, block, op, prev_phi_vals));
  }
  ir::instruction* ret = i->clone();
  for(size_t k = 0; k < new_ops.size(); k++)
    ret->set_operand(k, new_ops[k]);
  builder.insert(ret);
  return ret;
}

ir::value* rematerialize(ir::builder& builder, ir::basic_block* block,
                         ir::value* v, size_t phi_idx){
  ir::instruction* i = dynamic_cast<ir::instruction*>(v);
  if(!i || i->get_parent() != block)
    return v;
  if(ir::phi_node* phi = dynamic_cast<ir::phi_node*>(v))
    return phi->get_incoming_value(phi_idx);

  std::vector<ir::value*> new_ops;
  for(ir::value* op: i->ops()){
    new_ops.push_back(rematerialize(builder, block, op, phi_idx));
  }
  ir::instruction* ret = i->clone();
  for(size_t k = 0; k < new_ops.size(); k++)
    ret->set_operand(k, new_ops[k]);
  builder.insert(ret);
  return ret;
}

/// moving the prev phi vals to the next iteration
std::map<ir::phi_node*, ir::value*> update_prev_phi_vals(
  ir::builder& builder, ir::basic_block* block, std::map<ir::phi_node*, ir::value*>& prev_phi_vals) {
  std::map<ir::phi_node*, ir::value*> next_phi_vals;
  for (auto &[phi, val] : prev_phi_vals) {
    next_phi_vals[phi] = rematerialize_vals(builder, block, phi->get_incoming_value(1), prev_phi_vals);
  }
  return next_phi_vals;
}

void finalize_iv_vals(ir::builder& builder, ir::basic_block* block, std::map<ir::phi_node*, ir::value*>& load_ivs,
                                            std::map<ir::phi_node*, ir::value*>& next_load_ivs) {
  for (auto& [phi, val] : load_ivs) {
    if (auto new_phi = dynamic_cast<ir::phi_node*>(val)) {
      ir::value* next_k = rematerialize_vals(builder, block, phi->get_incoming_value(1), load_ivs);
      assert(new_phi->get_num_operands() == 1 && "should be incomplete phi");
      new_phi->add_incoming(next_k, phi->get_incoming_block(1));
      // cache next_k (to be used by next_mask)
      next_load_ivs[phi] = next_k;
    } else
      throw std::runtime_error("must be phi");
  }
}

struct pipeline_info_t {
  ir::load_inst* load;
  ir::phi_node* ptr;
  ir::dot_inst* dot;

  pipeline_info_t(ir::load_inst* load, ir::phi_node* ptr, ir::dot_inst* dot)
    : load(load), ptr(ptr), dot(dot) {}
};

void pipeline::run(ir::module &mod) {
  if (num_stages_ <= 1)
    return;
  // *Very* conservative heuristics for pre-fetching.
  // A load instruction can be pipelined if:
  //   - the pointer is a phi node that references a value
  //     in its basic block (i.e., pointer induction variable)
  //   - the load has only  a single use in a dot instruction
  // As more use cases become apparent, this pass will be improved
  std::vector<pipeline_info_t> to_pipeline;
  ir::for_each_instruction(mod, [&](ir::instruction *i){
    if(auto* load = dynamic_cast<ir::load_inst*>(i)){
      ir::phi_node* ptr = dynamic_cast<ir::phi_node*>(load->get_pointer_operand());
      auto users = load->get_users();
      auto dot = dynamic_cast<ir::dot_inst*>(*users.begin());
      if(ptr && ptr->get_incoming_block(1) == ptr->get_parent()
         && users.size() == 1 && dot)
        to_pipeline.push_back({load, ptr, dot});
    }});
  // do the pipelining
  std::vector<ir::phi_node*> new_loads;
  ir::builder &builder = mod.get_builder();
  const int num_stages = num_stages_;
  std::vector<std::pair<ir::phi_node*, std::vector<ir::value*>>> preheader_loads; // Used to reorder loads
  for(auto info: to_pipeline){
    ir::load_inst* load = info.load;
    ir::phi_node* ptr   = info.ptr;
    ir::basic_block* block = load->get_parent();
    ir::basic_block* header = block->get_predecessors()[0];
    auto* block_br = dynamic_cast<ir::cond_branch_inst*>(block->get_inst_list().back());
    auto* header_br = dynamic_cast<ir::cond_branch_inst*>(header->get_inst_list().back());
    assert(block_br);
    assert(header_br);
    ir::type* ty = load->get_type();
    // multi-stage pipe
    if (has_copy_async_ && num_stages > 2) {
      ir::value* header_cond = header_br->get_cond();
      ir::value* block_cond = block_br->get_cond();
      // 1. collect induction variables
      std::set<ir::phi_node*> induction_vars;
      get_induction_vars(block_cond, induction_vars);

      std::vector<ir::value*> first_ptrs(num_stages-1);
      std::vector<ir::value*> first_loads(num_stages-1);
      std::vector<ir::value*> first_masks(num_stages-1);
      std::vector<ir::value*> loop_conds(num_stages-1);

      std::map<ir::phi_node*, ir::value*> prev_phi_vals;
      // initialize prev_phi_vals
      // Add all phi nodes. The following DCE pass will delete dead ones.
      for (ir::instruction *instr : block->get_inst_list())
        if (auto *phi = dynamic_cast<ir::phi_node*>(instr))
          if (phi->get_incoming_block(1) == block)
            prev_phi_vals[phi] = phi->get_value_for_block(header);

      builder.set_insert_point(header->get_inst_list().back());
      first_ptrs[0] = ptr->get_value_for_block(header);
      loop_conds[0] = header_cond;
      first_masks[0] = builder.create_splat(loop_conds[0], ty->get_block_shapes());
      ir::value* false_value = nullptr;
      if (auto* masked_load = dynamic_cast<ir::masked_load_inst*>(load)) {
        ir::value* remat_mask =rematerialize_vals(builder, block, masked_load->get_mask_operand(), prev_phi_vals) ;
        ir::value* remat_false_value = 
            rematerialize_vals(builder, block, masked_load->get_false_value_operand(), prev_phi_vals);
        first_masks[0] = builder.create_and(first_masks[0], remat_mask);
        false_value = remat_false_value;
      } else
        false_value = builder.create_splat(ir::undef_value::get(ty->get_scalar_ty()), ty->get_block_shapes());
      first_loads[0] = builder.create_masked_load(first_ptrs[0], first_masks[0], false_value, load->get_cache_modifier());

      for (int stage = 1; stage < num_stages-1; ++stage) {
        // mask is the loop condition of the previous iteration
        loop_conds[stage] = rematerialize_vals(builder, block, block_cond, prev_phi_vals);
        prev_phi_vals = update_prev_phi_vals(builder, block, prev_phi_vals);
        first_ptrs[stage] = rematerialize_vals(builder, block, ptr, prev_phi_vals);
        first_masks[stage] = builder.create_splat(loop_conds[stage], ty->get_block_shapes());
        if (auto* masked_load = dynamic_cast<ir::masked_load_inst*>(load)) {
          ir::value* remat_mask = rematerialize_vals(builder, block, masked_load->get_mask_operand(), prev_phi_vals);
          ir::value* remat_false_value = 
              rematerialize_vals(builder, block, masked_load->get_false_value_operand(), prev_phi_vals);
          first_masks[stage] = builder.create_and(first_masks[stage], remat_mask);
          false_value = remat_false_value;
        }
        first_loads[stage] = builder.create_masked_load(first_ptrs[stage], first_masks[stage], false_value, load->get_cache_modifier());
      }

      // create new phis for induction variables
      builder.set_insert_point(block->get_first_non_phi());
      std::map<ir::phi_node*, ir::value*> load_ivs;
      std::map<ir::phi_node*, ir::value*> next_load_ivs;
      for (auto& [iv, val] : prev_phi_vals) {
        ir::phi_node* pn = builder.create_phi(iv->get_type(), 2);
        pn->add_incoming(prev_phi_vals[iv], header);
        load_ivs[iv] = pn;
      }
      // add incoming for phis & update next_load_ivs
      finalize_iv_vals(builder, block, load_ivs, next_load_ivs);
        
      // pre-fetch next iteration
      builder.set_insert_point(block->get_inst_list().back());
//      ir::value* next_ptr = ptr->get_value_for_block(block);
      ir::value* next_ptr = rematerialize_vals(builder, block, ptr->get_value_for_block(block), load_ivs);
      ir::value* next_mask = builder.create_splat(
          rematerialize_vals(builder, block, block_cond, load_ivs), ty->get_block_shapes());
      if (auto* masked_load = dynamic_cast<ir::masked_load_inst*>(load)) {
        ir::value* remat_mask = rematerialize_vals(builder, block, masked_load->get_mask_operand(), next_load_ivs);
        // TODO: false may depends on some other phi nodes
        ir::value* remat_false_value = 
            rematerialize_vals(builder, block, masked_load->get_false_value_operand(), next_load_ivs);
        next_mask = builder.create_and(next_mask, remat_mask);
        false_value = remat_false_value;
      }
      ir::value* next_load = builder.create_masked_load(next_ptr, next_mask, false_value, load->get_cache_modifier());


      // phi node
      ptr->set_incoming_value(0, first_ptrs.back());
      builder.set_insert_point(block->get_first_non_phi());
      // nested phis for load
      std::vector<ir::phi_node*> new_load_phis(num_stages-1);
      for (auto& pn : new_load_phis)
        pn = builder.create_phi(ty, 2);
      for (int i=0; i<num_stages-2; ++i) {
        new_load_phis[i]->add_incoming(first_loads[i], header);
        new_load_phis[i]->add_incoming(new_load_phis[i+1], block);
      }
      new_load_phis.back()->add_incoming(first_loads.back(), header);
      new_load_phis.back()->add_incoming(next_load, block);
      load->replace_all_uses_with(new_load_phis.front());
      new_loads.push_back(new_load_phis.back());

      // record first_loads to reorder them
      preheader_loads.push_back({new_load_phis.front(), first_loads});
    } else {
      // pre-fetch first iteration
      builder.set_insert_point(header->get_inst_list().back());
      ir::value* first_ptr = ptr->get_value_for_block(header);
      ir::value* first_mask = builder.create_splat(header_br->get_cond(), ty->get_block_shapes());
      ir::value* false_value;
      if(auto* masked_load = dynamic_cast<ir::masked_load_inst*>(load)){
        ir::value* remat_mask = rematerialize(builder, block, masked_load->get_mask_operand(), 0);
        ir::value* remat_false_value = rematerialize(builder, block, masked_load->get_false_value_operand(), 0);
        first_mask = builder.create_and(first_mask, remat_mask);
        false_value = remat_false_value;
      }
      else
        false_value = builder.create_splat(ir::undef_value::get(ty->get_scalar_ty()), ty->get_block_shapes());
      ir::value* first_load = builder.create_masked_load(first_ptr, first_mask, false_value, load->get_cache_modifier());
      // pre-fetch next iteration
      builder.set_insert_point(block->get_inst_list().back());
      ir::value* next_ptr = ptr->get_value_for_block(block);
      ir::value* next_mask = builder.create_splat(block_br->get_cond(), ty->get_block_shapes());
      if(auto* masked_load = dynamic_cast<ir::masked_load_inst*>(load)){
        ir::value* remat_mask = rematerialize(builder, block, masked_load->get_mask_operand(), 1);
        ir::value* remat_false_value = rematerialize(builder, block, masked_load->get_false_value_operand(), 1);
        next_mask = builder.create_and(next_mask, remat_mask);
        false_value = remat_false_value;
      }
      ir::value* next_load = builder.create_masked_load(next_ptr, next_mask, false_value, load->get_cache_modifier());
      // phi node
      builder.set_insert_point(block->get_first_non_phi());
      ir::phi_node* new_load = builder.create_phi(ty, 2);
      new_load->add_incoming(first_load, header);
      new_load->add_incoming(next_load, block);
      load->replace_all_uses_with(new_load);
      new_loads.push_back(new_load);
    }
  }

  // try to reorder prefetched value from a0, a1, a2, ..., b0, b1, b2, ...  to
  // a0, b0, a1, b1, ...
  if (!preheader_loads.empty()) {
    ir::basic_block* header = preheader_loads.begin()->first->get_incoming_block(0);
    builder.set_insert_point(header->get_inst_list().back());
    for (int i=1; i<num_stages-1; ++i) {
      for (auto iter = preheader_loads.begin(); iter != preheader_loads.end(); ++iter) {
        ir::instruction* original_load = static_cast<ir::instruction*>(iter->second.at(i));
        ir::instruction* moved_load = original_load->clone();
        builder.insert(moved_load);
        original_load->replace_all_uses_with(moved_load);
      }
    }
  }

  // try to move dot_inst after loads
  // for better overlap of io and compute
  struct move_config_t{
    std::vector<ir::instruction*> insts;
    ir::load_inst* dst;
  };
  std::vector<move_config_t> to_move(to_pipeline.size());

  if(has_copy_async_){
    for (size_t idx = 0; idx < to_pipeline.size(); ++idx) {
      auto info = to_pipeline[idx];
      ir::load_inst* load = info.load;
      ir::phi_node* ptr = info.ptr;
      ir::dot_inst* dot = info.dot;
      ir::basic_block* bb = dot->get_parent();
      recursive_deps(dot, bb, to_move[idx].insts);
      to_move[idx].dst = load;
    }

    for(auto& move_config: to_move){
      builder.set_insert_point_after(move_config.dst);
      for(ir::instruction* i: move_config.insts){
        i->get_parent()->erase(i);
        builder.insert(i);
      }
    }
  }


}

}
}
}