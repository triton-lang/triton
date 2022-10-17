#include <algorithm>
#include <iostream>
#include "triton/ir/utils.h"
#include "triton/ir/instructions.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/codegen/transform/coalesce.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/analysis/layout.h"

namespace triton {
namespace codegen{
namespace transform{

coalesce::coalesce(analysis::align* align, analysis::layouts *layouts, bool has_sm80)
  : align_(align), layout_(layouts), has_sm80_(has_sm80) { }

void coalesce::run(ir::module &mod) {
  std::set<analysis::data_layout*> invalidated;
  ir::builder& builder = mod.get_builder();
  // add layout conversion instructions
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction* i: block->get_inst_list()){
    // coalesce before store
    if(dynamic_cast<ir::store_inst*>(i) || dynamic_cast<ir::atomic_rmw_inst*>(i))
    if(ir::value* op = i->get_operand(1))
    if(op->get_type()->is_block_ty())
    if(op->get_type()->get_tile_ranks1() == 2)
    if(invalidated.find(layout_->get(op)) == invalidated.end())
    if(layout_->get(op)->to_mma())
    if(dynamic_cast<ir::io_inst*>(i)->get_eviction_policy()==ir::io_inst::NORMAL){
      ir::instruction* new_op = ir::cvt_layout_inst::create(op);
      builder.set_insert_point(i);
      builder.insert(new_op);
      i->replace_uses_of_with(op, new_op);
    }
    // coalesce before copy_to_shared
    // only necessary for sm < 80 as Ampere+ can handle reduction
    // on MMA layout
    if(!has_sm80_)
    if(dynamic_cast<ir::copy_to_shared_inst*>(i) || dynamic_cast<ir::reduce_inst*>(i))
    if(ir::value* op = i->get_operand(0))
    if(op->get_type()->is_block_ty())
    if(op->get_type()->get_tile_ranks1() == 2)
    if(invalidated.find(layout_->get(op)) == invalidated.end())
    if(layout_->get(op)->to_mma()){
      ir::instruction* new_op = ir::cvt_layout_inst::create(op);
      builder.set_insert_point(i);
      builder.insert(new_op);
      op->replace_all_uses_with(new_op);
      new_op->replace_uses_of_with(new_op, op);
      invalidated.insert(layout_->get(op));
    }
    // uncoalesce after load
    if(auto x = dynamic_cast<ir::load_inst*>(i))
    if(x->get_type()->is_block_ty())
    if(x->get_type()->get_tile_ranks1()==2)
    if(layout_->get(x)->to_mma())
    if(!has_sm80_ || dynamic_cast<ir::io_inst*>(i)->get_eviction_policy()==ir::io_inst::NORMAL){
        builder.set_insert_point_after(x);
        ir::instruction* new_x = ir::cvt_layout_inst::create(x);
        builder.insert(new_x);
        x->replace_all_uses_with(new_x);
        new_x->replace_uses_of_with(new_x, x);
    }
  }
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction* i: block->get_inst_list()){
    // re-arrange scanline to promote memory coalescing
    if(auto x = dynamic_cast<ir::store_inst*>(i)){
      ir::value* ptr = x->get_pointer_operand();
      ir::value* val = x->get_value_operand();
      auto out_contig = align_->contiguous(ptr);
      auto val_inst = dynamic_cast<ir::instruction*>(val);
      if(!val_inst)
        continue;
      if(dynamic_cast<ir::cvt_layout_inst*>(val))
        continue;
      if(!val->get_type()->is_block_ty() || val->get_type()->get_tile_ranks1()==1)
        continue;
      std::vector<unsigned> in_contig;
      std::vector<ir::instruction*> queue = {val_inst};
      std::set<ir::instruction*> seen;
      std::vector<ir::io_inst*> ios;
      while(!queue.empty()){
        ir::instruction* curr = queue.back();
        seen.insert(curr);
        queue.pop_back();
        if(auto dot_inst = dynamic_cast<ir::dot_inst*>(curr))
          break;
        if(auto io_inst = dynamic_cast<ir::io_inst*>(curr)){
          in_contig = align_->contiguous(io_inst->get_pointer_operand());
          break;
        }
        for(ir::value* op: curr->ops()){
          auto inst_op = dynamic_cast<ir::instruction*>(op);
          if(!inst_op || seen.find(inst_op) != seen.end())
            continue;
          if(!op->get_type()->is_block_ty() ||
             !val->get_type()->is_block_ty())
            continue;
          if(op->get_type()->get_tile_num_elements() ==
             val->get_type()->get_tile_num_elements())
            queue.push_back(inst_op);
        }
      }
      if(in_contig.size() <= 1 || out_contig==in_contig)
        continue;
      builder.set_insert_point_after(val_inst);
      auto new_val = builder.insert(ir::cvt_layout_inst::create(val_inst));
      x->replace_uses_of_with(val_inst, new_val);
    }
  }
}


}
}
}
