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

coalesce::coalesce(analysis::align* align, analysis::layouts *layouts)
  : align_(align), layout_(layouts) { }


// simplify layout conversions using the following simple rules:
//   - cvt_1(cvt_2(x)) if convert1 is the inverse of convert2
//   - cvt_1(elementwise(x, y)) = elementwise(convert(x), convert(y))
//ir::value* coalesce::simplify(ir::instruction *inst, ir::builder& builder){
//  ir::value* _op = inst->get_operand(0);
//  ir::instruction* op = dynamic_cast<ir::instruction*>(_op);
//  analysis::mma_layout* mma_in  = layout_->get(op)  ->to_mma();
//  analysis::mma_layout* mma_out = layout_->get(inst)->to_mma();
//  std::cout << 1 << std::endl;
//  // i must be layout conversion instruction
//  if(!mma_in && !mma_out)
//    return inst;
//  //   - cvt_1(cvt_2(x)) if convert1 is the inverse of convert2
//  bool is_op_cvt = op->get_id() == ir::INST_CVT_LAYOUT;
//  if((mma_in || mma_out) && is_op_cvt &&
//     (layout_->get(inst) == layout_->get(op->get_operand(0))))
//    return op->get_operand(0);
//  //   - cvt_1(elementwise(x, y)) = elementwise(cvt_1(x), cvt_2(y))
//  if(op->get_id() != ir::INST_BINOP && op->get_id() != ir::INST_GETELEMENTPTR)
//    return inst;
//  std::cout << 1 << std::endl;
//  for(size_t i = 0; i < op->get_num_operands(); i++){
//    ir::value* arg_i = op->get_operand(i);
//    builder.set_insert_point(op);
//    // create new layout transform
//    ir::instruction* new_arg_i = inst->clone();
//    builder.insert(new_arg_i);
//    // set the right args
//    new_arg_i->replace_uses_of_with(new_arg_i->get_operand(0), arg_i);
//    op->replace_uses_of_with(arg_i, simplify(new_arg_i, builder));
//  }
//  std::cout << 2 << std::endl;
//  return op;
//}

void coalesce::run(ir::module &mod) {
  ir::builder& builder = mod.get_builder();
  // add layout conversion instructions
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction* i: block->get_inst_list()){
    // coalesce before store
    if(dynamic_cast<ir::store_inst*>(i) || dynamic_cast<ir::atomic_rmw_inst*>(i))
    if(ir::value* op = i->get_operand(1))
    if(op->get_type()->is_block_ty())
    if(layout_->get(op)->to_mma()){
      ir::instruction* new_op = ir::cvt_layout_inst::create(op);
      builder.set_insert_point(i);
      builder.insert(new_op);
      i->replace_uses_of_with(op, new_op);
    }
    // uncoalesce after load
    if(auto x = dynamic_cast<ir::load_inst*>(i))
    if(x->get_type()->is_block_ty())
    if(x->get_type()->get_tile_rank()==2)
    if(layout_->get(x)->to_mma()){
        builder.set_insert_point_after(x);
        ir::instruction* new_x = ir::cvt_layout_inst::create(x);
        builder.insert(new_x);
        x->replace_all_uses_with(new_x);
        new_x->replace_uses_of_with(new_x, x);
//        new_x->replace_uses_of_with(new_x, new_x);
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
        break;
      if(dynamic_cast<ir::cvt_layout_inst*>(val))
        break;
      std::vector<unsigned> in_contig;
      std::vector<ir::instruction*> queue = {val_inst};
      std::set<ir::instruction*> seen;
      std::vector<ir::io_inst*> ios;
      while(!queue.empty()){
        ir::instruction* curr = queue.back();
        seen.insert(curr);
        queue.pop_back();
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
