#include "triton/codegen/transform/prefetch.h"
#include "triton/codegen/target.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/utils.h"
#include "triton/ir/print.h"
#include <iostream>
#include <vector>
#include <algorithm>

namespace triton::codegen::transform {

/// find defs till phis
static void recursive_defs(ir::value *v, ir::basic_block *bb, std::vector<ir::instruction*> &ret) {
  ir::instruction *i = dynamic_cast<ir::instruction*>(v);
  if (!i || i->get_parent() != bb)
    return;
  if (i->get_id() == ir::INST_PHI)
    return;
  ret.push_back(i);
  for (ir::value *op : i->ops())
    recursive_defs(op, bb, ret);
}

void prefetch::run(ir::module &mod) {
  // 1. collect dots that can be prefethced
  std::vector<ir::dot_inst*> to_prefetch;
  ir::for_each_instruction(mod, [&](ir::instruction *i) {
    if (auto *dot = dynamic_cast<ir::dot_inst*>(i)) {
      // Now only do prefetching when dot is using tensor cores
      if (!(dot->get_operand(0)->get_type()->get_scalar_ty()->is_fp16_ty() ||
            dot->get_operand(0)->get_type()->get_scalar_ty()->is_bf16_ty() ||
            (dot->get_operand(0)->get_type()->get_scalar_ty()->is_fp32_ty() && dot->allow_tf32()
             && tgt_->as_nvidia() && tgt_->as_nvidia()->sm() >= 80) || 
            (dot->get_operand(0)->get_type()->get_scalar_ty()->is_integer_ty(8)
             && dot->get_operand(1)->get_type()->get_scalar_ty()->is_integer_ty(8)
             && tgt_->as_nvidia() && tgt_->as_nvidia()->sm() >= 80)
           )
         )
        return;
      auto *a = dynamic_cast<ir::phi_node*>(dot->get_operand(0));
      auto *b = dynamic_cast<ir::phi_node*>(dot->get_operand(1));
      if (a && a->get_incoming_block(1) == a->get_parent() &&
          b && b->get_incoming_block(1) == b->get_parent()) 
        to_prefetch.push_back(dot);
    }
  });

  assert(to_prefetch.size() <=1 && "Don't know what to do with multiple dots");
  ir::builder &builder = mod.get_builder();
  // 2. do the prefetching
  for (ir::dot_inst* dot : to_prefetch) {
    auto *a = dynamic_cast<ir::phi_node*>(dot->get_operand(0));
    auto *b = dynamic_cast<ir::phi_node*>(dot->get_operand(1));
    assert(a->get_incoming_block(0) == b->get_incoming_block(0));
    ir::basic_block *loop_header = a->get_incoming_block(0);
    ir::basic_block *loop_body = a->get_parent();

    // mark as prefetched
    dot->set_prefetched(true);

    // 1. in the loop header (first iteration)
    builder.set_insert_point(loop_header->get_inst_list().back());
    assert(a && b);
    builder.create_prefetch_s(a->get_incoming_value(0), /*inc*/ 0);
    builder.create_prefetch_s(b->get_incoming_value(0), /*inc*/ 0);

    // 2. at the end of the loop body (next iteration)
    builder.set_insert_point(loop_body->get_inst_list().back());
    builder.create_prefetch_s(a->get_incoming_value(1), /*inc*/ 1);
    builder.create_prefetch_s(b->get_incoming_value(1), /*inc*/ 1);

    prefetched_vals_.insert(a->get_incoming_value(0));
    prefetched_vals_.insert(b->get_incoming_value(0));
    // nested phis
    ir::value* next_a = a->get_incoming_value(1);
    while (auto* next_a_phi = dynamic_cast<ir::phi_node*>(next_a)) {
      prefetched_vals_.insert(next_a_phi->get_incoming_value(0));
      next_a = next_a_phi->get_incoming_value(1);
    }
    prefetched_vals_.insert(next_a);

    ir::value* next_b = b->get_incoming_value(1);
    while (auto* next_b_phi = dynamic_cast<ir::phi_node*>(next_b)) {
      prefetched_vals_.insert(next_b_phi->get_incoming_value(0));
      next_b = next_b_phi->get_incoming_value(1);
    }
    prefetched_vals_.insert(next_b);
  }

  // move loads to the beginning of the loop
  if (tgt_->as_nvidia() && tgt_->as_nvidia()->sm() < 80) {
    for (ir::function *fn : mod.get_function_list())
    for (ir::basic_block *bb : fn->blocks()) {
      // only apply to loop body
      if (bb->get_predecessors().size() != 2 || bb->get_predecessors()[1] != bb)
        continue;
      // record loads (& dependency) to move
      std::vector<ir::instruction*> loads;
      // record original inst order
      std::map<ir::instruction*, size_t> idx_map;
      size_t idx = 0;
      for (ir::instruction *inst : bb->get_inst_list()) {
        if (auto *i = dynamic_cast<ir::masked_load_inst*>(inst))
          recursive_defs(i, bb, loads);
        idx_map[inst] = idx;
        idx++;
      }

      // remove duplicates & keep the original input order
      std::sort(loads.begin(), loads.end());
      loads.erase(std::unique(loads.begin(), loads.end()), loads.end());
      std::sort(loads.begin(), loads.end(), [&idx_map](ir::instruction *a, ir::instruction *b) {
        return idx_map[a] < idx_map[b];
      });

      builder.set_insert_point(bb->get_first_non_phi());
      auto& inst_list = bb->get_inst_list();
      for (ir::instruction *i : loads){
        auto it = std::find(inst_list.begin(), inst_list.end(), i);
        // make sure we don't invalidate insert point
        // in case instruction already at the top
        if(it == builder.get_insert_point())
          continue;
        bb->erase(i);
        builder.insert(i);
      }
    }
  }
}
} // namespace triton::codegen::transform
