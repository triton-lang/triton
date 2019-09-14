#include <iostream>
#include <algorithm>
#include <numeric>
#include "triton/ir/function.h"
#include "triton/ir/cfg.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/module.h"
#include "triton/codegen/analysis/meminfo.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/transform/coalesce.h"

namespace triton {
namespace codegen{
namespace transform{

coalesce::coalesce(analysis::align* align, analysis::meminfo *mem)
  : align_(align), mem_(mem) { }

std::vector<unsigned> coalesce::get_order(ir::value* v) {
  return order_.at(v);
}

void coalesce::run(ir::module &mod) {

  std::set<ir::io_inst*> io;

  std::function<void(ir::value*)> set_order = [&](ir::value *v) -> void {
    if(order_.find(v) != order_.end())
      return;
    ir::type *tile_ty = v->get_type();
    if(auto *x = dynamic_cast<ir::store_inst*>(v))
      tile_ty = x->get_operand(0)->get_type();
    if(!tile_ty->is_tile_ty())
      return;
    std::vector<unsigned> order(tile_ty->get_tile_shapes().size());
    std::iota(order.begin(), order.end(), 0);
    order_[v] = order;
    if(ir::user* u = dynamic_cast<ir::user*>(v))
      for(ir::value* op: u->ops())
        set_order(op);
  };

  // initialize work-list
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: ir::cfg::reverse_post_order(fn))
  for(ir::instruction *i: block->get_inst_list()){
    if(auto *x = dynamic_cast<ir::io_inst*>(i)) {
      ir::type* ptr_ty = x->get_pointer_operand()->get_type();
      if(ptr_ty->is_tile_ty())
        io.insert(x);
    }
    set_order(i);
  }

  ir::builder &builder = mod.get_builder();
  std::map<ir::value*, ir::value*> replaced;
  for(ir::io_inst *i: io) {
    ir::value *ptr = i->get_pointer_operand();
    auto max_contiguous = align_->get_max_contiguous_vec(ptr);
    std::vector<unsigned> order(max_contiguous.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) { return max_contiguous[a] > max_contiguous[b]; } );
    std::list<std::pair<ir::instruction*, ir::instruction*>> work_list;
    if(order != order_[i])
      work_list.push_back({i, nullptr});
    // rematerialize recursively
    while(!work_list.empty()) {
      auto pair = work_list.back();
      ir::instruction* cloned = pair.first;
      ir::instruction* original = pair.second;
      order_[cloned] = order;
      work_list.pop_back();
      for(ir::value *op: cloned->ops()) {
        ir::instruction* i_op = dynamic_cast<ir::instruction*>(op);
        if(replaced.find(i_op) != replaced.end()){
          cloned->replace_uses_of_with(i_op, replaced.at(i_op));
          continue;
        }
        if(!i_op)
          continue;
        ir::type *ty = i_op->get_type();
        if(!ty->is_tile_ty())
          continue;
        auto& inst_list = i_op->get_parent()->get_inst_list();
        auto it = std::find(inst_list.begin(), inst_list.end(), i_op);
        it++;
        builder.set_insert_point(it);
        // found a load; write to shared memory and stop recursion
        ir::instruction *n_op = nullptr;
        if(mem_->is_shared(i_op)){
          i_op->add_use(cloned);
          continue;
        }
        if(auto* ld = dynamic_cast<ir::load_inst*>(i_op))
          n_op = ir::copy_to_shared_inst::create(ld);
        // not a load; rematerialize and add to worklist
        else {
          n_op = i_op->clone();
          work_list.push_back({n_op, i_op});
        }
        n_op = builder.insert(n_op);
        replaced.insert({i_op, n_op});
        order_[n_op] = order;
        align_->copy(n_op, i_op);
        mem_->copy(n_op, i_op);
        if(original)
          n_op->erase_use(original);
        cloned->replace_uses_of_with(i_op, n_op);
      }
    }

  }
}


}
}
}
