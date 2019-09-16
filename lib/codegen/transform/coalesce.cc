#include <iostream>
#include <algorithm>
#include <numeric>
#include "triton/ir/function.h"
#include "triton/ir/cfg.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/module.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/codegen/analysis/meminfo.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/transform/coalesce.h"

namespace triton {
namespace codegen{
namespace transform{

coalesce::coalesce(analysis::align* align, analysis::layout *layouts, analysis::meminfo *mem)
  : align_(align), layout_(layouts), mem_(mem) { }

// Find all values that are used as pointer operands in LD/ST
void coalesce::extract_io_use(ir::value *v, std::set<ir::io_inst*>& result) {
  for(ir::user* u: v->get_users()){
    auto i = dynamic_cast<ir::io_inst*>(u);
    if(i && i->get_pointer_operand() == v)
      result.insert(i);
  }
}

void coalesce::extract_ld(ir::io_inst* i, std::map<int, std::vector<ir::io_inst*>>& result) {
  ir::value *ptr = i->get_pointer_operand();
  auto contiguous = align_->contiguous(ptr);
  auto it = std::max_element(contiguous.begin(), contiguous.end());
  int axis = std::distance(contiguous.begin(), it);
  result[axis].push_back(i);
}

void coalesce::run(ir::module &mod) {
  // find values to rematerialize
  size_t num_groups = layout_->get_num_groups();
  std::vector<ir::io_inst*> remat;
  for(size_t id = 0; id < num_groups; id++) {
    const auto& values = layout_->values(id);
    // extract pointers used in ld/st operations
    std::set<ir::io_inst*> io;
    for(ir::value *v: values)
      extract_io_use(v, io);
    // extract leading axes
    std::map<int, std::vector<ir::io_inst*>> axes;
    for(ir::io_inst *i: io)
      extract_ld(i, axes);
    // update list of values to rematerialize
    if(axes.empty())
      continue;
    for(auto it = ++axes.rbegin(); it != axes.rend(); it++)
      remat.insert(remat.begin(),
                   it->second.begin(), it->second.end());
  }

  // rematerialize values
  ir::builder &builder = mod.get_builder();
  for(ir::io_inst *r: remat) {
    std::list<std::pair<ir::instruction*, ir::instruction*>> work_list;
    std::map<ir::value*, ir::value*> replaced;
    work_list.push_back({r, nullptr});
    // rematerialize recursively
    while(!work_list.empty()) {
      auto pair = work_list.back();
      ir::instruction* cloned = pair.first;
      ir::instruction* original = pair.second;
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
