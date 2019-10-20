#include <algorithm>
#include "triton/ir/utils.h"
#include "triton/ir/instructions.h"
#include "triton/ir/module.h"
#include "triton/codegen/transform/coalesce.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/analysis/layout.h"

namespace triton {
namespace codegen{
namespace transform{

coalesce::coalesce(analysis::align* align, analysis::layout *layouts)
  : align_(align), layout_(layouts) { }

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

ir::value* coalesce::rematerialize(ir::value *x, ir::builder &builder,
                                   std::map<ir::value*, ir::value*>& seen) {
  if(seen.find(x) != seen.end())
    return seen.at(x);
  auto i = dynamic_cast<ir::instruction*>(x);
  // not an instruction -- forward value
  if(!i)
    return x;
  // already in shared memory -- forward value
  if(dynamic_cast<ir::copy_to_shared_inst*>(x)){
    return x;
  }
  // set insert point
  auto& inst_list = i->get_parent()->get_inst_list();
  auto pos = ++std::find(inst_list.begin(), inst_list.end(), i);
  builder.set_insert_point(pos);
  if(dynamic_cast<ir::load_inst*>(x)){
    ir::value *ret = builder.insert(ir::copy_to_shared_inst::create(x));
    return ret;
  }
  // default -- recursive clone
  ir::instruction *cloned = builder.insert(i->clone());
  seen[i] = cloned;
  // rematerialize operands
  for(ir::value *op: cloned->ops())
    cloned->replace_uses_of_with(op, rematerialize(op, builder, seen));
  return cloned;
}

void coalesce::run(ir::module &mod) {
  // find values to rematerialize
  size_t num_groups = layout_->num_layouts();
  std::vector<ir::io_inst*> remat;
  for(size_t id = 0; id < num_groups; id++) {
    const auto& values = layout_->values_of(id);
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
      remat.insert(remat.begin(), it->second.begin(), it->second.end());
  }
  // rematerialize values
  for(ir::io_inst *r: remat) {
    ir::builder& builder = mod.get_builder();
    // rematerialize operands
    std::map<ir::value*, ir::value*> seen;
    for(ir::value *op: r->ops())
      r->replace_uses_of_with(op, rematerialize(op, mod.get_builder(), seen));
    // copy to shared if load
    auto& inst_list = r->get_parent()->get_inst_list();
    auto pos = ++std::find(inst_list.begin(), inst_list.end(), r);
    builder.set_insert_point(pos);
    if(dynamic_cast<ir::load_inst*>(r)){
      ir::instruction *cts = builder.insert(ir::copy_to_shared_inst::create(r));
      r->replace_all_uses_with(cts);
      cts->replace_uses_of_with(cts, r);
    }
  }
}


}
}
}
