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
  ir::builder& builder = mod.get_builder();

  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction* i: block->get_inst_list()){
    // coalesce before store
    if(auto x = dynamic_cast<ir::masked_store_inst*>(i))
    if(ir::value* op = x->get_value_operand())
    if(layout_->get(op)->to_mma()){
      builder.set_insert_point(x);
      ir::instruction* new_op = ir::recoalesce_inst::create(op);
      builder.insert(new_op);
      x->replace_uses_of_with(op, new_op);
    }
    // uncoalesce after load
    if(auto x = dynamic_cast<ir::masked_load_inst*>(i))
    if(layout_->get(x)->to_mma()){
        builder.set_insert_point(x);
        ir::instruction* new_x = ir::decoalesce_inst::create(x);
        builder.insert(new_x);
        x->replace_all_uses_with(new_x);
    }

  }
}


}
}
}
