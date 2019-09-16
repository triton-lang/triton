#include <algorithm>
#include <iostream>
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"

namespace triton{
namespace codegen{
namespace analysis{


// axes
std::set<int> layout::axes_of(ir::value *value) {
  auto ty = value->get_type();
  // rank of value
  size_t rank = 0;
  if(ty->is_tile_ty())
    rank = ty->get_tile_rank();
  // create result
  std::set<int> result;
  for(size_t d = 0; d < rank; d++){
    if(axes_->has(value, d))
      result.insert(axes_->get(value, d));
  }
  return result;
}

// connected components
void layout::connected_components(node_t x, std::set<node_t> &nodes, graph_t &graph, unsigned group_id) {
  groups_[x] = group_id;
  values_[group_id].push_back(x);
  if(nodes.find(x) != nodes.end()){
    nodes.erase(x);
    for(const node_t &y: graph[x])
      connected_components(y, nodes, graph, group_id);
  }
}

// constructor
layout::layout(analysis::axes *axes)
  : axes_(axes) { }

// get group id
unsigned layout::id(ir::value *value) const
{ return groups_.at(value); }

// get values
const std::vector<ir::value*>& layout::values(unsigned id) const
{ return values_.at(id); }

// get number of groups
size_t layout::get_num_groups() const
{ return values_.size(); }

// run
void layout::run(ir::module &mod) {
  nodes_.clear();
  dependencies_.clear();
  groups_.clear();
  values_.clear();
  // Create graph
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i : block->get_inst_list()) {
    // skip scalars
    if(!i->get_type()->is_tile_ty())
      continue;
    // add an edge between i and the operands that share an axis
    std::set<int> i_axes = axes_of(i);
    nodes_.insert(i);
    for(ir::value* op: i->ops()){
      if(!op->get_type()->is_tile_ty())
        continue;
      nodes_.insert(op);
      std::set<int> op_axes = axes_of(op);
      std::set<int> common;
      std::set_intersection(i_axes.begin(), i_axes.end(),
                            op_axes.begin(), op_axes.end(),
                            std::inserter(common, common.begin()));
      if(!common.empty() || !op->get_type()->is_tile_ty()){
        dependencies_[i].insert(op);
        dependencies_[op].insert(i);
      }
    }
  }
  // Grids
  unsigned group_id = 0;
  while(!nodes_.empty()){
    connected_components(*nodes_.begin(), nodes_, dependencies_, group_id++);
  }
}

}
}
}
