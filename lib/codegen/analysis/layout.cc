#include <algorithm>
#include <iostream>
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/utils.h"

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
    if(axes_->has_id(value, d))
      result.insert(axes_->get_id(value, d));
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

void layout::connect(ir::value *x, ir::value *y) {
  if(x == y)
    return;
  if(!x->get_type()->is_tile_ty())
    return;
  if(!y->get_type()->is_tile_ty())
    return;
  std::set<int> x_axes = axes_of(x);
  std::set<int> y_axes = axes_of(y);
  std::set<int> common;
  std::set_intersection(x_axes.begin(), x_axes.end(),
                        y_axes.begin(), y_axes.end(),
                        std::inserter(common, common.begin()));
  if(!common.empty()){
    nodes_.insert(x);
    nodes_.insert(y);
    dependencies_[x].insert(y);
    dependencies_[y].insert(x);
  }
}

void layout::make_graph(ir::instruction *i) {
  for(ir::value* opx: i->ops())
  for(ir::value* opy: i->ops()){
    connect(i, opx);
    connect(opx, opy);
  }
}

// run
void layout::run(ir::module &mod) {
  nodes_.clear();
  dependencies_.clear();
  groups_.clear();
  values_.clear();
  // make graph
  ir::for_each_instruction(mod, [this](ir::instruction* i) { make_graph(i); });
  // connected components
  unsigned group_id = 0;
  while(!nodes_.empty()){
    connected_components(*nodes_.begin(), nodes_, dependencies_, group_id++);
  }
}

}
}
}
