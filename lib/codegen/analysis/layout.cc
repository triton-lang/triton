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
  for(size_t d = 0; d < rank; d++)
    result.insert(axes_->get_id(value, d));
  return result;
}

// constructor
layout::layout(analysis::axes *axes)
  : axes_(axes) { }

// get group id
unsigned layout::layout_of(ir::value *value) const
{ return groups_.at(value); }

// get values
const std::vector<ir::value*>& layout::values_of(unsigned id) const
{ return values_.at(id); }

// get number of groups
size_t layout::num_layouts() const
{ return values_.size(); }

// connect two values
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
  if(!common.empty())
    graph_.add_edge(x, y);
}

// make graph
void layout::make_graph(ir::instruction *i) {
  for(ir::value* opx: i->ops())
  for(ir::value* opy: i->ops()){
    connect(i, opx);
    connect(opx, opy);
  }
}

void layout::run(ir::module &mod) {
  // make graph
  graph_.clear();
  ir::for_each_instruction(mod, [this](ir::instruction* i) {
    make_graph(i);
  });
  // connected components
  values_.clear();
  groups_.clear();
  graph_.connected_components(&values_, &groups_);
}

}
}
}
