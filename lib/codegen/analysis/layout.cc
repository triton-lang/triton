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
  std::vector<int> x_axes = axes_->get(x);
  std::vector<int> y_axes = axes_->get(y);
  std::set<int> sx_axes(x_axes.begin(), x_axes.end());
  std::set<int> sy_axes(y_axes.begin(), y_axes.end());
  std::set<int> common;
  std::set_intersection(sx_axes.begin(), sx_axes.end(),
                        sy_axes.begin(), sy_axes.end(),
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
  graph_.connected_components(&values_, &groups_);
}

}
}
}
