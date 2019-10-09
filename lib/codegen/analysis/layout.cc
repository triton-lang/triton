#include <algorithm>
#include <iostream>
#include <numeric>
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/utils.h"

namespace triton{
namespace codegen{
namespace analysis{


// constructor
layout::layout(analysis::axes *axes, analysis::align *align)
  : axes_(axes), align_(align) { }

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

// hmma
bool is_hmma_c(ir::value *v){
  bool result = false;
  if(auto *x = dynamic_cast<ir::dot_inst*>(v)){
    ir::value *a = x->get_operand(0);
    ir::type *a_ty = a->get_type();
    ir::value *b = x->get_operand(1);
    ir::type *b_ty = b->get_type();
    result = a_ty->get_scalar_ty()->is_half_ty() &&
             b_ty->get_scalar_ty()->is_half_ty();
  }
  return result;
}

layout_t layout::get(ir::value *v) const {
  return layouts_.at(groups_.at(v));
}

const std::map<size_t, layout_t>& layout::get_all() const {
  return layouts_;
}

void extract_io_use(ir::value *v, std::set<ir::io_inst*>& result) {
  for(ir::user* u: v->get_users()){
    auto i = dynamic_cast<ir::io_inst*>(u);
    if(i && i->get_pointer_operand() == v)
      result.insert(i);
  }
}


inline bool is_trans(ir::value *v) {
  if(dynamic_cast<ir::trans_inst *>(v)) {
    return true;
  }
  if(auto *phi = dynamic_cast<ir::instruction *>(v)) {
    bool result = true;
    for(ir::value *op: phi->ops())
      result = result && is_trans(op);
    return result;
  }
  return false;
}


void layout::run(ir::module &mod) {
  // make graph
  graph_.clear();
  ir::for_each_instruction(mod, [this](ir::instruction* i) {
    make_graph(i);
  });
  // connected components
  graph_.connected_components(&values_, &groups_);
  // create layouts
  for(const auto& x: values_) {
    bool hmma_c = std::any_of(x.second.begin(), x.second.end(), &is_hmma_c);
    layouts_[x.first].type = hmma_c ? HMMA_884 : SCANLINE;

  }


  /* ---- TO CLEAN ---- */

  size_t num_groups = num_layouts();
  // helpers
  auto rank = [this](ir::value* v) {
    int ret = 0;
    for(int s: v->get_type()->get_tile_shapes())
      ret += s > 1;
    return ret;
  };

  // find out which value is the largest in each group
  for(const auto& x: values_) {
    auto cmp = [&rank](ir::value* x, ir::value *y) { return rank(x) < rank(y); };
    ir::value *largest = *std::max_element(x.second.begin(), x.second.end(), cmp);
    layouts_[x.first].axes = axes_->get(largest);
    layouts_[x.first].i = largest;
    layouts_[x.first].shapes = largest->get_type()->get_tile_shapes();
  }


  // find out the layout ordering of a group
  for(size_t i = 0; i < num_groups; i++){
    std::set<ir::io_inst*> io;
    for(ir::value* v: values_of(i))
      extract_io_use(v, io);
    auto cmp = [&rank](ir::io_inst* x, ir::io_inst *y) {
      return rank(x->get_pointer_operand()) < rank(y->get_pointer_operand());
    };
    auto it = std::max_element(io.begin(), io.end(), cmp);
    std::vector<int> order(layouts_[i].axes.size());
    std::iota(order.begin(), order.end(), 0);
    if(it != io.end()) {
      auto max_contiguous = align_->contiguous((*it)->get_pointer_operand());
      std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
        return max_contiguous[a] > max_contiguous[b]; }
      );
    }
    layouts_[i].order = order;
  }
  // matrix multiplication optimizations
  for(size_t i = 0; i < num_groups; i++){
    std::vector<ir::dot_inst*> dots;
    for(ir::value* v: values_of(i))
      if(auto *x = dynamic_cast<ir::dot_inst*>(v))
        dots.push_back(x);
    for(ir::dot_inst* dot: dots){
      ir::value* a = dot->get_operand(0);
      ir::value* b = dot->get_operand(1);
      if(get(dot).type == HMMA_884){
        auto a_val = values_of(layout_of(a));
        auto b_val = values_of(layout_of(b));
        for(ir::value *v: a_val)
          if(auto *cts = dynamic_cast<ir::copy_to_shared_inst*>(v))
            layouts_[layout_of(a)].order = layouts_[layout_of(cts->get_operand(0))].order;
        for(ir::value *v: b_val)
          if(auto *cts = dynamic_cast<ir::copy_to_shared_inst*>(v))
            layouts_[layout_of(b)].order = layouts_[layout_of(cts->get_operand(0))].order;
      }
      else{
        std::vector<int> col = {0, 1};
        std::vector<int> row = {1, 0};
        layouts_[layout_of(a)].order = is_trans(a) ? row : col;
        layouts_[layout_of(b)].order = is_trans(b) ? col : row;
      }
    }
  }

}

}
}
}
