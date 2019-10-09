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
layout::layout(analysis::axes *axes, analysis::align *align, size_t num_warps)
  : axes_(axes), align_(align), num_warps_(num_warps) { }

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

const layout_t &layout::get(ir::value *v) const {
  return layouts_.at(groups_.at(v));
}

std::map<size_t, layout_t>& layout::get_all() {
  return layouts_;
}

void extract_io_use(ir::value *v, std::set<ir::value*>& result) {
  for(ir::user* u: v->get_users()){
    auto i = dynamic_cast<ir::io_inst*>(u);
    if(i && i->get_pointer_operand() == v)
      result.insert(v);
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

inline unsigned clamp(unsigned x, unsigned lo, unsigned hi) {
  return std::min(std::max(x, lo), hi);
}

void layout::init_hmma_tile(layout_t& layout) {
  auto ord = layout.order;
  auto shapes = layout.shapes;
  unsigned shape_0 = shapes[ord[0]];
  unsigned shape_1 = shapes[ord[1]];
  /* fragments per warp */
  // try to make things as square as possible to maximize data re-use
  std::vector<unsigned> fpw = {1, 1, 1};
  std::vector<unsigned> fpw_nm1;
  unsigned num_fragments = std::min<unsigned>((shape_0/8)*(shape_1/8), 4);
  do {
    fpw_nm1 = fpw;
    if(fpw[0]*fpw[1] < num_fragments)
      fpw[0] = clamp(fpw[0]*2, 1, shape_0 / 8);
    if(fpw[0]*fpw[1] < num_fragments)
      fpw[1] = clamp(fpw[1]*2, 1, shape_1 / 8);
  }while(fpw_nm1 != fpw);
  // store parameters
  for(unsigned d = 0; d < shapes.size(); d++)
    layout.fpw[d] = fpw[d];
  /* warps per tile */
  // try to make things as square as possible to maximize data re-use
  std::vector<unsigned> wpt = {1, 1, 1};
  std::vector<unsigned> wpt_nm1;
  do{
    wpt_nm1 = wpt;
    if(wpt[0] * wpt[1] * wpt[2] < num_warps_)
      wpt[0] = clamp(wpt[0]*2, 1, shape_0 / (fpw[0]*8));
    if(wpt[0] * wpt[1] * wpt[2] < num_warps_)
      wpt[1] = clamp(wpt[1]*2, 1, shape_1 / (fpw[1]*8));
  }while(wpt_nm1 != wpt);
  // store parameters
  for(unsigned d = 0; d < shapes.size(); d++)
    layout.wpt[d] = wpt[d];
  /* sanity check */
  unsigned effective_num_warps = 1;
  for(size_t d = 0; d < shapes.size(); d++)
    effective_num_warps *= layout.wpt[d];
  if(num_warps_ != effective_num_warps)
    throw std::runtime_error("cannot create a kernel with this amount of warps");
}

void layout::init_scanline_tile(layout_t& layout) {
  auto ord = layout.order;
  auto shapes = layout.shapes;
  unsigned size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<int>());
  unsigned ld = ord[0];
  unsigned num_threads = num_warps_*32;
  unsigned current = num_threads;
  layout.nts[ld] = clamp(size / num_threads, 1, 4);
  layout.mts[ld] = clamp(current, 1, shapes[ld] / layout.nts[ld]);
  current = current / layout.mts[ld];
  for(size_t d = 1; d < shapes.size(); d++){
    ld = ord[d];
    layout.nts[ld] = 1;
    layout.mts[ld] = clamp(current, 1, shapes[ld]);
    current = current / layout.mts[ld];
  }
  /* sanity check */
  unsigned effective_num_threads = 1;
  for(size_t d = 0; d < shapes.size(); d++)
    effective_num_threads *= layout.mts[d];
  if(num_threads != effective_num_threads)
    throw std::runtime_error("cannot create a kernel with this amount of warps");
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
    // type
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

  // find out axes for each layout
  for(const auto& x: values_) {
    auto cmp = [&rank](ir::value* x, ir::value *y) { return rank(x) < rank(y); };
    ir::value *largest = *std::max_element(x.second.begin(), x.second.end(), cmp);
    layouts_[x.first].axes = axes_->get(largest);
    layouts_[x.first].shapes = largest->get_type()->get_tile_shapes();
  }


  // find out the layout ordering of a group
  for(const auto& x: values_) {
    std::set<ir::value*> ptr;
    for(ir::value* v: x.second)
      extract_io_use(v, ptr);
    size_t rank = layouts_[x.first].axes.size();
    std::vector<int> order(rank);
    std::iota(order.begin(), order.end(), 0);
    for(ir::value *v: ptr){
      auto max_contiguous = align_->contiguous(v);
      std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
        return max_contiguous[a] > max_contiguous[b]; }
      );
    }
    layouts_[x.first].order = order;
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

  // tiling parameters
  for(auto& x: layouts_){
    /* HMMA parameters*/
    if(x.second.type == HMMA_884)
      init_hmma_tile(x.second);
    else
      init_scanline_tile(x.second);
  }
}

}
}
}
