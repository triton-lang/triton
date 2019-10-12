#include <algorithm>
#include <iostream>
#include <numeric>
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/codegen/instructions.h"
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
  graph_.add_edge(x, x);
  graph_.add_edge(y, y);
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

const layout_t* layout::get(ir::value *v) const {
  return layouts_.at(groups_.at(v));
}

std::map<size_t, layout_t*>& layout::get_all() {
  return layouts_;
}

void extract_io_use(ir::value *v, std::set<ir::value*>& result) {
  for(ir::user* u: v->get_users()){
    auto i = dynamic_cast<ir::io_inst*>(u);
    if(i && i->get_pointer_operand() == v)
      result.insert(v);
  }
}

void extract_dot_use(ir::value *v, ir::value*& result, size_t n) {
  for(ir::user* u: v->get_users()){
    auto i = dynamic_cast<ir::dot_inst*>(u);
    if(i && i->get_operand(n) == v)
      result = v;
  }
}

void extract_hmma_dot_use(ir::value *v, ir::value*& result, size_t n) {
  for(ir::user* u: v->get_users()){
    auto i = dynamic_cast<ir::dot_inst*>(u);
    if(i && is_hmma_c(i) && i->get_operand(n) == v)
      result = v;
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



layout_t::layout_t(layout_type_t _type,
                   const std::vector<int> &_axes,
                   const std::vector<unsigned> &_shapes,
                   const std::vector<ir::value *> &_values,
                   size_t _id,
                   analysis::align* align): type(_type), axes(_axes), shapes(_shapes), values(_values), id(_id) {
  // io pointer
  std::set<ir::value*> ptr;
  for(ir::value* v: values)
    extract_io_use(v, ptr);
  order.resize(axes.size());
  std::iota(order.begin(), order.end(), 0);
  for(ir::value *v: ptr){
    auto max_contiguous = align->contiguous(v);
    std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
      return max_contiguous[a] > max_contiguous[b];
    });
  }
}

inline unsigned clamp(unsigned x, unsigned lo, unsigned hi) {
  return std::min(std::max(x, lo), hi);
}

layout_hmma_884_t::layout_hmma_884_t(size_t num_warps,
                                     const std::vector<int>& _axes,
                                     const std::vector<unsigned>& _shapes,
                                     const std::vector<ir::value *> &values, size_t _id,
                                     analysis::align* align): layout_t(HMMA_884, _axes, _shapes, values, _id, align) {

  unsigned shape_0 = shapes[order[0]];
  unsigned shape_1 = shapes[order[1]];
  /* fragments per warp */
  // try to make things as square as possible to maximize data re-use
  fpw = {1, 1, 1};
  std::vector<int> fpw_nm1;
  unsigned num_fragments = std::min<unsigned>((shape_0/8)*(shape_1/8), 4);
  do {
    fpw_nm1 = fpw;
    if(fpw[0]*fpw[1] < num_fragments)
      fpw[0] = clamp(fpw[0]*2, 1, shape_0 / 8);
    if(fpw[0]*fpw[1] < num_fragments)
      fpw[1] = clamp(fpw[1]*2, 1, shape_1 / 8);
  }while(fpw_nm1 != fpw);
  /* warps per tile */
  // try to make things as square as possible to maximize data re-use
  wpt = {1, 1, 1};
  std::vector<int> wpt_nm1;
  do{
    wpt_nm1 = wpt;
    if(wpt[0] * wpt[1] * wpt[2] < num_warps)
      wpt[0] = clamp(wpt[0]*2, 1, shape_0 / (fpw[0]*8));
    if(wpt[0] * wpt[1] * wpt[2] < num_warps)
      wpt[1] = clamp(wpt[1]*2, 1, shape_1 / (fpw[1]*8));
  }while(wpt_nm1 != wpt);
  /* sanity check */
  unsigned effective_num_warps = 1;
  for(size_t d = 0; d < shapes.size(); d++)
    effective_num_warps *= wpt[d];
  if(num_warps != effective_num_warps)
    throw std::runtime_error("cannot create a kernel with this amount of warps");
}

inline bool is_loop_latch(ir::phi_node *phi, ir::instruction *terminator){
  if(phi->get_parent() != terminator->get_parent())
    return false;
  if(auto *br = dynamic_cast<ir::cond_branch_inst*>(terminator))
    return br->get_true_dest() == phi->get_parent()
           || br->get_false_dest() == phi->get_parent();
  else if(dynamic_cast<ir::uncond_branch_inst*>(terminator))
    return false;
  else
    throw std::runtime_error("unreachable");
}



layout_scanline_t::layout_scanline_t(size_t num_warps,
                                     const std::vector<int>& _axes,
                                     const std::vector<unsigned>& _shapes,
                                     const std::vector<ir::value *> &values,
                                     size_t _id,
                                     analysis::align* align): layout_t(SCANLINE, _axes, _shapes, values, _id, align){
  unsigned size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<int>());
  unsigned num_threads = num_warps * 32;
  nts.resize(shapes.size());
  mts.resize(shapes.size());
  unsigned i = order[0];
  nts[i] = clamp(size / num_threads, 1, 4);
  mts[i] = clamp(num_threads, 1, shapes[i] / nts[i]);
  num_threads = num_threads / mts[i];
  for(size_t d = 1; d < shapes.size(); d++){
    i = order[d];
    nts[i] = 1;
    mts[i] = clamp(num_threads, 1, shapes[i]);
    num_threads = num_threads / mts[i];
  }
  /* sanity check */
  unsigned effective_num_threads = 1;
  for(size_t d = 0; d < shapes.size(); d++)
    effective_num_threads *= mts[d];
  if(num_warps * 32 != effective_num_threads)
    throw std::runtime_error("cannot create a kernel with this amount of warps");
}

void extract_double_bufferable(ir::value *v, std::shared_ptr<double_buffer_info_t>& res) {
  auto* phi = dynamic_cast<ir::phi_node*>(v);
  if(!phi || phi->get_num_incoming() != 2)
    return;
  ir::basic_block *block_0 = phi->get_incoming_block(0);
  ir::basic_block *block_1 = phi->get_incoming_block(1);
  ir::instruction *terminator_0 = block_0->get_inst_list().back();
  ir::instruction *terminator_1 = block_1->get_inst_list().back();
  bool is_latch_0 = is_loop_latch(phi, terminator_0);
  bool is_latch_1 = is_loop_latch(phi, terminator_1);
  ir::value *value_0 = phi->get_incoming_value(0);
  ir::value *value_1 = phi->get_incoming_value(1);
  ir::instruction *i_0 = dynamic_cast<ir::instruction*>(value_0);
  ir::instruction *i_1 = dynamic_cast<ir::instruction*>(value_1);
  if(!i_0 || !i_1 ||
     storage_info.at(i_0->get_id()).first != codegen::SHARED ||
     storage_info.at(i_1->get_id()).first != codegen::SHARED)
    return;
  if(is_latch_1)
    res.reset(new double_buffer_info_t{value_0, value_1, phi});
  if(is_latch_0)
    res.reset(new double_buffer_info_t{value_1, value_0, phi});
}


layout_shared_t::layout_shared_t(const layout_t *arg,
                                 const std::vector<int>& _axes,
                                 const std::vector<unsigned>& _shapes,
                                 const std::vector<ir::value *> &values,
                                 ir::type *ty,
                                 size_t _id,
                                 analysis::align* align): layout_t(SHARED, _axes, _shapes, values, _id, align) {

  this->ty = ty;
  size = 0;

  // double-buffering
  for(ir::value *v: values)
    extract_double_bufferable(v, double_buffer);

  // order
  if(arg->type == SCANLINE)
    order = arg->order;
  ir::value* dot_a = nullptr;
  ir::value* dot_b = nullptr;
  ir::value* hmma_dot_a = nullptr;
  ir::value* hmma_dot_b = nullptr;
  for(ir::value* v: values){
    extract_dot_use(v, dot_a, 0);
    extract_dot_use(v, dot_b, 1);
    extract_hmma_dot_use(v, hmma_dot_a, 0);
    extract_hmma_dot_use(v, hmma_dot_b, 1);
  }
  std::vector<int> col = {0, 1};
  std::vector<int> row = {1, 0};
  bool is_nonhmma_dot_a = dot_a && !hmma_dot_a;
  bool is_nonhmma_dot_b = dot_b && !hmma_dot_b;
  if(is_nonhmma_dot_a)
    order = is_trans(dot_a) ? row : col;
  if(is_nonhmma_dot_b)
    order = is_trans(dot_b) ? col : row;

  // padding
  pad = 0;
  if(hmma_dot_a){
    bool row = is_trans(hmma_dot_a) ^ order[0] == 1;
    pad = 24 - shapes[row ? 0: 1] % 32;
  }
  else if(hmma_dot_b){
    bool row = is_trans(hmma_dot_b) ^ order[0] == 1;
    pad = 24 - shapes[row ? 1 : 0] % 32;
  }
  else if(order != arg->order) {
    pad = 16;
  }

  // size
  auto shape = this->shapes;
  shape[order[0]] += pad;
  size = ty->get_primitive_size_in_bits() / 8;
  for(auto s: shape)
     size *= s;
  if(double_buffer)
    size *= 2;
}

void layout::create(size_t id, const std::vector<ir::value*>& values) {
  auto it_hmma_c = std::find_if(values.begin(), values.end(), &is_hmma_c);
  auto cmp = [](ir::value* x, ir::value *y) {
    return x->get_type()->get_tile_ranks1() <
           y->get_type()->get_tile_ranks1();
  };
  ir::value *largest = *std::max_element(values.begin(), values.end(), cmp);
  const auto& axes = axes_->get(largest);
  const auto& shapes = largest->get_type()->get_tile_shapes();
  auto it_cts = std::find_if(values.begin(), values.end(), [](ir::value* v) {
      return dynamic_cast<ir::copy_to_shared_inst*>(v);
  });
  // type
  if(it_hmma_c != values.end())
    layouts_[id] = new layout_hmma_884_t(num_warps_, axes, shapes, values, id, align_);
  else if(it_cts != values.end()){
    ir::copy_to_shared_inst *cts = (ir::copy_to_shared_inst*)*it_cts;
    ir::value *arg = cts->get_operand(0);
    create(groups_.at(arg), values_.at(groups_.at(arg)));
    layouts_[id] = new layout_shared_t(get(arg), axes, shapes, values, largest->get_type()->get_scalar_ty(), id, align_);
  }
  else
    layouts_[id] = new layout_scanline_t(num_warps_, axes, shapes, values, id, align_);
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
  for(const auto& x: values_)
    create(x.first, x.second);
}

}
}
}
