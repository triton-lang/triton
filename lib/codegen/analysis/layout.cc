#include <algorithm>
#include <numeric>
#include <iostream>
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/utils.h"
// #include "triton/ir/type.h"

namespace triton{
namespace codegen{
namespace analysis{

/* -------------------------------- *
 *          Helper Functions        *
 * -------------------------------- */

inline unsigned clamp(unsigned x, unsigned a, unsigned b) {
  unsigned lo = std::min(a, b);
  unsigned hi = std::max(a, b);
  return std::min(std::max(x, lo), hi);
}

inline bool is_hmma_c(ir::value *v){
  bool result = false;
  if(auto *x = dynamic_cast<ir::dot_inst*>(v)){
    ir::value *a = x->get_operand(0);
    ir::type *a_ty = a->get_type();
    ir::value *b = x->get_operand(1);
    ir::type *b_ty = b->get_type();
    result = a_ty->get_scalar_ty()->is_fp16_ty() &&
             b_ty->get_scalar_ty()->is_fp16_ty();
  }
  return result;
}

inline void extract_io_use(ir::value *v, std::set<ir::value*>& result) {
  for(ir::user* u: v->get_users()){
    auto i = dynamic_cast<ir::io_inst*>(u);
    if(i && i->get_pointer_operand() == v)
      result.insert(v);
  }
}

inline void extract_dot_use(ir::value *v, ir::value*& result, size_t n) {
  for(ir::user* u: v->get_users()){
    auto i = dynamic_cast<ir::dot_inst*>(u);
    if(i && i->get_operand(n) == v)
      result = v;
  }
}

inline void extract_hmma_dot_use(ir::value *v, ir::value*& result, size_t n) {
  for(ir::user* u: v->get_users()){
    auto i = dynamic_cast<ir::dot_inst*>(u);
    if(i && is_hmma_c(i) && i->get_operand(n) == v)
      result = i;
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


/* -------------------------------- *
 *          Layout Visitor          *
 * -------------------------------- */

void layout_visitor::visit_layout(data_layout *layout) {
  layout->accept(this);
}


/* -------------------------------- *
 *        Base Data Layout          *
 * -------------------------------- */

data_layout::data_layout(id_t id,
                         const std::vector<int> &axes,
                         const std::vector<unsigned> &shape,
                         const std::vector<ir::value *> &values,
                         analysis::align* align): id_(id), axes_(axes), shape_(shape), values_(values) {
  // io pointer
  std::set<ir::value*> ptr;
  for(ir::value* v: values_)
    extract_io_use(v, ptr);
  order_.resize(axes_.size());
  std::iota(order_.begin(), order_.end(), 0);
  std::vector<unsigned> max_contiguous;
  for(ir::value* p: ptr){
    std::vector<unsigned> curr = align->contiguous(p);
    if(curr.size() > max_contiguous.size())
      max_contiguous = curr;
    else if(curr.size() == max_contiguous.size()){
      if(*std::max_element(curr.begin(), curr.end()) > *std::max_element(max_contiguous.begin(), max_contiguous.end()))
        max_contiguous = curr;
    }
  }
  if(max_contiguous.size() > 0){
    std::sort(order_.begin(), order_.end(), [&](unsigned a, unsigned b) {
      return max_contiguous[a] > max_contiguous[b];
    });
//    std::cout << max_contiguous[0] << " " << max_contiguous[1] << std::endl;
//    std::cout << order_[0] << " " << order_[1] << std::endl;
  }
}

int data_layout::find_axis(int to_find) const {
  auto it = std::find(axes_.begin(), axes_.end(), to_find);
  if(it == axes_.end())
    return -1;
  return std::distance(axes_.begin(), it);
}


distributed_layout::distributed_layout(id_t id,
                         const std::vector<int> &axes,
                         const std::vector<unsigned> &shape,
                         const std::vector<ir::value *> &values,
                         analysis::align* align): data_layout(id, axes, shape, values, align)
{ }

/* -------------------------------- *
 *           MMA Layout             *
 * -------------------------------- */

mma_layout::mma_layout(size_t num_warps,
                       const std::vector<int>& axes,
                       const std::vector<unsigned>& shape,
                       const std::vector<ir::value *> &values,
                       analysis::align* align, target* tgt,
                       shared_layout *layout_a, shared_layout *layout_b): distributed_layout(MMA, axes, shape, values, align) {
  /* fragments per warp */
  // try to make things as square as possible to maximize data re-use
  if(tgt->as_nvidia()->sm() < 80){
    fpw_ = {2, 2, 1};
    auto ord_a = layout_a->get_order();
    auto ord_b = layout_b->get_order();
    bool is_a_row = ord_a[0] != 0;
    bool is_b_row = ord_b[0] != 0;
    bool is_a_vec4 = !is_a_row && (layout_a->get_shape()[ord_a[0]] <= 16);
    bool is_b_vec4 =  is_b_row && (layout_b->get_shape()[ord_b[0]] <= 16);
    int pack_size_0 = (is_a_row ||  is_a_vec4) ? 1 : 2;
    int pack_size_1 = (is_b_row && !is_b_vec4) ? 2 : 1;
    rep_ = {2*pack_size_0, 2*pack_size_1, 1};
    spw_ = {fpw_[0]*4*rep_[0], fpw_[1]*4*rep_[1], 1};
  }
  else{
    fpw_ = {1, 1, 1};
    spw_ = {16, 8, 1};
    rep_ = {2,  2, 1};
  }
  order_ = {0, 1};

  /* warps per tile */
  // try to make things as square as possible to maximize data re-use
  wpt_ = {1, 1, 1};
  std::vector<int> wpt_nm1;
  do{
    wpt_nm1 = wpt_;
    if(wpt_[0] * wpt_[1] * wpt_[2] < num_warps)
      wpt_[0] = clamp(wpt_[0]*2, 1, shape_[0] / spw_[0]);
    if(wpt_[0] * wpt_[1] * wpt_[2] < num_warps)
      wpt_[1] = clamp(wpt_[1]*2, 1, shape_[1] / spw_[1]);
  }while(wpt_nm1 != wpt_);

  /* shape per block */
  shape_per_cta_ = {spw_[0]*wpt_[0], spw_[1]*wpt_[1], 1};
}


/* -------------------------------- *
 *         Scanline Layout          *
 * -------------------------------- */

scanline_layout::scanline_layout(size_t num_warps,
                                 const std::vector<int>& axes,
                                 const std::vector<unsigned>& shape,
                                 const std::vector<ir::value *> &values,
                                 analysis::align* align, target *tgt): distributed_layout(SCANLINE, axes, shape, values, align){
  unsigned size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
  unsigned num_threads = tgt->is_gpu() ? num_warps * 32 : 1;
  nts_.resize(shape_.size());
  mts_.resize(shape_.size());
  bool is_dot = std::any_of(values.begin(), values.end(),
                            [&](ir::value* v) { return dynamic_cast<ir::dot_inst*>(v); });

  ir::value *ptr = nullptr;
  for(ir::value *v: values)
    for(ir::user *usr: v->get_users())
      if(auto *io = dynamic_cast<ir::io_inst*>(usr)){
        if(!ptr || ptr->get_type()->get_tile_rank() < io->get_pointer_operand()->get_type()->get_tile_rank())
        ptr = io->get_pointer_operand();
      }

  unsigned i = order_[0];
  int contiguous = 1;
  if(ptr){
    int nbits = ptr->get_type()->get_pointer_element_ty()->get_scalar_ty()->get_primitive_size_in_bits();
    contiguous = std::min<int>(align->get(ptr, i), 128 / nbits);
  }

  nts_[i] = clamp(size / num_threads, 1, std::min<int>(contiguous, shape_[i]));
  mts_[i] = clamp(num_threads, 1, shape_[i] / nts_[i]);
  size /= shape_[i];
  num_threads /= mts_[i];
  if(is_dot)
    nts_[order_[1]] = clamp(size / num_threads, 1, std::min<int>(4, shape_[order_[1]]));
  for(size_t d = 1; d < shape_.size(); d++){
    i = order_[d];
    if(d > 1 || !is_dot)
      nts_[i] = 1;
    mts_[i] = clamp(num_threads, 1, shape_[i] / nts_[i]);
    num_threads = num_threads / mts_[i];
  }

  shape_per_cta_.resize(shape_.size());
  for(size_t d = 0; d < shape_.size(); d++)
    shape_per_cta_[d] = mts_[d]*nts_[d];
}


/* -------------------------------- *
 *          Shared Layout           *
 * -------------------------------- */

bool shared_layout::is_loop_latch(ir::phi_node *phi, ir::instruction *terminator){
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


void shared_layout::extract_double_bufferable(ir::value *v, std::shared_ptr<double_buffer_info_t>& res) {
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
  if(!(i_0 && !i_1) &&
     !(dynamic_cast<ir::copy_to_shared_inst*>(i_0) && dynamic_cast<ir::copy_to_shared_inst*>(i_1)) &&
     !(dynamic_cast<ir::masked_load_async_inst*>(i_0) && dynamic_cast<ir::masked_load_async_inst*>(i_1)))
    return;
  if(is_latch_1)
    res.reset(new double_buffer_info_t{value_0, value_1, phi});
  if(is_latch_0)
    res.reset(new double_buffer_info_t{value_1, value_0, phi});
}

static bool is_smem(ir::value* v) {
  if (dynamic_cast<ir::copy_to_shared_inst*>(v) ||
      dynamic_cast<ir::masked_load_async_inst*>(v))
    return true;
  else
    return false;
}

/// param:
///    value_1: next_value
static bool is_multistage_pipe_phi(ir::phi_node* phi, ir::basic_block* bb0, ir::basic_block* bb1, 
    std::vector<ir::value*>& values_0, ir::value*& value_1) {
  ir::value* next = phi;
  while (auto cphi = dynamic_cast<ir::phi_node*>(next)) {
    // smem from previous bb & phi/smem from current bb
    ir::value* c0 = cphi->get_incoming_value(0);
    ir::value* c1 = cphi->get_incoming_value(1);
    ir::basic_block *cbb0 = cphi->get_incoming_block(0);
    ir::basic_block *cbb1 = cphi->get_incoming_block(1);

    if (is_smem(c0)) {
      assert(cbb0 == bb0);
      values_0.push_back(c0);
      if (auto phi1 = dynamic_cast<ir::phi_node*>(c1)) {
        next = phi1;
        continue;
      } else {
        if (is_smem(c1)) {
          value_1 = c1;
          assert(cbb1 == bb1);
          return true;
        } else {
          return false;
        }
      }
    } else
      return false;
  }
  return false;
}

void shared_layout::extract_N_bufferable(ir::value *v, std::shared_ptr<N_buffer_info_t> &res, int &prev_stages) {
  auto* phi = dynamic_cast<ir::phi_node*>(v);
  // if the phi node is nested
  if (!phi)
    return;

  ir::basic_block *bb0 = phi->get_incoming_block(0);
  ir::basic_block *bb1 = phi->get_incoming_block(1);

  std::vector<ir::value*> values_0;
  ir::value* value_1;
  
  if (!is_multistage_pipe_phi(phi, bb0, bb1, values_0, value_1))
    return;

  // double-buffer is a special case
  if (values_0.size() == 1)
    return;

  // compute original values_0 input order
  std::map<ir::value*, int> order;
  int idx = 0;
  for (ir::instruction* instr : *bb0) {
    if (std::find(values_0.begin(), values_0.end(), instr) != values_0.end())
      order[static_cast<ir::value*>(instr)] = idx++;
  }
  assert(order.size() == values_0.size() && "order size incorrect");
  
  int curr_stages = values_0.size() + 1;
  if (curr_stages > prev_stages) {
    res.reset(new N_buffer_info_t{values_0, value_1, phi, order});
    prev_stages = curr_stages;
  }
}


shared_layout::shared_layout(data_layout *arg,
                                 const std::vector<int>& axes,
                                 const std::vector<unsigned>& shape,
                                 const std::vector<ir::value *> &values,
                                 ir::type *ty,
                                 analysis::align* align): data_layout(SHARED, axes, shape, values, align), ty_(ty) {

  size_ = 0;
  arg_layout_ = arg;

  // N-stage buffering
  int prev_stages = 0;
  for (ir::value *v : values)
    extract_N_bufferable(v, N_buffer_, prev_stages);

  // double-buffering
  if (!N_buffer_)
    for(ir::value *v: values)
      extract_double_bufferable(v, double_buffer_);

  // order
  std::vector<int> arg_order = arg ? arg->get_order() : std::vector<int>{0};
  order_ = arg_order;

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
  hmma_dot_a_ = hmma_dot_a;
  hmma_dot_b_ = hmma_dot_b;

  // size
  size_ = ty_->get_primitive_size_in_bits() / 8;
  for(auto s: shape_)
     size_ *= s;
  if(double_buffer_)
    size_ *= 2;
  if (N_buffer_) {
    size_ *= (N_buffer_->firsts.size() + 1);
  }
}

int shared_layout::get_num_stages() const {
  if (double_buffer_)
    return 2;
  if (N_buffer_)
    return N_buffer_->firsts.size() + 1;
  return 1;
}

size_t shared_layout::get_per_stage_elements() const {
  return get_per_stage_size()/(ty_->get_primitive_size_in_bits()/8);
}

/* -------------------------------- *
 * ---- Layouts Inference Pass ---- *
 * -------------------------------- */

layouts::layouts(analysis::axes *axes, analysis::align *align, size_t num_warps, target* tgt)
  : axes_(axes), align_(align), num_warps_(num_warps), tgt_(tgt){ }


void layouts::connect(ir::value *x, ir::value *y) {
  if(x == y)
    return;
  if(!x->get_type()->is_block_ty())
    return;
  if(!y->get_type()->is_block_ty())
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

void layouts::make_graph(ir::instruction *i) {
  for(ir::value* opx: i->ops())
  for(ir::value* opy: i->ops()){
    connect(i, opx);
    connect(opx, opy);
  }
}

void layouts::create(size_t id, const std::vector<ir::value*>& values) {
//  if(layouts_.find(id) != layouts_.end())
//    return;
  auto it_hmma_c = std::find_if(values.begin(), values.end(), &is_hmma_c);
  auto cmp = [](ir::value* x, ir::value *y) {
    std::pair<int, int> xx = {x->get_type()->get_tile_rank(), x->get_type()->get_tile_num_elements()};
    std::pair<int, int> yy = {y->get_type()->get_tile_rank(), y->get_type()->get_tile_num_elements()};
    return xx < yy;
  };
  std::vector<ir::value*> lvalue = values;
  std::remove_if(lvalue.begin(), lvalue.end(), [&](ir::value* v) { return dynamic_cast<ir::trans_inst*>(v); });
  ir::value *largest = *std::max_element(lvalue.begin(), lvalue.end(), cmp);
  const auto& axes = axes_->get(largest);
  const auto& shapes = largest->get_type()->get_block_shapes();
  auto it_cts = std::find_if(values.begin(), values.end(), [](ir::value* v) {
      return dynamic_cast<ir::copy_to_shared_inst*>(v) ||
             dynamic_cast<ir::masked_load_async_inst*>(v);
  });
  // type
  if(it_hmma_c != values.end()){
    ir::instruction *dot = (ir::instruction*)*it_hmma_c;
    ir::value *a = dot->get_operand(0);
    ir::value *b = dot->get_operand(1);
    create(groups_.at(a), values_.at(groups_.at(a)));
    create(groups_.at(b), values_.at(groups_.at(b)));
    layouts_[id] = new mma_layout(num_warps_, axes, shapes, values, align_, tgt_, (shared_layout*)layouts_.at(groups_.at(a)), (shared_layout*)layouts_.at(groups_.at(b)));
  }
  else if(it_cts != values.end()){
    ir::instruction *cts = (ir::instruction*)*it_cts;
    ir::value *arg = cts->get_operand(0);
    create(groups_.at(arg), values_.at(groups_.at(arg)));
    layouts_[id] = new shared_layout(get(arg), axes, shapes, values, largest->get_type()->get_scalar_ty(), align_);
  }
  else{
    layouts_[id] = new scanline_layout(num_warps_, axes, shapes, values, align_, tgt_);
  }
}

void layouts::run(ir::module &mod) {
  // make graph
  graph_.clear();
  layouts_.clear();
  groups_.clear();

  ir::for_each_instruction(mod, [this](ir::instruction* i) {
    make_graph(i);
  });

  // connected components
  graph_.connected_components(&values_, &groups_);

  // create layouts
  for(const auto& x: values_)
    create(x.first, x.second);

  // create temporaries
  size_t id = values_.size();
  ir::for_each_instruction(mod, [this, &id](ir::instruction* i) {
    if(auto *red = dynamic_cast<ir::reduce_inst*>(i)) {
      id++;
      ir::value *arg = red->get_operand(0);
      unsigned axis = red->get_axis();
      // shape
      auto shapes = arg->get_type()->get_block_shapes();
      scanline_layout *layout = get(arg)->to_scanline();
      shapes[axis] = layout->mts(axis);
      // create layout
      layouts_[id] = new shared_layout(layout, axes_->get(arg), shapes, {red}, red->get_type()->get_scalar_ty(), align_);
      tmp_[red] = id;
    }
    if(auto *val = dynamic_cast<ir::cvt_layout_inst*>(i)){
      distributed_layout* out_layout = dynamic_cast<distributed_layout*>(get(val));
      distributed_layout* in_layout = dynamic_cast<distributed_layout*>(get(i->get_operand(0)));
      id++;
      size_t dim = val->get_type()->get_tile_rank();
      ir::type::block_shapes_t shape(dim);
      for(size_t k = 0; k < dim; k++){
        shape[k] = std::max(in_layout->shape_per_cta(k),
                            out_layout->shape_per_cta(k));
      }
      layouts_[id] = new shared_layout(out_layout, axes_->get(val), shape, {val}, val->get_type()->get_scalar_ty(), align_);
      tmp_[val] = id;
    }
    if(auto *atom = dynamic_cast<ir::atomic_inst*>(i)){
      id++;
      layouts_[id] = new shared_layout(nullptr, {}, {1}, {atom}, atom->get_type()->get_scalar_ty(), align_);
      tmp_[atom] = id;
    }
  });

}

}
}
}
