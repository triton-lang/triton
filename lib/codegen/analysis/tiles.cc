#include <algorithm>
#include <cstdlib>
#include <numeric>
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/tiles.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/ir/instructions.h"
#include "triton/ir/type.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/context_impl.h"
#include "triton/ir/constant.h"
#include "triton/driver/device.h"



namespace triton{
namespace codegen{
namespace analysis{

tiles::tiles(size_t num_warps, analysis::align *align, analysis::axes *axes, analysis::layout *layout):
    num_warps_(num_warps), align_(align), axes_(axes), layout_(layout)
{ }

bool is_hmma(ir::value *v){
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



bool tiles::hmma(ir::value *value) {
  return hmma_.at(layout_->id(value));
}

int tiles::mts(ir::value *value, unsigned ax) {
  return mts_.at(axes_->get(value, ax));
}

int tiles::nts(ir::value *value, unsigned ax) {
  return nts_.at(axes_->get(value, ax));
}

int tiles::fpw(ir::value *value, unsigned ax) {
  return fpw_.at(axes_->get(value, ax));
}

int tiles::wpt(ir::value *value, unsigned ax) {
  return wpt_.at(axes_->get(value, ax));
}

std::vector<int> tiles::order(ir::value *v) {
  auto ret = order_[layout_->id(v)];
  return ret;
}

const std::map<int, ir::value*>& tiles::largest() {
  return largest_;
}


unsigned clamp(unsigned x, unsigned lo, unsigned hi) {
  return std::min(std::max(x, lo), hi);
}


void tiles::init_hmma_tile(ir::value *i) {
  auto ord = order(i);
  auto shapes = i->get_type()->get_tile_shapes();
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
    fpw_[axes_->get(i, d)] = fpw[d];
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
    wpt_[axes_->get(i, d)] = wpt[d];
  /* sanity check */
  unsigned effective_num_warps = 1;
  for(size_t d = 0; d < shapes.size(); d++)
    effective_num_warps *= wpt_[axes_->get(i, d)];
  if(num_warps_ != effective_num_warps)
    throw std::runtime_error("cannot create a kernel with this amount of warps");
}

void tiles::init_scanline_tile(ir::value *i) {
  auto ord = order(i);
  auto shapes = i->get_type()->get_tile_shapes();
  unsigned size = i->get_type()->get_tile_num_elements();
  unsigned ld = ord[0];
  unsigned num_threads = num_warps_*32;
  unsigned current = num_threads;
  nts_[axes_->get(i, ld)] = clamp(size / num_threads, 1, 4);
  mts_[axes_->get(i, ld)] = clamp(current, 1, shapes[ld] / nts_[axes_->get(i, ld)]);
  current = current / mts_[axes_->get(i, ld)];
  for(size_t d = 1; d < shapes.size(); d++){
    ld = ord[d];
    nts_[axes_->get(i, ld)] = 1;
    mts_[axes_->get(i, ld)] = clamp(current, 1, shapes[ld]);
    current = current / mts_[axes_->get(i, ld)];
  }
  /* sanity check */
  unsigned effective_num_threads = 1;
  for(size_t d = 0; d < shapes.size(); d++)
    effective_num_threads *= mts_[axes_->get(i, d)];
  if(num_threads != effective_num_threads)
    throw std::runtime_error("cannot create a kernel with this amount of warps");
}

void extract_io_use(ir::value *v, std::set<ir::io_inst*>& result) {
  for(ir::user* u: v->get_users()){
    auto i = dynamic_cast<ir::io_inst*>(u);
    if(i && i->get_pointer_operand() == v)
      result.insert(i);
  }
}


void tiles::run(ir::module &) {
  hmma_.clear();
  largest_.clear();
  size_t num_groups = layout_->get_num_groups();
  // helpers
  auto rank = [](ir::value* v) {
    ir::type *ty = v->get_type();
    size_t ret = 0;
    if(ty->is_tile_ty())
      for(int s: ty->get_tile_shapes())
        ret += s > 1;
    return ret;
  };
  // find out which groups require hmma layout
  for(size_t i = 0; i < num_groups; i++) {
    const auto& values = layout_->values(i);
    hmma_[i] = std::any_of(values.begin(), values.end(), &is_hmma);
  }
  // find out which value is the largest in each group
  for(size_t i = 0; i < num_groups; i++) {
    const auto& values = layout_->values(i);
    auto cmp = [&rank](ir::value* x, ir::value *y) { return rank(x) < rank(y); };
    largest_[i] = *std::max_element(values.begin(), values.end(), cmp);
  }
  // find out the order of a group
  for(size_t i = 0; i < num_groups; i++){
    std::set<ir::io_inst*> io;
    for(ir::value* v: layout_->values(i))
      extract_io_use(v, io);
    auto cmp = [&rank](ir::io_inst* x, ir::io_inst *y) {
      return rank(x->get_pointer_operand()) < rank(y->get_pointer_operand());
    };
    auto it = std::max_element(io.begin(), io.end(), cmp);
    std::vector<int> order(rank(largest_[i]));
    std::iota(order.begin(), order.end(), 0);
    if(it != io.end()) {
      auto max_contiguous = align_->contiguous((*it)->get_pointer_operand());
      std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
        return max_contiguous[a] > max_contiguous[b]; }
      );
    }
    order_[i] = order;
    std::cout << "order: " << order[0] << " " << order[1] << std::endl;

  }
  // tiling parameters
  for(auto x: largest_){
    ir::value *i = x.second;
    if(!i->get_type()->is_tile_ty())
      continue;
    /* HMMA parameters*/
    if(hmma_[x.first])
      init_hmma_tile(i);
    else
      init_scanline_tile(i);
  }
}

}
}
}
