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


unsigned clamp(unsigned x, unsigned lo, unsigned hi) {
  return std::min(std::max(x, lo), hi);
}


void tiles::init_hmma_tile(const layout_t& layout) {
  auto ord = layout.order;
  auto shapes = layout.i->get_type()->get_tile_shapes();
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
    fpw_[layout.axes[d]] = fpw[d];
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
    wpt_[layout.axes[d]] = wpt[d];
  /* sanity check */
  unsigned effective_num_warps = 1;
  for(size_t d = 0; d < shapes.size(); d++)
    effective_num_warps *= wpt_[layout.axes[d]];
  if(num_warps_ != effective_num_warps)
    throw std::runtime_error("cannot create a kernel with this amount of warps");
}

void tiles::init_scanline_tile(const layout_t& layout) {
  auto ord = layout.order;
  auto shapes = layout.shapes;
  unsigned size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<int>());
  unsigned ld = ord[0];
  unsigned num_threads = num_warps_*32;
  unsigned current = num_threads;
  nts_[layout.axes[ld]] = clamp(size / num_threads, 1, 4);
  mts_[layout.axes[ld]] = clamp(current, 1, shapes[ld] / nts_[layout.axes[ld]]);
  current = current / mts_[layout.axes[ld]];
  for(size_t d = 1; d < shapes.size(); d++){
    ld = ord[d];
    nts_[layout.axes[ld]] = 1;
    mts_[layout.axes[ld]] = clamp(current, 1, shapes[ld]);
    current = current / mts_[layout.axes[ld]];
  }
  /* sanity check */
  unsigned effective_num_threads = 1;
  for(size_t d = 0; d < shapes.size(); d++)
    effective_num_threads *= mts_[layout.axes[d]];
//  std::cout << num_threads << " " << effective_num_threads << std::endl;
  if(num_threads != effective_num_threads)
    throw std::runtime_error("cannot create a kernel with this amount of warps");
}

void tiles::run(ir::module &) {
  // tiling parameters
  for(auto x: layout_->get_all()){
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
