#include "triton/codegen/analysis/swizzle.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/codegen/target.h"
#include "triton/ir/type.h"
#include <iostream>

namespace triton{
namespace codegen{
namespace analysis{


void swizzle::run(ir::module &) {
    per_phase_.clear();
    max_phase_.clear();

    for(auto &x: layouts_->get_all()){
      shared_layout* layout = dynamic_cast<shared_layout*>(x.second);
      if(!layout)
        continue;
      ir::value* mma_dot_a = layout->hmma_dot_a();
      ir::value* mma_dot_b = layout->hmma_dot_b();

      if(!mma_dot_a && !mma_dot_b){
        per_phase_[layout] = 1;
        max_phase_[layout] = 1;
        vec_[layout] = 1;
        continue;
      }
      auto ord = layout->get_order();
      scanline_layout* in_layout = dynamic_cast<scanline_layout*>(layout->get_arg_layout());
      int per_phase = 1;
      int dtsize = layout->get_type()->get_scalar_ty()->get_primitive_size_in_bits() / 8;
      if(in_layout)
        per_phase = std::max<int>(128 / (in_layout->mts(ord[0])*in_layout->nts(ord[0])*dtsize), 1);
      else
        per_phase = 1;
      if(tgt_->as_nvidia() && tgt_->as_nvidia()->sm() < 80){
        int inner = mma_dot_a ? 0 : 1;
        per_phase_[layout] = per_phase;
        max_phase_[layout] = (ord[inner] == 1 ? 4 : 4) / per_phase_[layout];
        if(mma_dot_a)
          vec_[layout] = 2*layouts_->get(mma_dot_a)->to_mma()->rep(0);
        else
          vec_[layout] = 2*layouts_->get(mma_dot_b)->to_mma()->rep(1);
      }
      else {
        if (!layout->allow_swizzle()) {
          per_phase_[layout] = 1;
          max_phase_[layout] = 1;
          vec_[layout] = 1;
        } else {
          per_phase_[layout] = per_phase;
          max_phase_[layout] = layout->get_mma_strided() / per_phase_[layout];
          vec_[layout]       = layout->get_mma_vec();
        }
      }
    }
}

}
}
}


