#include "triton/codegen/analysis/swizzle.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/codegen/target.h"
#include "triton/ir/type.h"

namespace triton{
namespace codegen{
namespace analysis{


void swizzle::run(ir::module &mod) {
    per_phase_.clear();
    max_phase_.clear();

    for(auto &x: layouts_->get_all()){
      shared_layout* layout = dynamic_cast<shared_layout*>(x.second);
      if(!layout)
        continue;
      auto ord = layout->get_order();
      scanline_layout* in_layout = dynamic_cast<scanline_layout*>(layout->get_arg_layout());
      if(!in_layout)
        continue;
      int dtsize = layout->get_type()->get_scalar_ty()->get_primitive_size_in_bits() / 8;
      if(tgt_->as_nvidia()->sm() < 80){
        per_phase_[layout] = std::max<int>(128 / (in_layout->mts(ord[0])*in_layout->nts(ord[0])*dtsize), 1);
        int inner = layout->is_hmma_dot_a() ? 0 : 1;
        vec_[layout] = 8;
        max_phase_[layout] = (ord[inner] == 1 ? 8 : 4) / per_phase_[layout];
      }
      else{
        per_phase_[layout] = std::max<int>(128 / (in_layout->mts(ord[0])*in_layout->nts(ord[0])*dtsize), 1);
        max_phase_[layout] = 8 / per_phase_[layout];
        vec_[layout] = 8;
      }
    }
}

}
}
}


