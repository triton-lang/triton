#include <iostream>
#include <algorithm>
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/codegen/transform/reorder.h"

namespace triton {
namespace codegen{
namespace transform{

void reorder::run(ir::module& mod){
//    ir::builder &builder = mod.get_builder();
//    std::vector<std::pair<ir::instruction*, ir::value*>> to_replace;

//    for(ir::function *fn: mod.get_function_list())
//    for(ir::basic_block *block: fn->blocks())
//    for(ir::instruction* i: block->get_inst_list()){
//      if(auto* ld = dynamic_cast<ir::masked_load_inst*>(i)){
//        ir::value* _ptr = ld->get_pointer_operand();
//        ir::value* _msk = ld->get_mask_operand();
//        ir::value* _val = ld->get_false_value_operand();
//        auto ptr = std::find(block->begin(), block->end(), _ptr);
//        auto msk = std::find(block->begin(), block->end(), _msk);
//        auto val = std::find(block->begin(), block->end(), _val);
//        if(ptr == block->end() || msk == block->end() || val == block->end())
//          continue;
//        auto it = std::find(block->begin(), block->end(), i);
//        int dist_ptr = std::distance(ptr, it);
//        int dist_msk = std::distance(msk, it);
//        int dist_val = std::distance(val, it);
//        if(dist_ptr < dist_msk && dist_ptr < dist_val)
//          builder.set_insert_point(++ptr);
//        if(dist_msk < dist_ptr && dist_msk < dist_val)
//          builder.set_insert_point(++msk);
//        if(dist_val < dist_ptr && dist_val < dist_msk)
//          builder.set_insert_point(++val);
//        ir::value* new_ld = builder.create_masked_load(_ptr, _msk, _val);
//        to_replace.push_back(std::make_pair(ld, new_ld));
//      }
//    }

//    for(auto& x: to_replace)
//      x.first->replace_all_uses_with(x.second);

}

}
}
}
