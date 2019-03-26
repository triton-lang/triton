#include "triton/codegen/liveness.h"
#include "triton/codegen/buffer_info.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/instructions.h"
#include "triton/ir/value.h"

namespace triton{
namespace codegen{


// Entry point
inline bool is_shared(ir::value* v) {
  if(auto x = dynamic_cast<ir::copy_to_shared_inst*>(v))
    return true;
  if(auto x = dynamic_cast<ir::phi_node*>(v)){
    bool res = true;
    for(unsigned inc = 0; inc < x->get_num_incoming(); inc++)
      res = res && is_shared(x->get_incoming_value(inc));
    return res;
  }
  return false;
}

void liveness::run(ir::module &mod) {
  for(ir::function *fn: mod.get_function_list()){
    // Assigns index to each instruction
    slot_index index = 0;
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *instr: block->get_inst_list()){
      index += 1;
      indices_.insert({instr, index});
    }
    // Liveness analysis
    // Creates live intervals
    for(auto i: indices_){
      ir::value *v = i.first;
      if(!info_->is_shared(v) || info_->get_reference(v))
        continue;
      unsigned start = i.second;
      unsigned end = start;
      for(ir::value *u: v->get_users()){
        start = std::min(start, indices_.at(u));
        end = std::max(end, indices_.at(u));
      }
      intervals_[v] = segment{start, end};
    }
    std::cout << "Number of intervals: " << intervals_.size() << std::endl;
  }
}

}
}
