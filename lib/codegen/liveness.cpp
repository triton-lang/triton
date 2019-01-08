#include "codegen/liveness.h"
#include "codegen/layout.h"
#include "ir/basic_block.h"
#include "ir/function.h"
#include "ir/instructions.h"
#include "ir/value.h"

namespace tdl{
namespace codegen{


// Entry point
void liveness::run(ir::function *fn) {
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
    if(!layouts_->get_num_shared_views(v))
      continue;
    if(!layouts_->get_shared_view(v, 0).has_dedicated_storage)
      continue;
    unsigned start = i.second;
    unsigned end = start;
    for(ir::value *u: v->get_users()){
      start = std::min(start, indices_.at(u));
      end = std::max(end, indices_.at(u));
    }
    intervals_[v] = segment{start, end};
  }
}

}
}
