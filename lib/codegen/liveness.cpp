#include "codegen/liveness.h"
#include "codegen/buffer_info.h"
#include "ir/basic_block.h"
#include "ir/function.h"
#include "ir/module.h"
#include "ir/instructions.h"
#include "ir/value.h"

namespace tdl{
namespace codegen{


// Entry point
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
}
}

}
}
