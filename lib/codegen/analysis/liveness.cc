#include <iostream>
#include "triton/codegen/instructions.h"
#include "triton/codegen/analysis/liveness.h"
#include "triton/codegen/transform/cts.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/instructions.h"
#include "triton/ir/value.h"
#include "triton/ir/utils.h"

namespace triton{
namespace codegen{
namespace analysis{

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

inline void extract_double_bufferable(ir::instruction *i, std::map<ir::value*, double_buffer_info_t>& result) {
  auto* phi = dynamic_cast<ir::phi_node*>(i);
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
  if(!i_0 || !i_1 || storage_info.at(i_0->get_id()).first != SHARED || storage_info.at(i_1->get_id()).first != SHARED)
    return;
  if(is_latch_1)
    result[value_0] = double_buffer_info_t{value_1, phi};
  if(is_latch_0)
    result[value_1] = double_buffer_info_t{value_0, phi};
}


// Entry point
void liveness::run(ir::module &mod) {
  double_.clear();
  indices_.clear();
  intervals_.clear();

  // set of pair of values that can be double-buffered
  ir::for_each_instruction(mod, [this](ir::instruction* i) {
    extract_double_bufferable(i, this->double_);
  });


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
      ir::instruction* instr = dynamic_cast<ir::instruction*>(v);
      if(!instr)
        continue;
      if(storage_info.at(instr->get_id()).first != SHARED)
        continue;
      unsigned start = i.second;
      unsigned end = start;
      for(ir::value *u: v->get_users()){
        start = std::min(start, indices_.at(u));
        end = std::max(end, indices_.at(u));
      }
      intervals_[v] = segment{start, end};
    }
    // Double-Buffering
    // Arrays are live throughout the end of the loop
    auto it = intervals_.begin();
    while(it != intervals_.end()) {
      ir::value *x = it->first;
      auto dit = double_.find(x);
      if(dit != double_.end()) {
        ir::value *y = dit->second.latch;
        unsigned start = intervals_[x].start;
        unsigned end = intervals_[y].end;
        intervals_[x] = segment{start, end};
        intervals_.erase(y);
      }
      it++;
    }
  }
}

}
}
}
