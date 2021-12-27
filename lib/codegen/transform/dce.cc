#include "triton/codegen/transform/dce.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/module.h"
#include "triton/ir/utils.h"
#include <iostream>

namespace triton {
namespace codegen{
namespace transform{


void dce::run(ir::module &mod) {
  std::list<ir::instruction*> work_list;
  std::set<ir::instruction*> marked;

  // initialize work-list
  for(ir::function *fn: mod.get_function_list()){
    std::vector<ir::basic_block*> rpo = ir::cfg::reverse_post_order(fn);
    // iterate through blocks
    for(ir::basic_block *block: rpo)
    for(ir::instruction *i: block->get_inst_list()){
      switch(i->get_id()){
        case ir::INST_RETURN:
        case ir::INST_UNCOND_BRANCH:
        case ir::INST_COND_BRANCH:
        case ir::INST_UNMASKED_STORE:
        case ir::INST_MASKED_STORE:
        case ir::INST_ATOMIC_CAS:
        case ir::INST_ATOMIC_RMW:
        case ir::INST_ATOMIC_EXCH:
        case ir::INST_BARRIER: {
          work_list.push_back(i);
          marked.insert(i);
          break;
        }
        default:
          break;
      }
    }
  }

  // mark -- ignore branches
  while(!work_list.empty()){
    ir::instruction* current = work_list.back();
    work_list.pop_back();
    // mark instruction operands
    for(ir::value* op: current->ops()) {
      if(auto *i = dynamic_cast<ir::instruction*>(op)){
        if(marked.insert(i).second)
          work_list.push_back(i);
      }
    }
    // TODO: mark last intstruction of current's reverse-dominance frontier
  }

  // sweep -- delete non-branch unmarked instructions
  std::vector<ir::instruction*> to_delete;
  for(ir::function *fn: mod.get_function_list()){
    std::vector<ir::basic_block*> rpo = ir::cfg::reverse_post_order(fn);
    // iterate through blocks
    for(ir::basic_block *block: rpo)
    for(ir::instruction *i: block->get_inst_list()){
      if(marked.find(i) == marked.end())
        to_delete.push_back(i);
    }
  }


  // delete
  for(ir::instruction* i: to_delete)
    i->erase_from_parent();
}

}
}
}
