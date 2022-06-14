#include <stack>
#include <iostream>
#include "triton/ir/utils.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"

namespace triton{
namespace ir{

std::vector<basic_block*> cfg::post_order(function* fn) {
  std::stack<basic_block*> stack;
  std::set<basic_block*> visited;
  std::vector<basic_block*> result;
  // initialize stack
  for(ir::basic_block* block: fn->blocks())
    if(block->get_predecessors().empty()){
      stack.push(block);
      visited.insert(block);
    }
  // DFS
  while(!stack.empty()) {
    basic_block* current = stack.top();
    bool tail = true;
    for(basic_block* succ: current->get_successors())
      if(visited.find(succ) == visited.end()){
        stack.push(succ);
        visited.insert(succ);
        tail = false;
        break;
      }
    if(tail){
      stack.pop();
      result.push_back(current);
    }
  }
  return result;
}

std::vector<basic_block*> cfg::reverse_post_order(function* fn) {
  auto result = post_order(fn);
  std::reverse(result.begin(), result.end());
  return result;
}

void for_each_instruction_backward(module &mod, const std::function<void (instruction *)> &do_work) {
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: cfg::post_order(fn))
  for(ir::instruction *i: block->get_inst_list())
    do_work(i);
}

void for_each_instruction(module &mod, const std::function<void (instruction *)> &do_work) {
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: cfg::reverse_post_order(fn))
  for(ir::instruction *i: block->get_inst_list())
    do_work(i);
}

void for_each_value(module &mod, const std::function<void (value *)> &do_work) {
  std::set<ir::value*> seen;
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: cfg::reverse_post_order(fn))
  for(ir::instruction *i: block->get_inst_list()){
    for(ir::value *op: i->ops()){
      if(seen.insert(op).second)
        do_work(op);
    }
    if(seen.insert(i).second)
      do_work(i);
  }
}

}
}
