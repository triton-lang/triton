#include <stack>
#include "triton/ir/cfg.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/function.h"

namespace triton{
namespace ir{

std::vector<basic_block*> cfg::reverse_post_order(function* fn) {
  std::stack<basic_block*> stack;
  std::set<basic_block*> visited;
  std::vector<basic_block*> result;
  // initialize stack
  for(ir::basic_block* block: fn->blocks())
    if(block->get_predecessors().empty())
      stack.push(block);
  // DFS
  while(!stack.empty()) {
    basic_block* current = stack.top();
    stack.pop();
    result.push_back(current);
    visited.insert(current);
    for(basic_block* succ: current->get_successors())
      if(visited.find(succ) == visited.end())
        stack.push(succ);
  }
  return std::move(result);
}

}
}
