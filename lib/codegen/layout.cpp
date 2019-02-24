#include "triton/codegen/layout.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"

namespace tdl{
namespace codegen{


shared_view_info layout::get_shared_view(ir::value *v, unsigned idx){
  return shared_views_.at(v)[idx];
}

unsigned layout::get_num_shared_views(ir::value *v){
  return shared_views_.at(v).size();
}

// Phi node
void layout::add_phi_nodes(ir::value *v){
  if(ir::phi_node *phi = dynamic_cast<ir::phi_node*>(v))
  if(shared_views_.find(phi) != shared_views_.end())
  for(ir::value *v: phi->ops()){
    shared_views_[v] = shared_views_[phi];
    for(shared_view_info &info: shared_views_[v])
      info.has_dedicated_storage = false;
  }
}

// Memory Layout
void layout::add_shared_views(ir::value *v){
  // GEMM has shared inputs
  if(dynamic_cast<ir::matmul_inst*>(v))
    shared_views_[v].push_back({v, true});
  if(dynamic_cast<ir::reshape_inst*>(v))
    shared_views_[v].push_back({v, true});
}

// Entry point
void layout::run(ir::module &mod) {
for(ir::function *fn: mod.get_function_list()){
  // Non-phis
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *instr: block->get_inst_list()) {
    add_shared_views(instr);
  }
  // Phi nodes
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *instr: block->get_inst_list()) {
    add_phi_nodes(instr);
  }
}
}

}
}
