#include "triton/codegen/transform/inline.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"

namespace triton{
namespace codegen{
namespace transform{

void inliner::run(ir::module &mod) {

  std::map<ir::function*, std::vector<ir::call_inst*>> call_sites;

  for(ir::function* fn: mod.get_function_list())
  for(ir::basic_block* block: fn->blocks())
  for(ir::instruction* instr: block->get_inst_list())
  if(ir::call_inst* call = dynamic_cast<ir::call_inst*>(instr)){
    call_sites[call->get_fn()].push_back(call);
  }
}

}
}
}
