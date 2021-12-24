#include "triton/codegen/transform/inline.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"

namespace triton{
namespace codegen{
namespace transform{

void inliner::do_inline(ir::function* fn, ir::call_inst* callsite, ir::builder& builder){
  ir::function* parent_fn = callsite->get_fn();
  // create new basic block to branch to when the called function return
  // and optionally also a phi node to hold the return values
  ir::basic_block* exit = ir::basic_block::create(callsite->get_parent()->get_context(), "after_call", parent_fn);
  ir::phi_node* ret_val = builder.create_phi(fn->get_fn_type()->get_return_ty(), 0);
  // clone all the basic blocks
  std::vector<ir::basic_block*> new_blocks;
  for(ir::basic_block* block: fn->blocks()){
   ir::context& ctx = block->get_context();
   const std::string& name = block->get_name();
   new_blocks.push_back(ir::basic_block::create(ctx, name, parent_fn));
  }
  // get arguments `fn` is called with
  std::vector<ir::value*> tgt_args(callsite->op_begin(), callsite->op_end());
  // replace the callsite by a branch to the function's entry block
  builder.set_insert_point(callsite);
  ir::value* br = builder.create_br(new_blocks[0]);
  callsite->replace_all_uses_with(br);


  std::vector<ir::argument*> src_args(fn->args().begin(), fn->args().end());
  for(ir::basic_block* old_block: fn->blocks()){
    ir::basic_block* new_block = ir::basic_block::create(old_block->get_context(), old_block->get_name(), callsite->get_fn());
    builder.set_insert_point(new_block);
    for(ir::instruction* old_instr: old_block->get_inst_list()){
      // `ret` instruction is a special case:
      // instead of returning we need to branch to after the function call
      if(ir::return_inst* ret = dynamic_cast<ir::return_inst*>(old_instr)){


        continue;
      }
      //
      ir::instruction* new_instr = old_instr->clone();
      for(size_t i = 0; i < src_args.size(); i++)
        new_instr->replace_uses_of_with(src_args[i], tgt_args[i]);
      builder.insert(new_instr);
    }
  }
}

void inliner::run(ir::module &mod) {

  // gather all call sites
  std::map<ir::function*, std::vector<ir::call_inst*>> callsites;
  for(ir::function* fn: mod.get_function_list())
  for(ir::basic_block* block: fn->blocks())
  for(ir::instruction* instr: block->get_inst_list())
  if(ir::call_inst* call = dynamic_cast<ir::call_inst*>(instr)){
    callsites[call->get_fn()].push_back(call);
  }

  // replace call sites with function bodies, one by one
  for(auto& x: callsites){
    ir::function* fn = x.first;
    for(ir::call_inst* callsite: x.second)
      do_inline(fn, callsite, mod.get_builder());

  }
}

}
}
}
