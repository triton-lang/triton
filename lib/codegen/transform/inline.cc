#include <iostream>
#include "triton/codegen/transform/inline.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/utils.h"

namespace triton{
namespace codegen{
namespace transform{


bool fncmp::operator()(ir::function* x, ir::function* y) const {
  auto fn_list = x->get_parent()->get_function_list();
  return std::find(fn_list.begin(), fn_list.end(), x) < std::find(fn_list.begin(), fn_list.end(), y);
};

void inliner::do_inline(ir::function* fn, ir::call_inst* callsite, ir::builder& builder,
                        std::list<ir::call_inst*>& callsites){
  ir::basic_block* parent_block = callsite->get_parent();
  ir::function* parent_fn = parent_block->get_parent();
   // the parent block is split into block A and block B:
  //   - block A (`new_blocks[0]`) is the entry block of the inlined function
  //   - block B (`exit`) resumes execution of the parent function
  ir::basic_block* entry = parent_block->split_before(callsite, fn->get_name());
  ir::basic_block* exit = entry->get_successors()[0];
  std::vector<ir::basic_block*> new_blocks = {entry};
  for(size_t i = 1; i < fn->blocks().size(); i++){
   ir::basic_block* block = fn->blocks()[i];
   ir::context& ctx = block->get_context();
   const std::string& name = block->get_parent()->get_name() + "_" + block->get_name();
   new_blocks.push_back(ir::basic_block::create(ctx, name, parent_fn));
  }
  // a phi node holds the return values of the inlined function
  if(exit->get_inst_list().empty())
    builder.set_insert_point(exit);
  else
    builder.set_insert_point(exit->get_first_non_phi());
  ir::phi_node* exit_val = builder.create_phi(fn->get_fn_type()->get_return_ty(), 0);
  callsite->replace_all_uses_with(exit_val);
  callsite->erase_from_parent();
  // get arguments `fn` is called with
  std::vector<ir::value*> tgt_args(callsite->op_begin(), callsite->op_end());
  std::vector<ir::argument*> src_args(fn->args().begin(), fn->args().end());
  // Actually generate the instructions:
  // - Remove the branch created by basic_block::split_before
  // - Clone all instructions
  // - Replace `ret` with incoming nodes to `exit_val` and branches to `exit`
  ir::instruction* terminator = new_blocks[0]->get_inst_list().back();
//  new_blocks[0]->get_inst_list().back()->erase_from_parent();
  terminator->erase_from_parent();
  std::map<ir::instruction*, ir::instruction*> inst_map;
  std::map<ir::argument*, ir::value*> arg_map;
  for(size_t k = 0; k < fn->args().size(); k++)
    arg_map[fn->args()[k]] = callsite->ops()[k];
  std::vector<ir::basic_block*> rpo = ir::cfg::reverse_post_order(fn);
  for(size_t i = 0; i < new_blocks.size(); i++){
    ir::basic_block* old_block = fn->blocks()[i];
    ir::basic_block* new_block = new_blocks[i];
    builder.set_insert_point(new_block);
    for(ir::instruction* old_inst: old_block->get_inst_list()){
      // clone instruction
      ir::instruction* new_inst = old_inst->clone();
      // replace basic block
      for(size_t k = 0; k < new_blocks.size(); k++)
        new_inst->replace_uses_of_with(fn->blocks()[k], new_blocks[k]);
      // replace values
      for(size_t k = 0; k < new_inst->get_num_operands(); k++){
        ir::value* op = new_inst->get_operand(k);
        if(auto arg_op = dynamic_cast<ir::argument*>(op))
          new_inst->set_operand(k, arg_map.at(arg_op));
        if(auto inst_op = dynamic_cast<ir::instruction*>(op))
          if(inst_map.find(inst_op) != inst_map.end())
            new_inst->set_operand(k, inst_map.at(inst_op));
      }
       // `ret` instruction is a special case:
      // instead of returning we need to branch to after the function call
      if(ir::return_inst* ret = dynamic_cast<ir::return_inst*>(new_inst)){
        if(ir::value* ret_val = ret->get_return_value())
          exit_val->add_incoming(ret_val, new_block);
        new_inst = ir::branch_inst::create(exit);
      }
      inst_map[old_inst] = new_inst;
      builder.insert(new_inst);
    }
  }
  if(exit_val->get_num_incoming() == 1)
    exit_val->replace_all_uses_with(exit_val->get_incoming_value(0));
  // done -- make sure insert point is properly set to exit block
  builder.set_insert_point(exit);
}

void inliner::run(ir::module &mod) {

  // gather all call sites
  while(true){
    std::map<ir::function*, size_t> counts;
    for(ir::function* fn: mod.get_function_list())
      counts[fn] = 0;

    std::list<ir::call_inst*> callsites;
    for(ir::function* fn: mod.get_function_list()){
      for(ir::basic_block* block: fn->blocks())
      for(ir::instruction* instr: block->get_inst_list())
      if(ir::call_inst* call = dynamic_cast<ir::call_inst*>(instr)){
        callsites.push_back(call);
        counts[call->get_fn()] += 1;
      }
    }

    for(auto& count: counts){
      if(count.first != mod.get_function_list().front() &&
         count.second == 0)
        count.first->get_parent()->remove_function(count.first);
    }

    if(callsites.empty())
      break;

    for(ir::call_inst* call: callsites)
      do_inline(call->get_fn(), call, mod.get_builder(), callsites);
  }


}

}
}
}
