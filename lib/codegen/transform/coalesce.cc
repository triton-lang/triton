#include <algorithm>
#include <iostream>
#include "triton/ir/utils.h"
#include "triton/ir/instructions.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/codegen/transform/coalesce.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/analysis/layout.h"

namespace triton {
namespace codegen{
namespace transform{

coalesce::coalesce(analysis::align* align, analysis::layouts *layouts)
  : align_(align), layout_(layouts) { }


// simplify layout conversions using the following simple rules:
//   - convert(elementwise(x, y)) = elementwise(convert(x), convert(y))
//   - convert2(convert1(x)) if convert1 is the inverse of convert2
ir::instruction* coalesce::simplify(ir::instruction *inst, ir::builder& builder){
  ir::decoalesce_inst* dc = dynamic_cast<ir::decoalesce_inst*>(inst);
  ir::recoalesce_inst* rc = dynamic_cast<ir::recoalesce_inst*>(inst);
  // i must be layout conversion instruction
  if(!dc && !rc)
    return inst;
  // operand must element-wise
  // TODO find a way to check for element-wise easily
  ir::value* op = inst->get_operand(0);
  ir::instruction* new_op = dynamic_cast<ir::instruction*>(op);
  if(new_op->get_id() != ir::INST_BINOP)
    return inst;

  for(size_t i = 0; i < new_op->get_num_operands(); i++){
    ir::value* arg_i = new_op->get_operand(i);
    ir::instruction* new_arg_i = inst->clone();
    new_arg_i->replace_uses_of_with(new_arg_i->get_operand(0), arg_i);
    new_op->replace_uses_of_with(arg_i, new_arg_i);
    builder.insert(new_arg_i);
  }
  std::cout << "replaced" << std::endl;
  return new_op;
}

void coalesce::run(ir::module &mod) {
  ir::builder& builder = mod.get_builder();

  // add layout conversion instructions
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction* i: block->get_inst_list()){
    // coalesce before store
    if(auto x = dynamic_cast<ir::masked_store_inst*>(i))
    if(ir::value* op = x->get_value_operand())
    if(layout_->get(op)->to_mma()){
      builder.set_insert_point(x);
      ir::instruction* new_op = ir::recoalesce_inst::create(op);
      new_op = simplify(new_op, builder);
      builder.insert(new_op);
      x->replace_uses_of_with(op, new_op);
    }
    // uncoalesce after load
    if(auto x = dynamic_cast<ir::masked_load_inst*>(i))
    if(layout_->get(x)->to_mma()){
        builder.set_insert_point_after(x);
        ir::instruction* new_x = ir::decoalesce_inst::create(x);
        builder.insert(new_x);
        x->replace_all_uses_with(new_x);
        new_x->replace_uses_of_with(new_x, x);
    }
  }


}


}
}
}
