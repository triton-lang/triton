#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/module.h"
#include "triton/codegen/optimize_dot.h"
#include "triton/codegen/tune.h"

namespace triton {
namespace codegen{

inline bool is_trans(ir::value *v){
  return dynamic_cast<ir::trans_inst*>(v) != nullptr;
}

void optimize_dot::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  std::vector<ir::instruction*> to_delete;
  // iterate
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list())
  if(auto dot = dynamic_cast<ir::dot_inst*>(i))
  if(dot->get_operand(1)->get_type()->get_tile_shapes()[1]->get_value() != 1)
  if(!dot->is_a_trans() && !dot->is_b_trans()){
    builder.set_insert_point(i);
    ir::value *A = dot->get_operand(0);
    ir::value *B = dot->get_operand(1);
    ir::value *D = dot->get_operand(2);
    // dot(op(a), trans(b))
    if(is_trans(B)){
      ir::value* BN = ((ir::trans_inst*)B)->get_operand(0);
      ir::instruction *NT = builder.insert(ir::dot_inst::create_nt(A, BN, D));
      dot->replace_all_uses_with(NT);
      to_delete.push_back((ir::instruction*)B);
      to_delete.push_back(dot);
    }
    // dot(op(a), b)
    if(!is_trans(B)){
      ir::value* BT = builder.create_trans(B);
      ir::instruction *NT = builder.insert(ir::dot_inst::create_nt(A, BT, D));
      dot->replace_all_uses_with(NT);
      to_delete.push_back(dot);
    }
  }

  for(ir::instruction* i: to_delete)
    i->erase_from_parent();
}

}
}
