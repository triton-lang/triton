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

inline bool is_hmma(ir::value *v){
  bool result = false;
  if(auto *x = dynamic_cast<ir::dot_inst*>(v)){
    ir::value *a = x->get_operand(0);
    ir::type *a_ty = a->get_type();
    ir::value *b = x->get_operand(1);
    ir::type *b_ty = b->get_type();
    // inputs have to be FP16
    result = a_ty->get_scalar_ty()->is_half_ty() && b_ty->get_scalar_ty()->is_half_ty();
    // reduction has to be multiple of 4
    result = result && ((a_ty->get_tile_shapes()[1]->get_value() % 4) == 0);
  }
  return result;
}

void optimize_dot::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  std::vector<ir::instruction*> to_delete;
  // iterate
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list())
  if(auto dot = dynamic_cast<ir::dot_inst*>(i))
  if(dot->get_operand(1)->get_type()->get_tile_shapes()[1]->get_value() != 1){
    builder.set_insert_point(i);
    ir::value *A = dot->get_operand(0);
    ir::value *B = dot->get_operand(1);
    ir::value *D = dot->get_operand(2);
    bool trans_a = is_trans(A);
    bool trans_b = is_trans(B);

    if(!dot->is_a_trans() && !dot->is_b_trans()){
      if(is_hmma(dot)){
        ir::value *AA = A;
        ir::value *BB = B;
        if(trans_a){
          AA = ((ir::trans_inst*)A)->get_operand(0);
          to_delete.push_back((ir::instruction*)A);
        }
        if(trans_b){
          BB = ((ir::trans_inst*)B)->get_operand(0);
          to_delete.push_back((ir::instruction*)B);
        }
        ir::instruction *dot_atbt = builder.insert(ir::dot_inst::create(AA, BB, D, trans_a, trans_b));
        dot->replace_all_uses_with(dot_atbt);
        to_delete.push_back(dot);
      }
      else{
        // dot(op(a), trans(b))
        if(trans_b){
          ir::value* BB = ((ir::trans_inst*)B)->get_operand(0);
          ir::instruction *NT = builder.insert(ir::dot_inst::create_nt(A, BB, D));
          dot->replace_all_uses_with(NT);
          to_delete.push_back((ir::instruction*)B);
          to_delete.push_back(dot);
        }
        // dot(op(a), b)
        if(!trans_b){
          ir::value* BB = builder.create_trans(B);
          ir::instruction *NT = builder.insert(ir::dot_inst::create_nt(A, BB, D));
          dot->replace_all_uses_with(NT);
          to_delete.push_back(dot);
        }
      }
    }
  }

  for(ir::instruction* i: to_delete)
    i->erase_from_parent();
}

}
}
