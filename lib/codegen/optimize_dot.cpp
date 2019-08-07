#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/module.h"
#include "triton/codegen/optimize_dot.h"
#include "triton/codegen/tune.h"

namespace triton {
namespace codegen{

inline bool is_trans(ir::value *v){
  auto *x = dynamic_cast<ir::trans_inst*>(v);
  if(!x)
    return false;
  std::vector<ir::constant_int*> perm = x->get_perm();
  std::vector<ir::constant_int*> ref;
  ir::type *int32_ty = ir::type::get_int32_ty(v->get_type()->get_context());
  for(size_t i = 0; i < perm.size(); i++)
    ref.push_back(ir::constant_int::get(int32_ty, i));
  std::swap(ref[0], ref[1]);
  // true is perm == ref
  return std::equal(perm.begin(), perm.end(), ref.begin());
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
//     reduction has to be multiple of 4
//    result = result && ((a_ty->get_tile_shapes()[1]->get_value() % 4) == 0);
  }
  return result;
}

void optimize_dot::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  // iterate
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list())
  if(auto dot = dynamic_cast<ir::dot_inst*>(i)){
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
        }
        else{
          if(auto *T = dynamic_cast<ir::trans_inst*>(A)){
            std::vector<ir::constant_int*> perm(T->get_perm());
            std::swap(perm[0], perm[1]);
            AA = builder.create_trans(T->get_operand(0), perm);
            T->replace_all_uses_with(AA);
            trans_a = true;
          }
        }
        if(trans_b){
          BB = ((ir::trans_inst*)B)->get_operand(0);
        }
        else{
//          if(auto *T = dynamic_cast<ir::trans_inst*>(A)){
//            std::vector<ir::constant_int*> perm(T->get_perm());
//            std::swap(perm[0], perm[1]);
//            AA = builder.create_trans(T->get_operand(0), perm);
//            T->replace_all_uses_with(AA);
//            trans_a = true;
//          }
        }
        ir::instruction *dot_atbt = builder.insert(ir::dot_inst::create(AA, BB, D, trans_a, trans_b));
        dot->replace_all_uses_with(dot_atbt);
      }
      else{
        // dot(op(a), trans(b))
        if(trans_b){
          ir::value* BB = ((ir::trans_inst*)B)->get_operand(0);
          ir::instruction *NT = builder.insert(ir::dot_inst::create_nt(A, BB, D));
          dot->replace_all_uses_with(NT);
        }
        // dot(op(a), b)
        if(!trans_b){
          // create permutations
          size_t size = B->get_type()->get_tile_shapes().size();
          std::vector<ir::constant_int*> perm(size);
          ir::type *int32_ty = ir::type::get_int32_ty(B->get_type()->get_context());
          for(size_t i = 0; i < size; i++)
            perm[i] = ir::constant_int::get(int32_ty, i);
          std::swap(perm[0], perm[1]);
          // replace NN -> NT (trans)
          ir::value* BB = builder.create_trans(B, perm);
          ir::instruction *NT = builder.insert(ir::dot_inst::create_nt(A, BB, D));
          dot->replace_all_uses_with(NT);
        }
      }
    }
  }
}

}
}
