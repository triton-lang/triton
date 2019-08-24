#include <algorithm>
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/codegen/transform/peephole.h"
#include <iostream>
namespace triton {
namespace codegen{
namespace transform{


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
//   reduction has to be multiple of 4
//    result = result && ((a_ty->get_tile_shapes()[1]->get_value() % 4) == 0);
  }
  return result;
}

ir::value* rewrite_trans_phi_impl(ir::value *value, ir::builder &builder,
                                 const std::vector<ir::constant_int*>& perm) {
  if(auto phi = dynamic_cast<ir::phi_node*>(value)) {
    // transpose operands
    std::vector<ir::value*> incs;
    for(unsigned n = 0; n < phi->get_num_incoming(); n++)
      incs.push_back(rewrite_trans_phi_impl(phi->get_incoming_value(n), builder, perm));
    // create phi for transposed values
    builder.set_insert_point(phi);
    ir::phi_node* result = builder.create_phi(incs[0]->get_type(), incs.size());
    for(unsigned n = 0; n < phi->get_num_incoming(); n++)
      result->add_incoming(incs[n], phi->get_incoming_block(n));
    return result;
  }
  else if(auto i = dynamic_cast<ir::instruction*>(value)){
    ir::basic_block* block = i->get_parent();
    auto it = std::find(block->begin(), block->end(), i);
    it++;
    builder.set_insert_point(it);
    ir::instruction *trans = (ir::instruction*)builder.create_trans(i, perm);
    trans->set_operand(0, i);
    return trans;
  }
  return nullptr;
}

bool peephole::rewrite_trans_phi(ir::instruction* value, ir::builder& builder) {
  auto trans = dynamic_cast<ir::trans_inst*>(value);
  if(!trans)
    return false;
  auto users = trans->get_users();
  auto ops = trans->ops();
  if(users.size() > 1 || ops.size() > 1)
    return false;
  ir::value* op = *ops.begin();
  // trans(phi) -> phi(trans(), trans()...)
  auto* phi = dynamic_cast<ir::phi_node*>(op);
  if(!phi)
    return false;
  ir::value* new_phi = rewrite_trans_phi_impl(phi, builder, trans->get_perm());
  if(!new_phi)
    return false;
  trans->replace_all_uses_with(new_phi);

  return true;
}

bool peephole::rewrite_dot_hmma(ir::dot_inst *dot, ir::builder& builder, bool trans_a, bool trans_b,
                                ir::value *A, ir::value *B, ir::value *D){
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
    if(auto *T = dynamic_cast<ir::trans_inst*>(A)){
      std::vector<ir::constant_int*> perm(T->get_perm());
      std::swap(perm[0], perm[1]);
      AA = builder.create_trans(T->get_operand(0), perm);
      T->replace_all_uses_with(AA);
      trans_a = true;
    }
  }
  if(!trans_a && !trans_b)
    return false;

  ir::instruction *dot_atbt = builder.insert(ir::dot_inst::create(AA, BB, D, trans_a, trans_b));
  dot->replace_all_uses_with(dot_atbt);

  return true;
}

bool peephole::rewrite_dot_fp32(ir::dot_inst *dot, ir::builder& builder, bool trans_a, bool trans_b,
                                ir::value *A, ir::value *B, ir::value *D){
  // dot(op(a), trans(b))
  if(trans_b){
    ir::value* BB = ((ir::trans_inst*)B)->get_operand(0);
    ir::instruction *NT = builder.insert(ir::dot_inst::create_nt(A, BB, D));
    dot->replace_all_uses_with(NT);
    return true;
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
    return true;
  }
  return false;
}

bool peephole::rewrite_dot(ir::instruction *value, ir::builder& builder){
  // dot(a, b, 0) + c -> dot(a, b, c)
  auto add = dynamic_cast<ir::binary_operator*>(value);
  if(add && add->get_op() == ir::binary_op_t::FAdd) {
    ir::value *lhs = add->get_operand(0);
    ir::value *rhs = add->get_operand(1);
    ir::dot_inst *lhs_dot = dynamic_cast<ir::dot_inst*>(lhs);
    ir::dot_inst *rhs_dot = dynamic_cast<ir::dot_inst*>(rhs);
    if(!lhs_dot && !rhs_dot)
      return false;
    ir::dot_inst *dot = lhs_dot ? lhs_dot : rhs_dot;
    ir::value *other = (dot == lhs) ? rhs : lhs;
    ir::value *acc = dot->get_operand(2);
    ir::splat_inst *splat = dynamic_cast<ir::splat_inst*>(acc);
    ir::constant_fp *_0 = nullptr;
    if(splat)
      _0 = dynamic_cast<ir::constant_fp*>(splat->get_operand(0));
    if(!(_0 && _0->get_value() == 0.0))
      return false;
    ir::value *a = dot->get_operand(0);
    ir::value *b = dot->get_operand(1);
    ir::value * new_dot = builder.insert(ir::dot_inst::create(a, b, other,
                                                              dot->is_a_trans(), dot->is_b_trans(),
                                                              dot->get_name()));
    add->replace_all_uses_with(new_dot);
    return true;
  }

  // dot(a, b, c)
  auto dot = dynamic_cast<ir::dot_inst*>(value);
  if(!dot)
    return false;
  builder.set_insert_point(value);
  ir::value *A = dot->get_operand(0);
  ir::value *B = dot->get_operand(1);
  ir::value *D = dot->get_operand(2);
  bool trans_a = is_trans(A);
  bool trans_b = is_trans(B);
  // only consider dot-nn
  if(dot->is_a_trans() || dot->is_b_trans())
    return false;
  // hmma
  if(is_hmma(dot)){
    return rewrite_dot_hmma(dot, builder, trans_a, trans_b, A, B, D);
  }
  else
    return rewrite_dot_fp32(dot, builder, trans_a, trans_b, A, B, D);
}

bool peephole::rewrite_unit_red(ir::instruction *value, ir::builder& builder){
  auto x = dynamic_cast<ir::reduce_inst*>(value);
  if(!x)
    return false;
  ir::value *arg = x->get_operand(0);
  auto shapes = arg->get_type()->get_tile_shapes();
  if(shapes[x->get_axis()] == 1){
    builder.set_insert_point(x);
    ir::value* new_red = builder.create_reshape(arg, x->get_type()->get_tile_shapes());
    x->replace_all_uses_with(new_red);
    return true;
  }
  return false;
}

bool peephole::rewrite_gep_ptr_min_off_plus_off(ir::instruction *value, ir::builder& builder) {
  auto x = dynamic_cast<ir::getelementptr_inst*>(value);
  if(!x)
    return false;
  auto y = dynamic_cast<ir::getelementptr_inst*>(x->get_pointer_operand());
  if(!y)
    return false;
  auto idx = *y->idx_begin();
  auto z = dynamic_cast<ir::binary_operator*>(idx);
  if(!z)
    return false;
  bool is_sub = z->get_op() == ir::binary_op_t::Sub;
  auto *lhs = dynamic_cast<ir::constant_int*>(z->get_operand(0));
  bool is_lhs_0 = lhs && (lhs->get_value()==0);
  bool is_rhs_eq_x_rhs = z->get_operand(1) == *x->idx_begin();
  if(is_sub && is_lhs_0 && is_rhs_eq_x_rhs){
    x->replace_all_uses_with(y->get_pointer_operand());
    return true;
  }
  return false;
}


void peephole::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  // keep track of whether any modification was made
  std::set<ir::value*> seen;
  size_t n_seen;

  // rewrite dots first
  do{
    n_seen = seen.size();
    for(ir::function *fn: mod.get_function_list())
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction* i: block->get_inst_list()){
      if(seen.find(i) != seen.end())
        continue;
      bool was_modified = rewrite_dot(i, builder);
      if(was_modified)
        seen.insert(i);
    }
  }while(seen.size() != n_seen);

  // rewrite other ops
  seen.clear();
  do{
    n_seen = seen.size();
    for(ir::function *fn: mod.get_function_list())
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction* i: block->get_inst_list()){
      if(seen.find(i) != seen.end())
        continue;
      bool was_modified = false;
      was_modified = was_modified || rewrite_trans_phi(i, builder);
      was_modified = was_modified || rewrite_unit_red(i, builder);
      was_modified = was_modified || rewrite_gep_ptr_min_off_plus_off(i, builder);
      if(was_modified)
        seen.insert(i);
    }
  }while(seen.size() != n_seen);
}

}
}
}
