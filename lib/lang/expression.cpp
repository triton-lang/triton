#include "triton/lang/expression.h"
#include "triton/lang/declaration.h"
#include "triton/ir/constant.h"
#include "triton/ir/module.h"
#include "triton/ir/builder.h"
#include "triton/ir/type.h"


namespace triton{

namespace lang{


/* Binary operator */
ir::value *binary_expression::llvm_op(ir::module *mod, ir::builder &builder, ir::value *lhs, ir::value *rhs, const std::string &name) const
{
  bool is_float = false, is_ptr = false, is_int = false, is_signed = false;
  implicit_cast(builder, lhs, rhs, is_float, is_ptr, is_int, is_signed);
  implicit_broadcast(mod, lhs, rhs);
  if(op_==MUL && is_float)
    return builder.create_fmul(lhs, rhs, name);
  if(op_==MUL && is_int)
    return builder.create_mul(lhs, rhs, name);
  if(op_==DIV && is_float)
    return builder.create_fdiv(lhs, rhs, name);
  if(op_==DIV && is_int && is_signed)
    return builder.create_sdiv(lhs, rhs, name);
  if(op_==DIV && is_int && !is_signed)
    return builder.create_udiv(lhs, rhs, name);
  if(op_==MOD && is_float)
    return builder.create_frem(lhs, rhs, name);
  if(op_==MOD && is_int && is_signed)
    return builder.create_srem(lhs, rhs, name);
  if(op_==MOD && is_int && !is_signed)
    return builder.create_urem(lhs, rhs, name);
  if(op_==ADD && is_float)
    return builder.create_fadd(lhs, rhs, name);
  if(op_==ADD && is_int)
    return builder.create_add(lhs, rhs);
  if(op_==ADD && is_ptr)
    return builder.create_gep(lhs, {rhs});
  if(op_==SUB && is_float)
    return builder.create_fsub(lhs, rhs, name);
  if(op_==SUB && is_int)
    return builder.create_sub(lhs, rhs, name);
  if(op_==SUB && is_ptr)
    return builder.create_gep(lhs, {builder.create_neg(rhs)});
  if(op_==LEFT_SHIFT)
    return builder.create_shl(lhs, rhs, name);
  if(op_==RIGHT_SHIFT)
    return builder.create_ashr(lhs, rhs, name);
  if(op_ == LT && is_float)
    return builder.create_fcmpOLT(lhs, rhs, name);
  if(op_ == LT && is_int && is_signed)
    return builder.create_icmpSLT(lhs, rhs, name);
  if(op_ == LT && is_int && !is_signed)
    return builder.create_icmpULT(lhs, rhs, name);
  if(op_ == GT && is_float)
    return builder.create_fcmpOGT(lhs, rhs, name);
  if(op_ == GT && is_int && is_signed)
    return builder.create_icmpSGT(lhs, rhs, name);
  if(op_ == GT && is_int && !is_signed)
    return builder.create_icmpUGT(lhs, rhs, name);
  if(op_ == LE && is_float)
    return builder.create_fcmpOLE(lhs, rhs, name);
  if(op_ == LE && is_int && is_signed)
    return builder.create_icmpSLE(lhs, rhs, name);
  if(op_ == LE && is_int && !is_signed)
    return builder.create_icmpULE(lhs, rhs, name);
  if(op_ == GE && is_float)
    return builder.create_fcmpOGE(lhs, rhs, name);
  if(op_ == GE && is_int && is_signed)
    return builder.create_icmpSGE(lhs, rhs, name);
  if(op_ == GE && is_int && !is_signed)
    return builder.create_icmpUGE(lhs, rhs, name);
  if(op_ == EQ && is_ptr)
    return builder.create_icmpEQ(lhs, rhs, name);
  if(op_ == EQ && is_float)
    return builder.create_fcmpOEQ(lhs, rhs, name);
  if(op_ == EQ && is_int)
    return builder.create_icmpEQ(lhs, rhs, name);
  if(op_ == NE && is_ptr)
    return builder.create_icmpNE(lhs, rhs, name);
  if(op_ == NE && is_float)
    return builder.create_fcmpONE(lhs, rhs, name);
  if(op_ == NE && is_int)
    return builder.create_icmpNE(lhs, rhs, name);
  if(op_ == AND)
    return builder.create_and(lhs, rhs, name);
  if(op_ == XOR)
    return builder.create_xor(lhs, rhs, name);
  if(op_ == OR)
    return builder.create_or(lhs, rhs, name);
  if(op_ == LAND)
    return builder.create_and(lhs, rhs, name);
  if(op_ == LOR)
    return builder.create_or(lhs, rhs, name);
  throw std::runtime_error("unreachable");
}

ir::value* binary_expression::codegen(ir::module *mod) const{
  ir::value *lhs = lhs_->codegen(mod);
  ir::value *rhs = rhs_->codegen(mod);
  ir::value *result = llvm_op(mod, mod->get_builder(), lhs, rhs, "");
  return result;
}

/* Builtin expression */

// alloc constant
ir::value* alloc_const_expression::codegen(ir::module *mod) const {
  ir::type *ty = spec_->type(mod);
  ir::constant_int *size = (ir::constant_int*)size_->codegen(mod);
  ir::alloc_const *res = new ir::alloc_const(ty, size);
  return res;
}

// get_range_id
ir::value* get_range_id_expression::codegen(ir::module *mod) const {
  return mod->get_builder().create_get_range_id(axis_->value());
}

// get_num_program
ir::value* get_num_program_expression::codegen(ir::module *mod) const {
  return mod->get_builder().create_get_num_program(axis_->value());
}

// atomic cas
ir::value* atomic_cas_expression::codegen(ir::module *mod) const {
  ir::value *ptr = ptr_->codegen(mod);
  ir::value *cmp = cmp_->codegen(mod);
  ir::value *val = val_->codegen(mod);
  return mod->get_builder().create_atomic_cas(ptr, cmp, val);
}

// atomic exch
ir::value* atomic_exch_expression::codegen(ir::module *mod) const {
  ir::value *ptr = ptr_->codegen(mod);
  ir::value *val = val_->codegen(mod);
  return mod->get_builder().create_atomic_exch(ptr, val);
}

// atomic add
ir::value* atomic_add_expression::codegen(ir::module *mod) const {
  ir::value *ptr = ptr_->codegen(mod);
  ir::value *val = val_->codegen(mod);
  return mod->get_builder().create_atomic_add(ptr, val);
}

// matmul
ir::value* matmul_expression::codegen(ir::module *mod) const {
  ir::value *A = A_->codegen(mod);
  ir::value *B = B_->codegen(mod);
  ir::value *C = C_->codegen(mod);
//  unsigned M = A->get_type()->get_tile_shapes()[0];
//  unsigned N = B->get_type()->get_tile_shapes()[1];
//  ir::type *scalar_ty = A->get_type()->get_scalar_ty();
//  ir::type *tile_ty = ir::tile_type::get(scalar_ty, {M, N});
//  ir::value *tmp = ir::undef_value::get(tile_ty);
//  implicit_broadcast(mod, tmp, C);
  return mod->get_builder().create_dot(A, B, C);
}

// min
ir::value* min_expression::codegen(ir::module *mod) const {
  ir::value* cmp = binary_expression(LT, (node*)x_, (node*)y_).codegen(mod);
  ir::value* x = ((ir::cmp_inst*)cmp)->get_operand(0);
  ir::value* y = ((ir::cmp_inst*)cmp)->get_operand(1);
  return mod->get_builder().create_select(cmp, x, y);
}

// max
ir::value* max_expression::codegen(ir::module *mod) const {
  ir::value* cmp = binary_expression(GT, (node*)x_, (node*)y_).codegen(mod);
  ir::value* x = ((ir::cmp_inst*)cmp)->get_operand(0);
  ir::value* y = ((ir::cmp_inst*)cmp)->get_operand(1);
  return mod->get_builder().create_select(cmp, x, y);
}

// select
ir::value* select_expression::codegen(ir::module *mod) const {
  ir::value* pred = pred_->codegen(mod);
  ir::value* if_value = if_value_->codegen(mod);
  ir::value* else_value = else_value_->codegen(mod);
  return mod->get_builder().create_select(pred, if_value, else_value);
}

// trans
ir::value* trans_expression::codegen(ir::module *mod) const {
  return mod->get_builder().create_trans(arg_->codegen(mod));
}

// sqrt
ir::value* sqrt_expression::codegen(ir::module *mod) const {
  return mod->get_builder().create_sqrt(arg_->codegen(mod));
}


// reduce
ir::value* reduce_expression::codegen(ir::module *mod) const {
  return mod->get_builder().create_reduce(arg_->codegen(mod));
}

/* Postfix expression */
ir::value* indexing_expression::codegen(ir::module *mod) const{
  ir::value *in = lhs_->codegen(mod);
  const std::vector<slice*> &slices = slices_->values();
  auto in_shapes = in->get_type()->get_tile_shapes();
  ir::type::tile_shapes_t::value_type one = ir::tile_type::make_one(mod->get_context());
  ir::type::tile_shapes_t out_shapes(slices.size());
  // create shapes
  size_t current = 0;
  for(size_t i = 0; i < out_shapes.size(); i++)
    out_shapes[i] = (slices[i]->type()==NEWAXIS)?one:in_shapes[current++];
  return mod->get_builder().create_reshape(in, out_shapes);
}


/* Unary operator */
ir::value *unary_expression::llvm_op(ir::builder &builder, ir::value *arg, const std::string &name) const{
  ir::type *atype = arg->get_type();
  bool is_float = atype->is_floating_point_ty();
  bool is_int = atype->is_integer_ty();
  if(op_ == INC)
    return builder.create_add(arg, builder.get_int32(1), name);
  if(op_ == DEC)
    return builder.create_sub(arg, builder.get_int32(1), name);
  if(op_ == PLUS)
    return arg;
  if(op_ == MINUS && is_float)
    return builder.create_fneg(arg, name);
  if(op_ == MINUS && is_int)
    return builder.create_neg(arg, name);
  if(op_ == ADDR)
    throw std::runtime_error("not supported");
  if(op_ == DEREF)
    return builder.create_load(arg, name);
  if(op_ == COMPL)
    throw std::runtime_error("not supported");
  if(op_ == NOT)
    return builder.create_not(arg, name);
  throw std::runtime_error("unreachable");
}

ir::value* unary_expression::codegen(ir::module *mod) const{
  ir::value *arg = arg_->codegen(mod);
  ir::value *result = llvm_op(mod->get_builder(), arg, "");
  return result;
}

/* Cast operator */
ir::value *cast_expression::llvm_op(ir::builder &builder, ir::type *T, ir::value *arg, const std::string &name) const{
  return nullptr;
}

ir::value* cast_expression::codegen(ir::module *mod) const{
  ir::value *arg = arg_->codegen(mod);
  ir::type *T = T_->type(mod);
  return llvm_op(mod->get_builder(), T, arg, "");
}

/* Conditional expression */
ir::value *conditional_expression::codegen(ir::module *mod) const {
  ir::builder &builder = mod->get_builder();
  ir::value *mask = cond_->codegen(mod);
  ir::value *true_value = true_value_->codegen(mod);
  ir::value *false_value = false_value_->codegen(mod);
  bool is_float, is_ptr, is_int, is_signed;
  implicit_cast(builder, true_value, false_value, is_float, is_ptr, is_int, is_signed);
  implicit_broadcast(mod, mask, true_value);
  implicit_broadcast(mod, mask, false_value);
  if(ir::load_inst* load = dynamic_cast<ir::load_inst*>(true_value)){
    load->erase_from_parent();
    return builder.create_masked_load(load->get_pointer_operand(), mask, false_value);
  }
  if(ir::load_inst* load = dynamic_cast<ir::load_inst*>(false_value)){
    load->erase_from_parent();
    return builder.create_masked_load(load->get_pointer_operand(), mask, true_value);
  }
  throw std::runtime_error("not implemented");
}

/* Assignment expression */
ir::value *assignment_expression::codegen(ir::module *mod) const{
  ir::value *rvalue = rvalue_->codegen(mod);
  if(auto *x = dynamic_cast<const named_expression*>(lvalue_)){
    ir::type *ty = mod->get_scope().types.at(x->id()->name());
    rvalue = explicit_cast(mod->get_builder(), rvalue, ty);
    implicit_broadcast(mod, ty, rvalue);
    mod->set_value(x->id()->name(), rvalue);
  }
  else if(auto* x = dynamic_cast<const unary_expression*>(lvalue_)){
    assert(x->get_op()==DEREF);
    assert(x->lvalue());
    ir::value *ptr = x->lvalue()->codegen(mod);
    rvalue = mod->get_builder().create_store(ptr, rvalue);
  }
  return rvalue;
}


/* String literal */
ir::value* string_literal::codegen(ir::module *) const{
  throw std::runtime_error("not supported");
//  return ir::constant_data_array::get_string(mod->get_context(), value_);
}

/* Constant */
ir::value* constant::codegen(ir::module *mod) const{
  return mod->get_builder().get_int32(value_);
}

int constant::value() const{
  return value_;
}

/* Constant range */
ir::value* constant_range::codegen(ir::module *mod) const{
  return ir::constant_range::get((ir::constant_int*)first_->codegen(mod),
                                 (ir::constant_int*)last_->codegen(mod));
}

/* Named */
ir::value* named_expression::codegen(ir::module *mod) const{
  const std::string &name = id()->name();
  const auto& declarations = mod->get_scope().types;
  if(declarations.find(name) == declarations.end())
    throw std::runtime_error("variable " + name + " not declared");
  return mod->get_value(name);
}

}

}
