#include <functional>
#include <algorithm>
#include "ast/ast.h"
#include "ir/constant.h"
#include "ir/function.h"
#include "ir/module.h"
#include "ir/basic_block.h"
#include "ir/builder.h"
#include "ir/type.h"
#include <iostream>


namespace tdl{

namespace ast{

/* node */
ir::value *node::explicit_cast(ir::builder &builder, ir::value *src, ir::type *dst_ty){
  ir::type *src_scalar_ty = src->get_type()->get_scalar_ty();
  ir::type *dst_scalar_ty = dst_ty->get_scalar_ty();
  bool src_signed = false;
  bool dst_signed = false;
  if(src_scalar_ty == dst_scalar_ty)
    return src;
  else if(src_scalar_ty->is_integer_ty() && src_signed && dst_scalar_ty->is_floating_point_ty())
    return builder.create_si_to_fp(src, dst_ty);

  else if(src_scalar_ty->is_integer_ty() && !src_signed && dst_scalar_ty->is_floating_point_ty())
    return builder.create_ui_to_fp(src, dst_ty);

  else if(src_scalar_ty->is_floating_point_ty() && dst_scalar_ty->is_integer_ty() && dst_signed)
    return builder.create_fp_to_si(src, dst_ty);

  else if(src_scalar_ty->is_floating_point_ty() && dst_scalar_ty->is_integer_ty() && !dst_signed)
    return builder.create_fp_to_ui(src, dst_ty);

  else if(src_scalar_ty->is_floating_point_ty() && dst_scalar_ty->is_floating_point_ty() &&
          src_scalar_ty->get_fp_mantissa_width() < dst_scalar_ty->get_fp_mantissa_width())
    return builder.create_fp_ext(src, dst_ty);

  else if(src_scalar_ty->is_floating_point_ty() && dst_scalar_ty->is_floating_point_ty() &&
          src_scalar_ty->get_fp_mantissa_width() > dst_scalar_ty->get_fp_mantissa_width())
    return builder.create_fp_trunc(src, dst_ty);

  else if(src_scalar_ty->is_integer_ty() && dst_scalar_ty->is_integer_ty() &&
          src_scalar_ty->get_integer_bitwidth())
    return builder.create_int_cast(src, dst_ty, dst_signed);

  else
    throw std::runtime_error("unreachable");
}


void node::implicit_cast(ir::builder &builder, ir::value *&lhs, ir::value *&rhs,
                          bool &is_float, bool &is_ptr, bool &is_int, bool &is_signed){
  // Input types
  ir::type *left_ty = lhs->get_type()->get_scalar_ty();
  ir::type *right_ty = rhs->get_type()->get_scalar_ty();
  // One operand is pointer
  if(left_ty->is_pointer_ty() || right_ty->is_pointer_ty()){
    if(left_ty->is_pointer_ty() && right_ty->is_pointer_ty())
      throw std::runtime_error("invalid operands");
    if(right_ty->is_pointer_ty())
      std::swap(lhs, rhs);
    is_ptr = true;
  }
  // One operand is double
  else if(left_ty->is_double_ty() || right_ty->is_double_ty()){
    ir::value *&to_convert = left_ty->is_double_ty()?rhs:lhs;
    to_convert = explicit_cast(builder, to_convert, builder.get_double_ty());
    is_float = true;
  }
  // One operand is float
  else if(left_ty->is_float_ty() || right_ty->is_float_ty()){
    ir::value *&to_convert = left_ty->is_float_ty()?rhs:lhs;
    to_convert = explicit_cast(builder, to_convert, builder.get_float_ty());
    is_float = true;
  }
  // Both operands are integers
  else if(left_ty->is_integer_ty() && right_ty->is_integer_ty()){
    is_int = true;
    is_signed = true; // always signed for now
    if(left_ty->get_integer_bitwidth() != right_ty->get_integer_bitwidth()){
      ir::value *&to_convert = (left_ty->get_integer_bitwidth() > right_ty->get_integer_bitwidth())?rhs:lhs;
      ir::type *dst_ty = (to_convert==lhs)?right_ty:left_ty;
      to_convert = explicit_cast(builder, to_convert, dst_ty);
    }
  }
  // Not reachable
  else
    throw std::runtime_error("unreachable");
}

void node::implicit_broadcast(ir::module *mod, ir::value *&arg, ir::type *ty) {
  ir::value *tmp = ir::undef_value::get(ty);
  implicit_broadcast(mod, arg, tmp);
}

void node::implicit_broadcast(ir::module *mod, ir::value *&lhs, ir::value *&rhs){
  ir::builder &builder = mod->get_builder();
  ir::type *lhs_ty = lhs->get_type();
  ir::type *rhs_ty = rhs->get_type();
  // Both are scalar
  if(!lhs_ty->is_tile_ty() && !rhs_ty->is_tile_ty())
    return;
  // One argument is scalar
  if(lhs_ty->is_tile_ty() ^ rhs_ty->is_tile_ty()){
    auto &shapes = lhs_ty->is_tile_ty()?lhs_ty->get_tile_shapes():rhs_ty->get_tile_shapes();
    auto &scalar = lhs_ty->is_tile_ty()?rhs:lhs;
    scalar = builder.create_splat(scalar, shapes);
    return;
  }
  // Both are arrays
  std::vector<unsigned> lhs_shapes = lhs->get_type()->get_tile_shapes();
  std::vector<unsigned> rhs_shapes = rhs->get_type()->get_tile_shapes();
  if(lhs_shapes == rhs_shapes)
    return;
  int lhs_dim = lhs_shapes.size();
  int rhs_dim = rhs_shapes.size();
  std::vector<unsigned> &shortest = (lhs_dim < rhs_dim)?lhs_shapes:rhs_shapes;
  std::vector<unsigned> &longest  = (lhs_dim < rhs_dim)?rhs_shapes:lhs_shapes;
  size_t ndim = longest.size();
  int off = longest.size() - shortest.size();
  for(int i = longest.size() - 1; i>= 0; i--){
    if(shortest[off + i] != longest[i] && shortest[off + i] != 1 && longest[i] != 1)
      throw std::runtime_error("cannot broadcast");
  }
  // Pad
  for(size_t i = 0; i < off; i++)
    shortest.insert(shortest.begin(), 1);
  ir::value *&target = (lhs_dim < rhs_dim)?lhs:rhs;
  if(off > 0)
    target = builder.create_reshape(target, shortest);
  // Broadcast
  std::vector<unsigned> shapes(ndim);
  for(size_t i = 0; i < ndim; i++)
    shapes[i] = std::max(shortest[i], longest[i]);
  if(shapes != lhs_shapes)
    lhs = builder.create_broadcast(lhs, shapes);
  if(shapes != rhs_shapes)
    rhs = builder.create_broadcast(rhs, shapes);
}

/* Translation unit */
ir::value* translation_unit::codegen(ir::module *mod) const{
  decls_->codegen(mod);
  return nullptr;
}

/* Declaration specifier */
ir::type* declaration_specifier::type(ir::module *mod) const {
  ir::context &ctx = mod->get_context();
  switch (spec_) {
  case VOID_T:      return ir::type::get_void_ty(ctx);
  case INT1_T:      return ir::type::get_int1_ty(ctx);
  case INT8_T:      return ir::type::get_int8_ty(ctx);
  case INT16_T:     return ir::type::get_int16_ty(ctx);
  case INT32_T:     return ir::type::get_int32_ty(ctx);
  case INT64_T:     return ir::type::get_int64_ty(ctx);
  case FLOAT32_T:   return ir::type::get_float_ty(ctx);
  case FLOAT64_T:   return ir::type::get_double_ty(ctx);
  default: throw std::runtime_error("unreachable");
  }
}

/* Parameter */
ir::type* parameter::type(ir::module *mod) const {
  return decl_->type(mod, spec_->type(mod));
}

const identifier *parameter::id() const {
  return decl_->id();
}

/* Declarators */
ir::type* declarator::type(ir::module *mod, ir::type *type) const{
  if(ptr_)
    return type_impl(mod, ptr_->type(mod, type));
  return type_impl(mod, type);
}

// Identifier
ir::type* identifier::type_impl(ir::module *, ir::type *type) const{
  return type;
}

const std::string &identifier::name() const{
  return name_;
}

// Tile
ir::type* tile::type_impl(ir::module*, ir::type *type) const{
  std::vector<unsigned> shapes;
  for(constant *cst: shapes_->values())
    shapes.push_back(cst->value());
  return ir::tile_type::get(type, shapes);
}


// Pointer
ir::type* pointer::type_impl(ir::module*, ir::type *type) const{
  return ir::pointer_type::get(type, 1);
}

// Function
void function::bind_parameters(ir::module *mod, ir::function *fn) const{
  std::vector<ir::argument*> args = fn->args();
  assert(args.size() == args_->values().size());
  for(size_t i = 0; i < args.size(); i++){
    parameter *param_i = args_->values().at(i);
    const identifier *id_i = param_i->id();
    if(id_i){
      args[i]->set_name(id_i->name());
      mod->set_value(id_i->name(), nullptr, args[i]);
      mod->set_type(id_i->name(), nullptr, args[i]->get_type());
    }
  }
}

ir::type* function::type_impl(ir::module* mod, ir::type *type) const{
  std::vector<ir::type*> types;
  for(parameter* param: args_->values())
    types.push_back(param->type(mod));
  return ir::function_type::get(type, types);
}

/* Function definition */
ir::value* function_definition::codegen(ir::module *mod) const{
  ir::function_type *prototype = (ir::function_type*)header_->type(mod, spec_->type(mod));
  const std::string &name = header_->id()->name();
  ir::function *fn = mod->get_or_insert_function(name, prototype);
  header_->bind_parameters(mod, fn);
  ir::basic_block *entry = ir::basic_block::create(mod->get_context(), "entry", fn);
  mod->seal_block(entry);
  mod->get_builder().set_insert_point(entry);
  body_->codegen(mod);
  mod->get_builder().create_ret_void();
  return nullptr;
}

/* Statements */
ir::value* compound_statement::codegen(ir::module* mod) const{
  if(decls_)
    decls_->codegen(mod);
  if(statements_)
    statements_->codegen(mod);
  return nullptr;
}

/* expression statement */
ir::value* expression_statement::codegen(ir::module *mod) const{
  ir::value *expr = expr_->codegen(mod);
  if(mask_) {
    ir::instruction *itn = dynamic_cast<ir::instruction*>(expr);
    assert(itn);
    ir::value *mask = mask_->codegen(mod);
    itn->set_mask(mask);
  }
  return expr;
}

/* Iteration statement */
ir::value* iteration_statement::codegen(ir::module *mod) const{
  ir::builder &builder = mod->get_builder();
  ir::context &ctx = mod->get_context();
  ir::basic_block *current_bb = builder.get_insert_block();
  ir::function *fn = current_bb->get_parent();
  ir::basic_block *loop_bb = ir::basic_block::create(ctx, "loop", fn);
  init_->codegen(mod);
  builder.create_br(loop_bb);
  builder.set_insert_point(loop_bb);
  statements_->codegen(mod);
  exec_->codegen(mod);
  ir::value *cond = stop_->codegen(mod);
  ir::basic_block *stop_bb = builder.get_insert_block();
  ir::basic_block *next_bb = ir::basic_block::create(ctx, "postloop", fn);
  builder.create_cond_br(cond, loop_bb, next_bb);
  mod->seal_block(stop_bb);
  mod->seal_block(loop_bb);
  mod->seal_block(builder.get_insert_block());
  mod->seal_block(next_bb);
  builder.set_insert_point(next_bb);
  return nullptr;
}

/* Selection statement */
ir::value* selection_statement::codegen(ir::module* mod) const{
  ir::builder &builder = mod->get_builder();
  ir::context &ctx = mod->get_context();
  ir::function *fn = builder.get_insert_block()->get_parent();
  ir::value *cond = cond_->codegen(mod);
  ir::basic_block *then_bb = ir::basic_block::create(ctx, "then", fn);
  ir::basic_block *else_bb = else_value_?ir::basic_block::create(ctx, "else", fn):nullptr;
  ir::basic_block *endif_bb = ir::basic_block::create(ctx, "endif", fn);
  // Branch
  if(else_value_)
    builder.create_cond_br(cond, then_bb, else_bb);
  else
    builder.create_cond_br(cond, then_bb, endif_bb);
  // Then
  builder.set_insert_point(then_bb);
  then_value_->codegen(mod);
  builder.create_br(endif_bb);
  mod->seal_block(then_bb);
  // Else
  if(else_value_){
    builder.set_insert_point(else_bb);
    else_value_->codegen(mod);
    builder.create_br(endif_bb);
    mod->seal_block(else_bb);
  }
  // Endif
  builder.set_insert_point(endif_bb);
}

/* Declaration */
ir::value* declaration::codegen(ir::module* mod) const{
  for(initializer *init: init_->values())
    init->specifier(spec_);
  init_->codegen(mod);
  return nullptr;
}

/* Initializer */
ir::type* initializer::type_impl(ir::module *mod, ir::type *type) const{
  return decl_->type(mod, type);
}

void initializer::specifier(const declaration_specifier *spec) {
  spec_ = spec;
}

ir::value* initializer::codegen(ir::module * mod) const{
  ir::type *ty = decl_->type(mod, spec_->type(mod));
  std::string name = decl_->id()->name();
  ir::value *value = ir::undef_value::get(ty);
  if(expr_){
    value = expr_->codegen(mod);
    value = explicit_cast(mod->get_builder(), value, ty);
    implicit_broadcast(mod, value, ty);
  }
  value->set_name(name);
  mod->set_value(name, value);
  mod->set_type(name, ty);
  return value;
}

/*------------------*/
/*    Expression    */
/*------------------*/
/* Binary operator */
ir::value *binary_operator::llvm_op(ir::module *mod, ir::builder &builder, ir::value *lhs, ir::value *rhs, const std::string &name) const
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
  if(op_ == EQ && is_float)
    return builder.create_fcmpOEQ(lhs, rhs, name);
  if(op_ == EQ && is_int)
    return builder.create_icmpEQ(lhs, rhs, name);
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

ir::value* binary_operator::codegen(ir::module *mod) const{
  ir::value *lhs = lhs_->codegen(mod);
  ir::value *rhs = rhs_->codegen(mod);
  ir::value *result = llvm_op(mod, mod->get_builder(), lhs, rhs, "");
  return result;
}

/* Builtin expression */

// get_global_range
ir::value* get_global_range::codegen(ir::module *mod) const {
  ir::builder &builder = mod->get_builder();
  return builder.create_get_global_range(axis_->value(), size_->value());
}


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
  return mod->get_builder().create_matmul(A, B, C);
}



/* Postfix expression */
ir::value* indexing_expression::codegen(ir::module *mod) const{
  ir::value *in = mod->get_value(id_->name());
  const std::vector<slice*> &slices = slices_->values();
  std::vector<unsigned> in_shapes = in->get_type()->get_tile_shapes();
  std::vector<unsigned> out_shapes(slices.size());
  size_t current = 0;
  for(size_t i = 0; i < out_shapes.size(); i++)
    out_shapes[i] = (slices[i]->type()==NEWAXIS)?1:in_shapes[current++];
  return mod->get_builder().create_reshape(in, out_shapes);
}

/* Unary operator */
ir::value *unary_operator::llvm_op(ir::builder &builder, ir::value *arg, const std::string &name) const{
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

ir::value* unary_operator::codegen(ir::module *mod) const{
  ir::value *arg = arg_->codegen(mod);
  ir::value *result = llvm_op(mod->get_builder(), arg, "");
  return result;
}

/* Cast operator */
ir::value *cast_operator::llvm_op(ir::builder &builder, ir::type *T, ir::value *arg, const std::string &name) const{
  return nullptr;
}

ir::value* cast_operator::codegen(ir::module *mod) const{
  ir::value *arg = arg_->codegen(mod);
  ir::type *T = T_->type(mod);
  return llvm_op(mod->get_builder(), T, arg, "");
}

/* Conditional expression */
ir::value *conditional_expression::llvm_op(ir::builder &builder, ir::value *cond, ir::value *true_value, ir::value *false_value, const std::string &name) const{
  return nullptr;
}

ir::value *conditional_expression::codegen(ir::module *mod) const{
  ir::value *cond = cond_->codegen(mod);
  ir::value *true_value = true_value_->codegen(mod);
  ir::value *false_value = false_value_->codegen(mod);
  return llvm_op(mod->get_builder(), cond, true_value, false_value, "");
}

/* Assignment expression */
ir::value *assignment_expression::codegen(ir::module *mod) const{
  ir::value *rvalue = rvalue_->codegen(mod);
  if(auto *x = dynamic_cast<const named_expression*>(lvalue_))
    mod->set_value(x->id()->name(), rvalue);
  else if(auto* x = dynamic_cast<const unary_operator*>(lvalue_)){
    assert(x->get_op()==DEREF);
    assert(x->lvalue());
    ir::value *ptr = x->lvalue()->codegen(mod);
    rvalue = mod->get_builder().create_store(ptr, rvalue);
  }
  return rvalue;
}

/* Type name */
ir::type *type_name::type(ir::module *mod) const{
  return decl_->type(mod, spec_->type(mod));
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
  return ir::constant_range::get((ir::constant*)first_->codegen(mod),
                                 (ir::constant*)last_->codegen(mod));
}

/* Named */
ir::value* named_expression::codegen(ir::module *mod) const{
  const std::string &name = id()->name();
  return mod->get_value(name);
}


}

}
