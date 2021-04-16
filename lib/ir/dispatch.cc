#include "triton/ir/dispatch.h"
#include <iostream>

namespace triton{
namespace ir{


ir::value* throw_unreachable(std::string key) {
  throw std::runtime_error("Encountered unimplemented code path in `" + key + "`. "
                           "This is likely a bug on our side.");
  return 0;
}

//===----------------------------------------------------------------------===//
//                              Programming Model
//===----------------------------------------------------------------------===//

ir::value *dispatch::program_id(int axis, ir::builder *builder) {
  return builder->create_get_program_id(axis);
}

ir::value *dispatch::num_programs(int axis, ir::builder *builder) {
  return builder->create_get_num_programs(axis);
}

//===----------------------------------------------------------------------===//
//                               Implicit Casting Utilities
//===----------------------------------------------------------------------===//

ir::type *computation_type(ir::type* a_ty, ir::type* b_ty){
  context &ctx = a_ty->get_context();
  // 1) if one operand is double, the other is implicitly
  //    converted to double
  if(a_ty->is_double_ty() || b_ty->is_double_ty())
    return type::get_double_ty(ctx);
  // 2) if one operand is float, the other is implicitly
  //    converted to float
  if(a_ty->is_float_ty() || b_ty->is_float_ty())
    return type::get_float_ty(ctx);
  // 3 ) if one operand is half, the other is implicitly
  //     converted to half
  if(a_ty->is_half_ty() || b_ty->is_half_ty())
    return type::get_half_ty(ctx);
  if(!a_ty->is_integer_ty() || !b_ty->is_integer_ty())
    throw_unreachable("augment_types");
  // 4 ) both operands are integer and undergo
  //    integer promotion
  int a_rank = a_ty->get_integer_bitwidth();
  int b_rank = b_ty->get_integer_bitwidth();
  return a_rank > b_rank ? a_ty : b_ty;
}

//===----------------------------------------------------------------------===//
//                               Binary Operators
//===----------------------------------------------------------------------===//

void throw_incompatible_types(ir::type* type_a, ir::type* type_b) {
  throw semantic_error("invalid operands of type " + type_a->repr() + " and " + type_b->repr());
}

void check_ptr_type(ir::type* type_a, ir::type* type_b, bool allow_ptr_a){

  if(type_a->is_pointer_ty()){
    if(!allow_ptr_a)
      throw_incompatible_types(type_a, type_b);
    // T* + U* with T != U
    if(type_b->is_pointer_ty() && (type_a != type_b))
      throw_incompatible_types(type_a, type_b);
    // T* + float
    if(type_b->is_floating_point_ty())
      throw_incompatible_types(type_a, type_b);
  }
}

void binary_op_type_checking(ir::value*& lhs, ir::value*& rhs, ir::builder* builder,
                             bool allow_lhs_ptr = false, bool allow_rhs_ptr = false,
                             bool arithmetic_check = true){
  // implicit broadcasting
  std::tie(lhs, rhs) = dispatch::broadcast(lhs, rhs, builder);
  // implicit typecasting
  ir::type *lhs_sca_ty = lhs->get_type()->get_scalar_ty();
  ir::type *rhs_sca_ty = rhs->get_type()->get_scalar_ty();
  check_ptr_type(lhs_sca_ty, rhs_sca_ty, allow_lhs_ptr);
  check_ptr_type(rhs_sca_ty, lhs_sca_ty, allow_rhs_ptr);
  if(arithmetic_check && !lhs_sca_ty->is_pointer_ty() && !rhs_sca_ty->is_pointer_ty()){
    ir::type *ret_sca_ty = computation_type(lhs_sca_ty, rhs_sca_ty);
    lhs = dispatch::cast(lhs, ret_sca_ty, builder);
    rhs = dispatch::cast(rhs, ret_sca_ty, builder);
  }
}

ir::value *dispatch::add(ir::value *input, ir::value *other, ir::builder *builder) {
  binary_op_type_checking(input, other, builder, true, true);
  ir::type *input_scalar_ty = input->get_type()->get_scalar_ty();
  ir::type *other_scalar_ty = other->get_type()->get_scalar_ty();
  // offset + ptr
  // ptr + offset
  if(other_scalar_ty->is_pointer_ty() && !input_scalar_ty->is_pointer_ty())
    std::swap(input, other);
  if (input_scalar_ty->is_pointer_ty())
    return builder->create_gep(input, {other});
  // float + float
  else if (input_scalar_ty->is_floating_point_ty())
    return builder->create_fadd(input, other);
  // int + int
  else if (input_scalar_ty->is_integer_ty())
    return builder->create_add(input, other);
  return throw_unreachable("add");
}

ir::value *dispatch::sub(ir::value *input, ir::value *other, ir::builder *builder) {
  binary_op_type_checking(input, other, builder, false, false);
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // ptr - offset
//  if (scalar_ty->is_pointer_ty())
//    return builder->create_gep(input, {other});
  // float + float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fsub(input, other);
  // int + int
  else if (scalar_ty->is_integer_ty())
    return builder->create_sub(input, other);
  return throw_unreachable("sub");
}

ir::value *dispatch::mul(ir::value *input, ir::value *other, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float * float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fmul(input, other);
  // int * int
  else if (scalar_ty->is_integer_ty())
    return builder->create_mul(input, other);
  return throw_unreachable("mul");
}

ir::value *dispatch::truediv(ir::value *input, ir::value *other, ir::builder *builder) {
  binary_op_type_checking(input, other, builder, false, false, false);
  ir::type *input_scalar_ty = input->get_type()->get_scalar_ty();
  ir::type *other_scalar_ty = other->get_type()->get_scalar_ty();
  // float / int
  if(input_scalar_ty->is_floating_point_ty() && other_scalar_ty->is_integer_ty())
    other = cast(other, input_scalar_ty, builder);
  // int / float
  else if(input_scalar_ty->is_integer_ty() && other_scalar_ty->is_floating_point_ty())
    input = cast(input, other_scalar_ty, builder);
  // int / int (cast to float32)
  else if(input_scalar_ty->is_integer_ty() && other_scalar_ty->is_integer_ty()){
    input = cast(input, builder->get_float_ty(), builder);
    other = cast(other, builder->get_float_ty(), builder);
  }
  // float / float (cast to highest exponent type)
  else if(input_scalar_ty->is_floating_point_ty() && other_scalar_ty->is_floating_point_ty()){
    if(input_scalar_ty->get_fp_mantissa_width() > other_scalar_ty->get_fp_mantissa_width())
      other = cast(other, input_scalar_ty, builder);
    else
      input = cast(input, other_scalar_ty, builder);
  }
  // unreachable
  else
    return throw_unreachable("div");
  return builder->create_fdiv(input, other);
}

ir::value *dispatch::mod(ir::value *input, ir::value *other, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float % int
  if (scalar_ty->is_floating_point_ty())
    return builder->create_frem(input, other);
  // int % int
  else if (scalar_ty->is_integer_ty())
    return builder->create_srem(input, other);
  return throw_unreachable("mod");
}


void bitwise_op_type_checking(ir::value *&input, ir::value *&other, ir::builder *builder, bool force_lhs_type = false){
  binary_op_type_checking(input, other, builder, false, false, false);
  ir::type *input_sca_ty = input->get_type()->get_scalar_ty();
  ir::type *other_sca_ty = other->get_type()->get_scalar_ty();
  if(!input_sca_ty->is_integer_ty() || !other_sca_ty->is_integer_ty())
    throw_incompatible_types(input_sca_ty, other_sca_ty);
  // for some reason pytorch assigns the result of binary op to have the type of the lhs...
  if(force_lhs_type){
    if(input_sca_ty->get_integer_bitwidth() != other_sca_ty->get_integer_bitwidth())
      other = dispatch::cast(other, input_sca_ty, builder);
  }
  else{
    if(input_sca_ty->get_integer_bitwidth() < other_sca_ty->get_integer_bitwidth())
      input = dispatch::cast(input, other_sca_ty, builder);
    else if(other_sca_ty->get_integer_bitwidth() < input_sca_ty->get_integer_bitwidth())
      other = dispatch::cast(other, input_sca_ty, builder);
  }

}

ir::value *dispatch::and_(ir::value *input, ir::value *other, ir::builder *builder) {
  bitwise_op_type_checking(input, other, builder, true);
  return builder->create_and(input, other);
}

ir::value *dispatch::or_(ir::value *input, ir::value *other, ir::builder *builder) {
  bitwise_op_type_checking(input, other, builder, true);
  return builder->create_or(input, other);
}


ir::value *dispatch::xor_(ir::value *input, ir::value *other, ir::builder *builder) {
  bitwise_op_type_checking(input, other, builder, true);
  return builder->create_xor(input, other);
}


ir::value *dispatch::lshr(ir::value *input, ir::value *other, ir::builder *builder) {
  bitwise_op_type_checking(input, other, builder, false);
  return builder->create_lshr(input, other);
}


ir::value *dispatch::shl(ir::value *input, ir::value *other, ir::builder *builder) {
  bitwise_op_type_checking(input, other, builder, false);
  return builder->create_shl(input, other);
}

//===----------------------------------------------------------------------===//
//                               Unary Operators
//===----------------------------------------------------------------------===//

ir::value *dispatch::plus(ir::value *input, ir::builder *) {
  return input;
}

ir::value *dispatch::minus(ir::value *input, ir::builder *builder) {
  ir::type* input_sca_ty = input->get_type()->get_scalar_ty();
  if(input_sca_ty->is_pointer_ty())
    throw semantic_error("wrong type argument to unary minus (" + input_sca_ty->repr() + ")");
  ir::value *_0 = ir::constant::get_null_value(input_sca_ty);
  return dispatch::sub(_0, input, builder);
}

ir::value *dispatch::invert(ir::value *input, ir::builder *builder) {
  ir::type* input_sca_ty = input->get_type()->get_scalar_ty();
  if(input_sca_ty->is_pointer_ty() || input_sca_ty->is_floating_point_ty())
    throw semantic_error("wrong type argument to unary invert (" + input_sca_ty->repr() + ")");
  ir::value *_1 = ir::constant::get_all_ones_value(input_sca_ty);
  return dispatch::xor_(input, _1, builder);
}


//===----------------------------------------------------------------------===//
//                               Comparison Operators
//===----------------------------------------------------------------------===//

ir::value *dispatch::greater_than(ir::value *input, ir::value *other, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float > float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpOGT(input, other);
  // int > int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpSGT(input, other);
  return throw_unreachable("greater_than");
}

ir::value *dispatch::greater_equal(ir::value *input, ir::value *other, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float >= float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpOGE(input, other);
  // int >= int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpSGE(input, other);
  return throw_unreachable("greater_equal");
}

ir::value *dispatch::less_than(ir::value *input, ir::value *other, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float < float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpOLT(input, other);
  // int < int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpSLT(input, other);
  return throw_unreachable("less_than");
}

ir::value *dispatch::less_equal(ir::value *input, ir::value *other, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float < float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpOLE(input, other);
  // int < int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpSLE(input, other);
  return throw_unreachable("less_equal");
}

ir::value *dispatch::equal(ir::value *input, ir::value *other, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float == float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpOEQ(input, other);
  // int == int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpEQ(input, other);
  return throw_unreachable("equal");
}

ir::value *dispatch::not_equal(ir::value *input, ir::value *other, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float == float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpONE(input, other);
  // int == int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpNE(input, other);
  return throw_unreachable("equal");
}

//===----------------------------------------------------------------------===//
//                               Block Creation
//===----------------------------------------------------------------------===//

ir::value* dispatch::arange(int start, int end, ir::builder *builder) {
  return builder->get_range(start, end);
}

ir::value* dispatch::zeros(shape_t shape, ir::type *dtype, ir::builder *builder) {
  ir::value *_0 = ir::constant::get_null_value(dtype);
  return builder->create_splat(_0, shape);
}

//===----------------------------------------------------------------------===//
//                               Shape Manipulation
//===----------------------------------------------------------------------===//


ir::value *dispatch::reshape(ir::value *input, shape_t dst_shape, ir::builder *builder) {
  unsigned numel = 1;
  for(unsigned s: dst_shape) numel *= s;
  if(input->get_type()->get_tile_num_elements() != numel)
    throw semantic_error("cannot reshape block of different shape");
  return builder->create_reshape(input, dst_shape);
}

ir::value *dispatch::broadcast(ir::value *input, shape_t shape, ir::builder *builder) {
  if (!input->get_type()->is_block_ty())
    return builder->create_splat(input, shape);
  auto src_shape = input->get_type()->get_block_shapes();
  if (src_shape.size() != shape.size())
    throw std::runtime_error("Cannot broadcast");
  return builder->create_broadcast(input, shape);
}

std::tuple<ir::value*, ir::value*> dispatch::broadcast(ir::value *lhs, ir::value* rhs, ir::builder *builder) {
  ir::type *lhs_ty = lhs->get_type();
  ir::type *rhs_ty = rhs->get_type();
  // make_shape_compatible(block, scalar)
  if (lhs_ty->is_block_ty() && !rhs_ty->is_block_ty())
    rhs = builder->create_splat(rhs, lhs_ty->get_block_shapes());
  // make_shape_compatible(scalar, block)
  else if (!lhs_ty->is_block_ty() && rhs_ty->is_block_ty())
    lhs = builder->create_splat(lhs, rhs_ty->get_block_shapes());
  // make_shape_compatible(block, block)
  else if (lhs_ty->is_block_ty() && rhs_ty->is_block_ty()) {
    auto lhs_shape = lhs_ty->get_block_shapes();
    auto rhs_shape = rhs_ty->get_block_shapes();
    if (lhs_shape.size() != rhs_shape.size())
      throw std::runtime_error("Cannot make_shape_compatible: blocks must have the same rank");
    ir::type::block_shapes_t ret_shape;
    for (size_t i = 0; i < lhs_shape.size(); ++i) {
      unsigned left = lhs_shape[i];
      unsigned right = rhs_shape[i];
      if (left == 1)
        ret_shape.push_back(right);
      else if (right == 1)
        ret_shape.push_back(left);
      else if (left == right)
        ret_shape.push_back(left);
      else
        throw std::runtime_error("Cannot make_shape_compatible: incompatible dimensions at index " + std::to_string(i) +
                                 ": " + std::to_string(left) + " and " + std::to_string(right));
    }
    if (lhs_shape != ret_shape)
      lhs = builder->create_broadcast(lhs, ret_shape);
    if (rhs_shape != ret_shape)
      rhs = builder->create_broadcast(rhs, ret_shape);
  }
  return std::make_tuple(lhs, rhs);
}

ir::value *dispatch::cast(ir::value *input, ir::type *dst_ty, ir::builder *builder) {
  ir::type *src_ty = input->get_type();
  if (src_ty->is_block_ty())
    dst_ty = ir::block_type::get(dst_ty, input->get_type()->get_block_shapes());
  if(src_ty == dst_ty)
    return input;
  ir::type *src_sca_ty = src_ty->get_scalar_ty();
  ir::type *dst_sca_ty = dst_ty->get_scalar_ty();
  // FP Truncation
  bool truncate_fp = src_sca_ty->is_floating_point_ty() &&
                     dst_sca_ty->is_floating_point_ty() &&
                     src_sca_ty->get_fp_mantissa_width() > dst_sca_ty->get_fp_mantissa_width();
  if (truncate_fp)
    return builder->create_fp_trunc(input, dst_ty);
  // FP Extension
  bool ext_fp = src_sca_ty->is_floating_point_ty() &&
                dst_sca_ty->is_floating_point_ty() &&
                src_sca_ty->get_fp_mantissa_width() < dst_sca_ty->get_fp_mantissa_width();
  if (ext_fp)
    return builder->create_fp_ext(input, dst_ty);
  // Int cast
  if (src_sca_ty->is_integer_ty() && dst_sca_ty->is_integer_ty() &&
      src_sca_ty->get_integer_bitwidth() != dst_sca_ty->get_integer_bitwidth())
    return builder->create_int_cast(input, dst_ty, true);
  // Float -> Int
  if (src_sca_ty->is_floating_point_ty() && dst_sca_ty->is_integer_ty())
    return builder->create_fp_to_si(input, dst_ty);
  // int -> Float
  if (src_sca_ty->is_integer_ty() && dst_sca_ty->is_floating_point_ty())
    return builder->create_si_to_fp(input, dst_ty);
  // Ptr -> Ptr
  if (src_sca_ty->is_pointer_ty() && dst_sca_ty->is_pointer_ty())
    return builder->create_cast(ir::BitCast, input, dst_ty);
  // * -> Bool
  if (dst_sca_ty->is_bool_ty()) {
    if (src_sca_ty->is_pointer_ty())
      input = cast(input, builder->get_int64_ty(), builder);
    ir::value *other = builder->get_int64(0);
    if (src_ty->is_bool_ty())
      other = builder->create_splat(other, src_ty->get_block_shapes());
    return builder->create_icmpNE(input, other);
  }
  return throw_unreachable("cast");
}

//===----------------------------------------------------------------------===//
//                               Memory Operators
//===----------------------------------------------------------------------===//

ir::value *dispatch::load(ir::value* ptr, ir::value* mask, ir::value* other, ir::builder* builder) {
  if (!mask && !other)
    return builder->create_load(ptr);
  if (!mask)
    throw std::runtime_error("`other` cannot be provided without `mask`");
  ir::type *elt_ty = ptr->get_type()->get_scalar_ty()->get_pointer_element_ty();
  auto shape = ptr->get_type()->get_block_shapes();
  if(!other)
    other = ir::undef_value::get(elt_ty);
  return builder->create_masked_load(ptr, mask, other);
}

ir::value *dispatch::store(ir::value* ptr, ir::value *val, ir::value* mask, ir::builder *builder) {
  val = dispatch::broadcast(val, ptr->get_type()->get_block_shapes(), builder);
  if(mask)
    mask = dispatch::broadcast(mask, ptr->get_type()->get_block_shapes(), builder);
  ir::type *ptr_ty = ptr->get_type();
  ir::type *val_ty = val->get_type();
  if(ptr_ty->get_scalar_ty()->get_pointer_element_ty() !=
     val_ty->get_scalar_ty())
    throw semantic_error("Cannot store " + val_ty->repr() + " into " + ptr_ty->repr() + "." +
                         "Please convert data explicitly before storing.");
  if (!mask)
    return builder->create_store(ptr, val);
  if(!mask->get_type()->get_scalar_ty()->is_bool_ty())
    throw semantic_error("Mask must have boolean scalar type");
  return builder->create_masked_store(ptr, val, mask);
}

ir::value *dispatch::atomic_cas(ir::value* ptr, ir::value *cmp, ir::value *val, ir::builder *builder){
  return builder->create_atomic_cas(ptr, cmp, val);
}

ir::value *dispatch::atomic_xchg(ir::value* ptr, ir::value *val, ir::builder *builder){
  return builder->create_atomic_exch(ptr, val);
}

//===----------------------------------------------------------------------===//
//                               Linear Algebra
//===----------------------------------------------------------------------===//

ir::value *dispatch::dot(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
  ir::value *_0 = builder->get_float32(0);
  unsigned M = lhs->get_type()->get_block_shapes()[0];
  unsigned N = rhs->get_type()->get_block_shapes()[1];
  _0 = builder->create_splat(_0, {M, N});
  return builder->create_dot(lhs, rhs, _0);
}


//===----------------------------------------------------------------------===//
//                               Indexing
//===----------------------------------------------------------------------===//

ir::value *dispatch::where(ir::value* condition, ir::value *x, ir::value *y, ir::builder *builder)
{ return builder->create_select(condition, x, y); }


//===----------------------------------------------------------------------===//
//                               Reductions
//===----------------------------------------------------------------------===//

ir::value *reduce_impl(ir::value *input, unsigned int axis, ir::builder *builder, const std::string &name,
                       ir::reduce_inst::op_t FLOAT_OP, ir::reduce_inst::op_t INT_OP) {
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  if (scalar_ty->is_floating_point_ty())
    return builder->create_reduce(input, FLOAT_OP, axis);
  else if (scalar_ty->is_integer_ty())
    return builder->create_reduce(input, INT_OP, axis);
  return throw_unreachable(name);
}

ir::value *dispatch::min(ir::value *input, unsigned int axis, ir::builder *builder) {
  return reduce_impl(input, axis, builder, "min", ir::reduce_inst::FMIN, ir::reduce_inst::MIN);
}

ir::value *dispatch::max(ir::value *input, unsigned int axis, ir::builder *builder) {
  return reduce_impl(input, axis, builder, "max", ir::reduce_inst::FMAX, ir::reduce_inst::MAX);
}

ir::value *dispatch::sum(ir::value *input, unsigned int axis, ir::builder *builder) {
  return reduce_impl(input, axis, builder, "sum", ir::reduce_inst::FADD, ir::reduce_inst::ADD);
}


//===----------------------------------------------------------------------===//
//                               Math
//===----------------------------------------------------------------------===//

ir::value *dispatch::exp(ir::value *x, ir::builder *builder) {
  return builder->create_exp(x);
}

ir::value *dispatch::log(ir::value *x, ir::builder *builder) {
  return builder->create_log(x);
}

ir::value *dispatch::sqrt(ir::value *x, ir::builder *builder) {
  return builder->create_sqrt(x);
}


}
}
