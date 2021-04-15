#include "triton/ir/dispatch.h"

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
//                               Binary Operators
//===----------------------------------------------------------------------===//

ir::value *dispatch::add(ir::value *input, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // ptr + offset
  if (scalar_ty->is_pointer_ty())
    return builder->create_gep(input, {other});
  // float + float
  else if (scalar_ty->is_floating_point_ty())
    return builder->create_fadd(input, other);
  // int + int
  else if (scalar_ty->is_integer_ty())
    return builder->create_add(input, other);
  return throw_unreachable("add");
}

ir::value *dispatch::sub(ir::value *input, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // ptr + offset
  if (scalar_ty->is_pointer_ty())
    return builder->create_gep(input, {other});
  // float + float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fsub(input, other);
  // int + int
  else if (scalar_ty->is_integer_ty())
    return builder->create_sub(input, other);
  return throw_unreachable("sub");
}

ir::value *dispatch::mul(ir::value *input, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float * float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fmul(input, other);
  // int * int
  else if (scalar_ty->is_integer_ty())
    return builder->create_mul(input, other);
  return throw_unreachable("mul");
}

ir::value *dispatch::div(ir::value *input, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float / float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fdiv(input, other);
  // int / int
  else if (scalar_ty->is_integer_ty())
    return builder->create_sdiv(input, other);
  return throw_unreachable("div");
}

ir::value *dispatch::mod(ir::value *input, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float % int
  if (scalar_ty->is_floating_point_ty())
    return builder->create_frem(input, other);
  // int % int
  else if (scalar_ty->is_integer_ty())
    return builder->create_srem(input, other);
  return throw_unreachable("mod");
}

ir::value *dispatch::and_(ir::value *input, ir::value *other, ir::builder *builder) {
  return builder->create_and(input, other);
}


//===----------------------------------------------------------------------===//
//                               Comparison Operators
//===----------------------------------------------------------------------===//

ir::value *dispatch::greater_than(ir::value *input, ir::value *other, ir::builder *builder) {
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
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float == float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpOEQ(input, other);
  // int == int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpEQ(input, other);
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


ir::value *dispatch::reshape(ir::value *input, shape_t shape, ir::builder *builder) {
  return builder->create_reshape(input, shape);
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
  if (!mask)
    return builder->create_store(ptr, val);
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
