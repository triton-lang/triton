#include <cassert>
#include <stdexcept>
#include "triton/ir/constant.h"
#include "triton/ir/type.h"
#include "triton/ir/context.h"
#include "triton/ir/context_impl.h"

namespace triton{
namespace ir{


// constant

constant *constant::get_null_value(type *ty) {
  context &ctx = ty->get_context();
  switch (ty->get_scalar_ty()->get_type_id()) {
  case type::IntegerTyID:
    return constant_int::get(ty, 0);
  case type::FP16TyID:
    return constant_fp::get(type::get_fp16_ty(ctx), 0);
  case type::BF16TyID:
    return constant_fp::get(type::get_bf16_ty(ctx), 0);
  case type::FP32TyID:
    return constant_fp::get(type::get_fp32_ty(ctx), 0);
  case type::FP64TyID:
    return constant_fp::get(type::get_fp64_ty(ctx), 0);
  default:
    throw std::runtime_error("Cannot create a null constant of that type!");
  }
}

// FIXME

constant *constant::get_all_ones_value(type *ty) {
  if(ty->is_integer_ty())
    return constant_int::get(ty, 0xFFFFFFFFFFFFFFFF);
  if(ty->is_floating_point_ty())
    return constant_fp::get(ty, 0xFFFFFFFFFFFFFFFF);
  throw std::runtime_error("Cannot create all ones value for that type!");
}

// constant_int
// FIXME use something like APInt

constant_int::constant_int(type *ty, uint64_t value)
  : constant(ty, 0), value_(value){ }

constant_int *constant_int::get(type *ty, uint64_t value) {
  if (!ty->is_integer_ty())
    throw std::runtime_error("Cannot create constant_int with non integer ty");
  context_impl *impl = ty->get_context().p_impl.get();
  std::unique_ptr<constant_int> &cst = impl->int_constants_[std::make_pair(ty, value)];
  if(!cst)
    cst.reset(new constant_int(ty, value));
  return cst.get();
}


// constant_fp
// FIXME use something like APFloat

constant_fp::constant_fp(type *ty, double value)
  : constant(ty, 0), value_(value){ }

constant *constant_fp::get_negative_zero(type *ty){
  double neg_zero = 0;
  return get(ty, neg_zero);
}

constant *constant_fp::get_zero_value_for_negation(type *ty) {
  if(ty->get_scalar_ty()->is_floating_point_ty())
    return constant_fp::get(ty, 0);
  return constant::get_null_value(ty);
}

constant *constant_fp::get(type *ty, double v){
  context_impl *impl = ty->get_context().p_impl.get();
  std::unique_ptr<constant_fp> &result = impl->fp_constants_[std::make_pair(ty, v)];
  if(!result)
    result.reset(new constant_fp(ty, v));
  return result.get();
}


// undef value
undef_value::undef_value(type *ty)
  : constant(ty, 0) { }

undef_value *undef_value::get(type *ty) {
  context_impl *impl = ty->get_context().p_impl.get();
  std::unique_ptr<undef_value> &result = impl->uv_constants_[ty];
  if(!result)
    result.reset(new undef_value(ty));
  return result.get();
}

/* global value */
global_value::global_value(type *ty, unsigned num_ops,
                           linkage_types_t linkage,
                           const std::string &name, unsigned addr_space)
    : constant(pointer_type::get(ty, addr_space), num_ops, name),
      linkage_(linkage) { }


/* global object */
global_object::global_object(type *ty, unsigned num_ops,
                            linkage_types_t linkage,
                            const std::string &name, unsigned addr_space)
  : global_value(ty, num_ops, linkage, name, addr_space) { }


/* alloc const */
alloc_const::alloc_const(type *ty, constant_int *size, const std::string &name)
  : global_object(ty, 1, global_value::external, name, 4) {
  set_operand(0, size);
}


}
}
