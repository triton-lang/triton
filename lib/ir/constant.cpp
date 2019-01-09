#include <cassert>
#include "ir/constant.h"
#include "ir/type.h"
#include "ir/context.h"
#include "ir/context_impl.h"

namespace tdl{
namespace ir{


// constant

constant *constant::get_null_value(type *ty) {
  context &ctx = ty->get_context();
  switch (ty->get_type_id()) {
  case type::IntegerTyID:
    return constant_int::get(ty, 0);
  case type::HalfTyID:
    return constant_fp::get(ctx, 0);
  case type::FloatTyID:
    return constant_fp::get(ctx, 0);
  case type::DoubleTyID:
    return constant_fp::get(ctx, 0);
  case type::X86_FP80TyID:
    return constant_fp::get(ctx, 0);
  case type::FP128TyID:
    return constant_fp::get(ctx, 0);
  case type::PPC_FP128TyID:
    return constant_fp::get(ctx, 0);
  default:
    throw std::runtime_error("Cannot create a null constant of that type!");
  }
}

// FIXME

constant *constant::get_all_ones_value(type *ty) {
  if(ty->is_integer_ty())
    return constant_int::get(ty, 0xFFFFFFFF);
  if(ty->is_floating_point_ty())
    return constant_fp::get(ty->get_context(), 0xFFFFFFFF);
  throw std::runtime_error("Cannot create all ones value for that type!");
}

// constant_int
// FIXME use something like APInt

constant_int::constant_int(type *ty, uint64_t value)
  : constant(ty, 0), value_(value){ }

constant *constant_int::get(type *ty, uint64_t value) {
  return new constant_int(ty, value);
}

// constant_range
// FIXME use something like APInt

constant_range::constant_range(type *ty, uint64_t first, uint64_t last)
  : constant(ty, 0), first_(first), last_(last){ }

constant *constant_range::get(constant *first, constant *last) {
  assert(first->get_type()->is_integer_ty());
  assert(first->get_type() == last->get_type());
  uint64_t vfirst = ((constant_int*)first)->get_value();
  uint64_t vlast = ((constant_int*)first)->get_value();
  return new constant_range(first->get_type(), vfirst, vlast);
}


// constant_fp
// FIXME use something like APFloat

constant_fp::constant_fp(context &ctx, double value)
  : constant(type::get_float_ty(ctx), 0), value_(value){ }

constant *constant_fp::get_negative_zero(type *ty){
  double neg_zero = 0;
  return get(ty->get_context(), neg_zero);
}

constant *constant_fp::get_zero_value_for_negation(type *ty) {
  if(ty->get_scalar_ty()->is_floating_point_ty())
    return get_negative_zero(ty);
  return constant::get_null_value(ty);
}

constant *constant_fp::get(context &ctx, double v){
  context_impl *impl = ctx.p_impl.get();
  constant_fp *&result = impl->fp_constants_[v];
  if(!result)
    result = new constant_fp(ctx, v);
  return result;
}

// undef value
undef_value::undef_value(type *ty)
  : constant(ty, 0) { }

undef_value *undef_value::get(type *ty) {
  context_impl *impl = ty->get_context().p_impl.get();
  undef_value *&result = impl->uv_constants_[ty];
  if(!result)
    result = new undef_value(ty);
  return result;
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


}
}
