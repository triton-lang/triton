#include <cassert>
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
  case type::HalfTyID:
    return constant_fp::get(type::get_half_ty(ctx), 0);
  case type::FloatTyID:
    return constant_fp::get(type::get_float_ty(ctx), 0);
  case type::DoubleTyID:
    return constant_fp::get(type::get_double_ty(ctx), 0);
  default:
    throw std::runtime_error("Cannot create a null constant of that type!");
  }
}

// FIXME

constant *constant::get_all_ones_value(type *ty) {
  if(ty->is_integer_ty())
    return constant_int::get(ty, 0xFFFFFFFF);
  if(ty->is_floating_point_ty())
    return constant_fp::get(ty, 0xFFFFFFFF);
  throw std::runtime_error("Cannot create all ones value for that type!");
}

// constant_int
// FIXME use something like APInt

constant_int::constant_int(type *ty, uint64_t value)
  : constant(ty, 0), value_(value){ }

constant_int *constant_int::get(type *ty, uint64_t value) {
  context_impl *impl = ty->get_context().p_impl.get();
  constant_int *& cst = impl->int_constants_[std::make_pair(ty, value)];
  if(cst == nullptr)
    cst = new constant_int(ty, value);
  return cst;
}

// constant_range
// FIXME use something like APInt

//"[" + std::to_string(first->get_value()) + " ... " + std::to_string(ty->get_tile_shapes()[0]->get_value()) + "]"

constant_range::constant_range(type *ty, constant_int *first, constant_int *last)
  : constant(ty, 0), first_(first), last_(last){ }

constant *constant_range::get(constant_int *first, constant_int *last) {
  assert(first->get_type()->is_integer_ty());
  assert(first->get_type() == last->get_type());
  assert(((constant_int*)first)->get_value() == 0);
  type *ty = tile_type::get(first->get_type(), {(unsigned)last->get_value()});
  return new constant_range(ty, first, last);
}

const constant_int* constant_range::get_first() const {
  return first_;
}

const constant_int* constant_range::get_last() const {
  return last_;
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
    return get_negative_zero(ty);
  return constant::get_null_value(ty);
}

constant *constant_fp::get(type *ty, double v){
  context_impl *impl = ty->get_context().p_impl.get();
  constant_fp *&result = impl->fp_constants_[std::make_pair(ty, v)];
  if(!result)
    result = new constant_fp(ty, v);
  return result;
}

// metaparameter
metaparameter::metaparameter(type *ty, const std::vector<unsigned> &space)
  : constant_int(ty, 0), space_(space), has_value_(false){ }

metaparameter* metaparameter::create(context &ctx, type *ty, unsigned lo, unsigned hi) {
  context_impl *impl = ctx.p_impl.get();
  std::vector<unsigned> space;
  for(unsigned i = lo; i <= hi; i *= 2)
    space.push_back(i);
  metaparameter *result = new metaparameter(ty, space);
  impl->mp_constants_.push_back(result);
  return result;
}

metaparameter* metaparameter::create(context &ctx, type *ty, const std::vector<unsigned> &space) {
  context_impl *impl = ctx.p_impl.get();
  metaparameter *result = new metaparameter(ty, space);
  impl->mp_constants_.push_back(result);
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


/* alloc const */
alloc_const::alloc_const(type *ty, constant_int *size, const std::string &name)
  : global_object(ty, 1, global_value::external, name, 4) {
  set_operand(0, size);
}


}
}
