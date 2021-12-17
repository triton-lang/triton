#pragma once

#ifndef _TRITON_IR_DISPATCH_H_
#define _TRITON_IR_DISPATCH_H_

#include "triton/ir/builder.h"
#include <stdexcept>

namespace triton{
namespace ir{


/*----------------------------------------------
 higher level functions that follow the likely
 semantics of most expected frontends
 ----------------------------------------------*/

struct semantic_error: public std::runtime_error {
  semantic_error(const std::string& msg):
    std::runtime_error(msg) { }
};

struct dispatch{
  typedef ir::type::block_shapes_t shape_t;


  // programming model
  static ir::value *program_id(int axis, ir::builder *builder);
  static ir::value *num_programs(int axis, ir::builder *builder);

  // binary operators
  static ir::value *add(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *sub(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *mul(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *truediv(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *floordiv(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *mod(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *and_(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *or_(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *xor_(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *lshr(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *shl(ir::value *input, ir::value *other, ir::builder *builder);

  // unary operators
  static ir::value *plus(ir::value *input, ir::builder *builder);
  static ir::value *minus(ir::value *input, ir::builder *builder);
  static ir::value *invert(ir::value *input, ir::builder *builder);

  // comparison operators
  static ir::value *greater_than(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *greater_equal(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *less_than(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *less_equal(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *equal(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *not_equal(ir::value *input, ir::value *other, ir::builder *builder);

  // block creation
  static ir::value* arange(int start, int end, ir::builder *builder);
  static ir::value* zeros(shape_t shape, ir::type *dtype, ir::builder *builder);


  // casting ops
  static ir::value *reshape(ir::value *input, shape_t shape, ir::builder *builder);
  static ir::value *cat(ir::value *lhs, ir::value *rhs, ir::builder *builder);
  static ir::value *broadcast(ir::value *input, shape_t shape, ir::builder *builder);
  static std::tuple<ir::value*, ir::value*> broadcast(ir::value *lhs, ir::value* rhs, ir::builder *builder);
  static ir::value *bitcast(ir::value *input, ir::type *type, ir::builder *builder);
  static ir::value *cast(ir::value *input, ir::type *type, ir::builder *builder);

  // memory operators
  static ir::value *load(ir::value* ptr, ir::value* mask, ir::value* other, const std::string &cache, ir::builder *builder);
  static ir::value *store(ir::value* ptr, ir::value *value, ir::value *mask, ir::builder *builder);
  static ir::value *atomic_cas(ir::value* ptr, ir::value *cmp, ir::value *val, ir::builder *builder);
  static ir::value *atomic_add(ir::value* ptr, ir::value *val, ir::value *msk, ir::builder *builder);
  static ir::value *atomic_max(ir::value* ptr, ir::value *val, ir::value *msk, ir::builder *builder);
  static ir::value *atomic_min(ir::value* ptr, ir::value *val, ir::value *msk, ir::builder *builder);
  static ir::value *atomic_and(ir::value* ptr, ir::value *val, ir::value *msk, ir::builder *builder);
  static ir::value *atomic_or(ir::value* ptr, ir::value *val, ir::value *msk, ir::builder *builder);
  static ir::value *atomic_xor(ir::value* ptr, ir::value *val, ir::value *msk, ir::builder *builder);
  static ir::value *atomic_xchg(ir::value* ptr, ir::value *val, ir::value *msk, ir::builder *builder);

  // linear algebra
  static ir::value *dot(ir::value *lhs, ir::value *rhs, ir::builder *builder);

  // indexing
  static ir::value *where(ir::value* condition, ir::value *x, ir::value *y, ir::builder *builder);

  // reduction
  static ir::value *min(ir::value *input, unsigned int axis, ir::builder *builder);
  static ir::value *max(ir::value *input, unsigned int axis, ir::builder *builder);
  static ir::value *sum(ir::value *input, unsigned int axis, ir::builder *builder);
  static ir::value *xor_sum(ir::value *input, unsigned axis, ir::builder *builder);

  // math
  static ir::value *umulhi(ir::value *x, ir::value *y, ir::builder *builder);
  static ir::value *exp(ir::value *x, ir::builder *builder);
  static ir::value *log(ir::value *x, ir::builder *builder);
  static ir::value *cos(ir::value *x, ir::builder *builder);
  static ir::value *sin(ir::value *x, ir::builder *builder);
  static ir::value *sqrt(ir::value *x, ir::builder *builder);

  // internal (debug/optimization)
  static ir::value *multiple_of(ir::value *x, int value, ir::builder *builder);
  static ir::value *max_contiguous(ir::value *x, int value, ir::builder *builder);
  static ir::value *debug_barrier(ir::builder *builder);
};

}
}

#endif
