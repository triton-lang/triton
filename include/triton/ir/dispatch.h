#pragma once

#ifndef _TRITON_IR_DISPATCH_H_
#define _TRITON_IR_DISPATCH_H_

#include "triton/ir/builder.h"

namespace triton{
namespace ir{


/*----------------------------------------------
 higher level functions that follow the likely
 semantics of most expected frontends
 ----------------------------------------------*/

struct dispatch{
  typedef ir::type::block_shapes_t shape_t;


  // programming model
  static ir::value *program_id(int axis, ir::builder *builder);
  static ir::value *num_programs(int axis, ir::builder *builder);

  // binary operators
  static ir::value *add(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *sub(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *mul(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *div(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *mod(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *and_(ir::value *input, ir::value *other, ir::builder *builder);

  // comparison operators
  static ir::value *greater_than(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *greater_equal(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *less_than(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *less_equal(ir::value *input, ir::value *other, ir::builder *builder);
  static ir::value *equal(ir::value *input, ir::value *other, ir::builder *builder);

  // block creation
  static ir::value* arange(int start, int end, ir::builder *builder);
  static ir::value* zeros(shape_t shape, ir::type *dtype, ir::builder *builder);


  // shape manipulation
  static ir::value *reshape(ir::value *input, shape_t shape, ir::builder *builder);
  static ir::value *broadcast(ir::value *input, shape_t shape, ir::builder *builder);
  static std::tuple<ir::value*, ir::value*> broadcast(ir::value *lhs, ir::value* rhs, ir::builder *builder);


  // memory operators
  static ir::value *load(ir::value* ptr, ir::value* mask, ir::value* other, ir::builder *builder);
  static ir::value *store(ir::value* ptr, ir::value *value, ir::value *mask, ir::builder *builder);
  static ir::value *atomic_cas(ir::value* ptr, ir::value *cmp, ir::value *val, ir::builder *builder);
  static ir::value *atomic_xchg(ir::value* ptr, ir::value *val, ir::builder *builder);

  // linear algebra
  static ir::value *dot(ir::value *lhs, ir::value *rhs, ir::builder *builder);

  // indexing
  static ir::value *where(ir::value* condition, ir::value *x, ir::value *y, ir::builder *builder);

  // reduction
  static ir::value *min(ir::value *input, unsigned int axis, ir::builder *builder);
  static ir::value *max(ir::value *input, unsigned int axis, ir::builder *builder);
  static ir::value *sum(ir::value *input, unsigned int axis, ir::builder *builder);

  // math
  static ir::value *exp(ir::value *x, ir::builder *builder);
  static ir::value *log(ir::value *x, ir::builder *builder);
  static ir::value *sqrt(ir::value *x, ir::builder *builder);
};

}
}

#endif
