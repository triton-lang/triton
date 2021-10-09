#include "triton/ir/builder.h"
#include <functional>
#include <iostream>
#include <pybind11/pybind11.h>

namespace ir = triton::ir;
namespace py = pybind11;

static const std::string _builder_doc = R"pbdoc(
  :param builder: IR builder to generate code into, optional, set automatically when called inside a @triton.jit function
  :type builder: triton.ir.builder
)pbdoc";

#define VA_ARGS(...) , ##__VA_ARGS__
#define DEF_FUNC(MOD, PY_NAME, C_FUNC, ...)                          \
  MOD.def(PY_NAME, C_FUNC, (C_FUNC##_docstr + _builder_doc).c_str(), \
          ret::reference VA_ARGS(__VA_ARGS__), "builder"_a)

void throw_not_implemented(std::string key) {
  throw std::runtime_error("Encountered unimplemented code path in `" + key + "`. This is likely a bug on our side.");
}

void throw_not_int_or_float(std::string key) {
  throw std::runtime_error("`" + key + "` only supported for integer and floating point types.");
}

enum type_code {
  _bool,
  int8,
  int16,
  int32,
  int64,
  float16,
  float32,
  float64
};

ir::type *make_ir(type_code ty, ir::builder *builder) {
  switch (ty) {
  case float16:
    return builder->get_half_ty();
  case float32:
    return builder->get_float_ty();
  default:
    throw_not_implemented("make_ir");
  }
}

type_code from_ir(ir::type *ty) {
  if (ty->is_half_ty())
    return float16;
  if (ty->is_float_ty())
    return float32;
  throw_not_implemented("from_ir");
}

/*----------------------------------------------
 definition of triton.cast / triton.ir.value.to
 ----------------------------------------------*/
std::string cast_docstr = R"pbdoc(
  Tries to cast a block to a new data type.

  :param input: The input block.
  :type input: triton.ir.value
)pbdoc";

ir::value *cast(ir::value *input, type_code _dtype, ir::builder *builder) {
  ir::type *src_ty = input->get_type();
  ir::type *dst_ty = make_ir(_dtype, builder);
  if (src_ty->is_block_ty())
    dst_ty = ir::block_type::get(dst_ty, input->get_type()->get_block_shapes());
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
      input = cast(input, int64, builder);
    ir::value *other = builder->get_int64(0);
    if (src_ty->is_bool_ty())
      other = builder->create_splat(other, src_ty->get_block_shapes());
    return builder->create_icmpNE(input, other);
  }
  throw_not_implemented("cast");
}

/*----------------------------------------------
 definition of triton.broadcast_check
 ----------------------------------------------*/
std::string try_broadcast_docstr = R"pbdoc(
    Tries to broadcast two blocks to a common compatible shape.

    :param input: The first input block.
    :type input: triton.ir.value
    :param other: The second input block.
    :type other: triton.ir.value
)pbdoc";

std::tuple<ir::value *, ir::value *> try_broadcast(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
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

/*----------------------------------------------
 definition of triton.broadcast_to
 ----------------------------------------------*/
std::string broadcast_to_docstr = R"pbdoc(
    Tries to broadcast a block to a new shape.

    :param input: The input block.
    :type input: triton.value
    :param shape: The new shape.
    :type shape: tuple of int
)pbdoc";

ir::value *broadcast_to(ir::value *input, const ir::type::block_shapes_t &shape, ir::builder *builder) {
  if (!input->get_type()->is_block_ty())
    return builder->create_splat(input, shape);
  auto src_shape = input->get_type()->get_block_shapes();
  if (src_shape.size() != shape.size())
    throw std::runtime_error("Cannot broadcast");
  return builder->create_broadcast(input, shape);
}

/*----------------------------------------------
 definition of triton.load
 ----------------------------------------------*/
std::string load_docstr = R"pbdoc(
    Return a block of data whose values are, elementwise, loaded from memory at location defined by `pointer`.

    :param pointer: Pointer to the data to be loaded.
    :type pointer: Block of triton.pointer
    :param mask: if mask[idx] is false, do not load the data at `pointer[idx]`.
    :type mask: Block of triton.bool, optional
    :param other: if mask[idx] is false, return other[idx] instead of 'pointer[idx]`
    :type other: Block of triton.value, optional
  )pbdoc";

ir::value *load(ir::value *pointer, std::optional<ir::value *> _mask, std::optional<ir::value *> _other, ir::builder *builder) {
  if (!_mask.has_value() && !_other.has_value())
    return builder->create_load(pointer);
  if (!_mask.has_value())
    throw std::runtime_error("`other` cannot be provided without `mask`");
  ir::value *mask = _mask.value();
  ir::type *elt_ty = pointer->get_type()->get_scalar_ty()->get_pointer_element_ty();
  auto shape = pointer->get_type()->get_block_shapes();
  ir::value *other = _other.has_value() ? _other.value() : ir::undef_value::get(elt_ty);
  other = cast(other, from_ir(elt_ty), builder);
  other = broadcast_to(other, shape, builder);
  mask = broadcast_to(mask, shape, builder);
  return builder->create_masked_load(pointer, mask, other);
}

/*----------------------------------------------
 definition of triton.store
 ----------------------------------------------*/
std::string store_docstr = R"pbdoc(
    Stores `value` block of elements in memory, element-wise, at the memory locations specified by `pointer`. 

    :param pointer: The memory locations where the elements of `value` are stored.
    :type pointer: Block of triton.pointer
    :param value: The block of elements to be stored.
    :type value: Block of triton.value
    :param mask: If mask[idx] is false, do not store `value[idx]` at `pointer[idx]`.
    :type mask: Block of triton.bool, optional
  )pbdoc";
ir::value *store(ir::value *ptr, ir::value *val, std::optional<ir::value *> _mask, ir::builder *builder) {
  if (!_mask.has_value())
    return builder->create_store(ptr, val);
  ir::value *mask = _mask.value();
  return builder->create_masked_store(ptr, val, mask);
}

/*----------------------------------------------
 definition of triton.dot
 ----------------------------------------------*/
std::string dot_docstr = R"pbdoc(
    Returns the matrix product of two blocks.
    The two blocks must be two dimensionals and have compatible inner dimensions.

    :param input: The first block to be multiplied.
    :type input: 2D block of scalar-type in {`float16`, `float32`}
    :param other: The second block to be multiplied.
    :type other: 2D block of scalar-type in {`float16`, `float32`}
  )pbdoc";
ir::value *dot(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
  ir::value *_0 = builder->get_float32(0);
  unsigned M = lhs->get_type()->get_block_shapes()[0];
  unsigned N = rhs->get_type()->get_block_shapes()[1];
  _0 = builder->create_splat(_0, {M, N});
  return builder->create_dot(lhs, rhs, _0);
}

/*----------------------------------------------
 definition of triton.where
 ----------------------------------------------*/
std::string where_docstr = R"pbdoc(
    Returns a block of elements from either `x` or `y`, depending on `condition`.
    Note that `x` and `y` are always evaluated regardless of the value of `condition`.
    If you want to avoid unintented memory operations, use the `mask` arguments in `triton.load` and `triton.store` instead.

    :param condition: When True (nonzero), yield x, otherwise yield y.
    :type condition: Block of triton.bool
    :param x: values selected at indices where condition is True.
    :param y: values selected at indices where condition is False.
  )pbdoc";
ir::value *where(ir::value *condition, ir::value *x, ir::value *y, ir::builder *builder) {
  return builder->create_select(condition, x, y);
};

/*----------------------------------------------
 definition of triton.arange
 ----------------------------------------------*/
std::string arange_docstr = R"pbdoc(
    Returns contiguous values within the open interval [start, end).

    :param start: Start of the interval.
    :type start: int
    :param stop: End of the interval.
    :type stop: int
  )pbdoc";
ir::value *arange(int start, int end, ir::builder *builder) {
  return builder->get_range(start, end);
};

/*----------------------------------------------
 definition of triton.program_id
 ----------------------------------------------*/
std::string program_id_docstr = R"pbdoc(
    Returns the id of the current program instance along the given `axis`.
    Triton uses an SPMD model in which different @triton.jit functions run in parallel with different `program_id`s.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
  )pbdoc";
ir::value *program_id(int axis, ir::builder *builder) {
  return builder->create_get_program_id(axis);
};

/*----------------------------------------------
 definition of triton.num_programs
 ----------------------------------------------*/
std::string num_programs_docstr = R"pbdoc(
    Returns the number of program instances launched along the given `axis`.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
  )pbdoc";
ir::value *num_programs(int axis, ir::builder *builder) {
  return builder->create_get_num_programs(axis);
};

/*----------------------------------------------
 definition of triton.zeros
 ----------------------------------------------*/
std::string zeros_docstr = R"pbdoc(
    Returns a block filled with the scalar value 0 and the given shape.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., tl.float16
    :type dtype: triton.ir.dtype
  )pbdoc";
ir::value *zeros(ir::type::block_shapes_t shape, type_code _dtype, ir::builder *builder) {
  ir::type *dtype = make_ir(_dtype, builder);
  ir::value *_0 = ir::constant::get_null_value(dtype);
  return builder->create_splat(_0, shape);
};

/*----------------------------------------------
 definition of triton.exp
 ----------------------------------------------*/
std::string _exp_docstr = R"pbdoc(
    Returns the element-wise exponential of `input`.
 )pbdoc";
ir::value *_exp(ir::value *input, ir::builder *builder) {
  return builder->create_exp(input);
};

/*----------------------------------------------
 definition of triton.log
 ----------------------------------------------*/
std::string _log_docstr = R"pbdoc(
    Returns the element-wise natural logarithm of `input`.
 )pbdoc";
ir::value *_log(ir::value *input, ir::builder *builder) {
  return builder->create_log(input);
};

/*----------------------------------------------
 definition of triton.sqrt
 ----------------------------------------------*/
std::string sqrt_docstr = R"pbdoc(
    Returns the element-wise square root of `input`.
 )pbdoc";
ir::value *sqrt(ir::value *input, ir::builder *builder) {
  return builder->create_sqrt(input);
};

/*----------------------------------------------
 definition of triton.min
 ----------------------------------------------*/
ir::value *reduce_impl(ir::value *input, unsigned int axis, ir::builder *builder, const std::string &name,
                       ir::reduce_inst::op_t FLOAT_OP, ir::reduce_inst::op_t INT_OP) {
  ir::type *scalar_ty = input->get_type()->get_scalar_ty();
  if (scalar_ty->is_floating_point_ty())
    return builder->create_reduce(input, FLOAT_OP, axis);
  else if (scalar_ty->is_integer_ty())
    return builder->create_reduce(input, INT_OP, axis);
  else
    throw_not_int_or_float(name);
}

std::string min_docstr = R"pbdoc(
    Returns the minimum value of `input`.
 )pbdoc";
ir::value *min(ir::value *input, unsigned int axis, ir::builder *builder) {
  return reduce_impl(input, axis, builder, "min", ir::reduce_inst::FMIN, ir::reduce_inst::MIN);
};

/*----------------------------------------------
 definition of triton.max
 ----------------------------------------------*/
std::string max_docstr = R"pbdoc(
    Returns the maximum value of `input`.
 )pbdoc";
ir::value *max(ir::value *input, unsigned int axis, ir::builder *builder) {
  return reduce_impl(input, axis, builder, "max", ir::reduce_inst::FMAX, ir::reduce_inst::MAX);
};

/*----------------------------------------------
 definition of triton.sum
 ----------------------------------------------*/
std::string sum_docstr = R"pbdoc(
    Returns the sum of `input`.
 )pbdoc";
ir::value *sum(ir::value *input, unsigned int axis, ir::builder *builder) {
  return reduce_impl(input, axis, builder, "sum", ir::reduce_inst::FADD, ir::reduce_inst::ADD);
};

/*----------------------------------------------
 definition of triton.atomic_cas
 ----------------------------------------------*/
std::string atomic_cas_docstr = R"pbdoc(
    Atomic compare-and-swap.
 )pbdoc";
ir::value *atomic_cas(ir::value *ptr, ir::value *cmp, ir::value *val, ir::builder *builder) {
  return builder->create_atomic_cas(ptr, cmp, val);
};

/*----------------------------------------------
 definition of triton.atomic_xchg
 ----------------------------------------------*/
std::string atomic_xchg_docstr = R"pbdoc(
    Atomic exchange.
 )pbdoc";
ir::value *atomic_xchg(ir::value *ptr, ir::value *val, ir::builder *builder) {
  return builder->create_atomic_exch(ptr, val);
};

/*----------------------------------------------
 debug barrier
 ----------------------------------------------*/
std::string debug_barrier_docstr = R"pbdoc(
   Temporary hacky fixup for when the compiler forgets to insert sync barriers
)pbdoc";
ir::value *debug_barrier(ir::builder *builder) {
  return builder->create_barrier();
}

#define DEF_BINARY_OP(MOD, PY_NAME, C_FUNC, ...)                                \
  MOD.def(PY_NAME, binary_op(C_FUNC), (C_FUNC##_docstr + _builder_doc).c_str(), \
          ret::reference VA_ARGS(__VA_ARGS__), "builder"_a)

template <class FN>
std::function<ir::value *(ir::value *, ir::value *, ir::builder *builder)>
binary_op(const FN &fn) {
  auto ret = [&fn](ir::value *self, ir::value *other, ir::builder *builder) {
    //std::tie(self, other) = try_broadcast(self, other, builder);
    return fn(self, other, builder);
  };
  return ret;
}

/*----------------------------------------------
 definition of self + other
 ----------------------------------------------*/
std::string add_docstr = R"pbdoc(
    Returns self + other, element-wise.
)pbdoc";
ir::value *add(ir::value *self, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = self->get_type()->get_scalar_ty();
  // ptr + offset
  if (scalar_ty->is_pointer_ty())
    return builder->create_gep(self, {other});
  // float + float
  else if (scalar_ty->is_floating_point_ty())
    return builder->create_fadd(self, other);
  // int + int
  else if (scalar_ty->is_integer_ty())
    return builder->create_add(self, other);
  throw_not_implemented("add");
}

/*----------------------------------------------
 definition of self - other
 ----------------------------------------------*/
std::string sub_docstr = R"pbdoc(
    Returns self - other, element-wise.
)pbdoc";
ir::value *sub(ir::value *self, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = self->get_type()->get_scalar_ty();
  // ptr + offset
  if (scalar_ty->is_pointer_ty())
    return builder->create_gep(self, {other});
  // float + float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fsub(self, other);
  // int + int
  else if (scalar_ty->is_integer_ty())
    return builder->create_sub(self, other);
  throw_not_implemented("sub");
}

/*----------------------------------------------
 definition of self * other
 ----------------------------------------------*/
std::string mul_docstr = R"pbdoc(
    Returns self * other, element-wise.
)pbdoc";
ir::value *mul(ir::value *self, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = self->get_type()->get_scalar_ty();
  // float * float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fmul(self, other);
  // int * int
  else if (scalar_ty->is_integer_ty())
    return builder->create_mul(self, other);
  throw_not_implemented("mul");
}

/*----------------------------------------------
 definition of self > other
 ----------------------------------------------*/
std::string greater_than_docstr = R"pbdoc(
    Returns self > other, element-wise.
)pbdoc";
ir::value *greater_than(ir::value *self, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = self->get_type()->get_scalar_ty();
  // float > float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpOGT(self, other);
  // int > int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpSGT(self, other);
  throw_not_implemented("greater_than");
}

/*----------------------------------------------
 definition of self >= other
 ----------------------------------------------*/
std::string greater_equal_docstr = R"pbdoc(
    Returns self >= other, element-wise.
)pbdoc";
ir::value *greater_equal(ir::value *self, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = self->get_type()->get_scalar_ty();
  // float >= float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpOGE(self, other);
  // int >= int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpSGE(self, other);
  throw_not_implemented("greater_equal");
}

/*----------------------------------------------
 definition of self < other
 ----------------------------------------------*/
std::string less_than_docstr = R"pbdoc(
    Returns self < other, element-wise.
)pbdoc";
ir::value *less_than(ir::value *self, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = self->get_type()->get_scalar_ty();
  // float < float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpOLT(self, other);
  // int < int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpSLT(self, other);
  throw_not_implemented("less_than");
}

/*----------------------------------------------
 definition of self <= other
 ----------------------------------------------*/
std::string less_equal_docstr = R"pbdoc(
    Returns self <= other, element-wise.
)pbdoc";
ir::value *less_equal(ir::value *self, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = self->get_type()->get_scalar_ty();
  // float < float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpOLE(self, other);
  // int < int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpSLE(self, other);
  throw_not_implemented("less_equal");
}

/*----------------------------------------------
 definition of self == other
 ----------------------------------------------*/
std::string equal_docstr = R"pbdoc(
    Returns self == other, element-wise.
)pbdoc";
ir::value *equal(ir::value *self, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = self->get_type()->get_scalar_ty();
  // float == float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpOEQ(self, other);
  // int == int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpEQ(self, other);
  throw_not_implemented("equal");
}

/*----------------------------------------------
 definition of self / other
 ----------------------------------------------*/
std::string _div_docstr = R"pbdoc(
    Returns self / other, element-wise.
)pbdoc";
ir::value *_div(ir::value *self, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = self->get_type()->get_scalar_ty();
  // float / float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fdiv(self, other);
  // int / int
  else if (scalar_ty->is_integer_ty())
    return builder->create_sdiv(self, other);
  throw_not_implemented("div");
}

/*----------------------------------------------
 definition of self % other
 ----------------------------------------------*/
std::string mod_docstr = R"pbdoc(
    Returns self % other, element-wise.
)pbdoc";
ir::value *mod(ir::value *self, ir::value *other, ir::builder *builder) {
  ir::type *scalar_ty = self->get_type()->get_scalar_ty();
  // float % int
  if (scalar_ty->is_floating_point_ty())
    return builder->create_frem(self, other);
  // int % int
  else if (scalar_ty->is_integer_ty())
    return builder->create_srem(self, other);
  throw_not_implemented("mod");
}

/*----------------------------------------------
 definition of self & other
 ----------------------------------------------*/
std::string _and_docstr = R"pbdoc(
    Returns self & other, element-wise.
)pbdoc";
ir::value *_and(ir::value *self, ir::value *other, ir::builder *builder) {
  return builder->create_and(self, other);
}

/*----------------------------------------------
 definition of minimum(self, other)
 ----------------------------------------------*/
std::string minimum_docstr = R"pbdoc(
    Returns element-wise minimum of self and other
)pbdoc";
ir::value *minimum(ir::value *self, ir::value *other, ir::builder *builder) {
  return where(less_than(self, other, builder), self, other, builder);
}

/*----------------------------------------------
 definition of self[slices]
 ----------------------------------------------*/

enum slice_mode_t {
  NEWAXIS,
  ALL
};

std::string subscript_docstr = R"pbdoc(
    returns self[slices].

    :param slices: The slices to subscript with.
    :type slices: List of `None` or `:` slices.
)pbdoc";
ir::value *subscript(ir::value *self, std::vector<py::object> slices, ir::builder *builder) {
  std::vector<slice_mode_t> modes;
  for (py::object slice : slices) {
    py::object none = py::none();
    py::object all = py::make_tuple(none, none, none);
    if (slice.is(none))
      modes.push_back(NEWAXIS);
    else if (all.attr("__eq__")(slice))
      modes.push_back(ALL);
    else
      throw std::runtime_error("slice must be None or (None, None, None)");
  }

  ir::type::block_shapes_t shape;
  size_t curr = 0;
  for (slice_mode_t mode : modes) {
    if (mode == NEWAXIS)
      shape.push_back(1);
    else {
      assert(mode == ALL);
      shape.push_back(self->get_type()->get_block_shapes()[curr++]);
    }
  }
  return builder->create_reshape(self, shape);
}
