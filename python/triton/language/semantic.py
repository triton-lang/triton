from triton._C.libtriton.triton import ir



## Create custom exception that prints message "hello"
class IncompatibleTypeErrorimpl(Exception):
  def __init__(self, type_a, type_b):
    self.type_a = type_a
    self.type_b = type_b
    self.message = "invalid operands of type " + self.type_a.repr() + " and " + self.type_b.repr()
    super(IncompatibleTypeErrorimpl, self).__init__(self.message)


##===----------------------------------------------------------------------===##
##                              Programming Model
##===----------------------------------------------------------------------===##

def program_id(axis, builder):
  return builder.create_get_program_id(axis)

def num_programs(axis, builder):
  return builder.create_get_num_programs(axis)

#===----------------------------------------------------------------------===//
#                               Implicit Casting Utilities
#===----------------------------------------------------------------------===//

def integer_promote_impl(a_ty, b_ty):
  a_rank = a_ty.int_bitwidth
  b_rank = b_ty.int_bitwidth
  a_sn = a_ty.int_signedness
  b_sn = b_ty.int_signedness
  # Rules for signedness taken from "Usual arithmetic conversions" on
  # https://en.cppreference.com/w/c/language/conversion.
  if a_sn == b_sn:
    return a_ty if a_rank > b_rank else b_ty
  elif a_sn == ir.SIGNEDNESS.UNSIGNED:
    return a_ty if a_rank >= b_rank else b_ty
  elif b_sn == ir.SIGNEDNESS.UNSIGNED:
    return b_ty if b_rank >= a_rank else a_ty
  else:
    assert False
  

def computation_type_impl(a_ty, b_ty, div_or_mod):
  ctx = a_ty.context
  # 1) if one operand is double, the other is implicitly
  #    converted to double
  if a_ty.is_fp64() or b_ty.is_fp64():
    return ir.type.get_fp64(ctx)
  # 2) if one operand is float, the other is implicitly
  #    converted to float
  if a_ty.is_fp32() or b_ty.is_fp32():
    return ir.type.get_fp32(ctx)
  # 3 ) if one operand is half, the other is implicitly converted to half
  #     unless we're doing / or %, which do not exist natively in PTX for fp16.
  if a_ty.is_fp16() or b_ty.is_fp16():
    if div_or_mod:
      return ir.type.get_fp32(ctx)
    else:
      return ir.type.get_fp16(ctx)
  if not a_ty.is_int() or not b_ty.is_int():
    assert False
  # 4 ) both operands are integer and undergo
  #    integer promotion
  if div_or_mod and a_ty.int_signedness != b_ty.int_signedness:
    raise ValueError("Cannot use /, #, or % with " + a_ty.repr() + " and " + b_ty.repr() + " because they have different signedness;" 
                        "this is unlikely to result in a useful answer. Cast them to the same signedness.")
  return integer_promote_impl(a_ty, b_ty)

#===----------------------------------------------------------------------===//
#                               Binary Operators
#===----------------------------------------------------------------------===//

def check_ptr_type_impl(type_a, type_b, allow_ptr_a):
  if type_a.is_ptr():
    if not allow_ptr_a:
      raise IncompatibleTypeErrorimpl(type_a, type_b)
    # T* + U* with T != U
    if type_b.is_ptr() and (type_a != type_b):
      raise IncompatibleTypeErrorimpl(type_a, type_b)
    # T* + float
    if type_b.is_floating():
      raise IncompatibleTypeErrorimpl(type_a, type_b)
  

def binary_op_type_checking_impl(lhs,  rhs,  builder,
                            allow_lhs_ptr = False, allow_rhs_ptr = False,
                            arithmetic_check = True, div_or_mod = False):
  # implicit broadcasting
  lhs, rhs = broadcast_impl(lhs, rhs, builder)
  # implicit typecasting
  lhs_sca_ty = lhs.type.scalar
  rhs_sca_ty = rhs.type.scalar
  check_ptr_type_impl(lhs_sca_ty, rhs_sca_ty, allow_lhs_ptr)
  check_ptr_type_impl(rhs_sca_ty, lhs_sca_ty, allow_rhs_ptr)
  if arithmetic_check and not lhs_sca_ty.is_ptr() and not rhs_sca_ty.is_ptr():
    ret_sca_ty = computation_type_impl(lhs_sca_ty, rhs_sca_ty, div_or_mod)
    lhs = cast_impl(lhs, ret_sca_ty, builder)
    rhs = cast_impl(rhs, ret_sca_ty, builder)
  return lhs, rhs
  

def add(input, other, builder):
  input, other = binary_op_type_checking_impl(input, other, builder, True, True)
  input_scalar_ty = input.type.scalar
  other_scalar_ty = other.type.scalar
  # offset + ptr
  # ptr + offset
  if other_scalar_ty.is_ptr() and not input_scalar_ty.is_ptr():
    input, other = other, input
  if input_scalar_ty.is_ptr():
    return builder.create_gep(input, [other])
  # float + float
  elif input_scalar_ty.is_floating():
    return builder.create_fadd(input, other)
  # int + int
  elif input_scalar_ty.is_int():
    return builder.create_add(input, other)
  assert False

def sub(input, other, builder):
  input, other = binary_op_type_checking_impl(input, other, builder, True, False)
  scalar_ty = input.type.scalar
  # ptr - offset
  if scalar_ty.is_ptr():
    return builder.create_gep(input, [minus(other, builder)])
  # float + float
  if scalar_ty.is_floating():
    return builder.create_fsub(input, other)
  # + int
  elif scalar_ty.is_int():
    return builder.create_sub(input, other)
  assert False

def mul(input, other, builder):
  input, other = binary_op_type_checking_impl(input, other, builder)
  scalar_ty = input.type.scalar
  # float * float
  if scalar_ty.is_floating():
    return builder.create_fmul(input, other)
  # * int
  elif scalar_ty.is_int():
    return builder.create_mul(input, other)
  assert False

def truediv(input, other, builder):
  input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
  input_scalar_ty = input.type.scalar
  other_scalar_ty = other.type.scalar
  # float / int
  if input_scalar_ty.is_floating() and other_scalar_ty.is_int():
    other = cast_impl(other, input_scalar_ty, builder)
  # / float
  elif input_scalar_ty.is_int() and other_scalar_ty.is_floating():
    input = cast_impl(input, other_scalar_ty, builder)
  # / (cast to float32)
  elif input_scalar_ty.is_int() and other_scalar_ty.is_int():
    input = cast_impl(input, builder.get_float_ty(), builder)
    other = cast_impl(other, builder.get_float_ty(), builder)
  
  # float / float (cast to highest exponent type)
  elif input_scalar_ty.is_floating() and other_scalar_ty.is_floating():
    if input_scalar_ty.get_fp_mantissa_width() > other_scalar_ty.get_fp_mantissa_width():
      other = cast_impl(other, input_scalar_ty, builder)
    else:
      input = cast_impl(input, other_scalar_ty, builder)
  
  # unreachable
  else:
    assert False
  return builder.create_fdiv(input, other)

def floordiv(input, other, builder):
  input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
  input_scalar_ty = input.type.scalar
  other_scalar_ty = other.type.scalar
  if input_scalar_ty.is_int() and other_scalar_ty.is_int():
    ret_ty = integer_promote_impl(input_scalar_ty, other_scalar_ty)
    input = cast_impl(input, ret_ty, builder)
    other = cast_impl(other, ret_ty, builder)
    if ret_ty.is_int_signed():
      return builder.create_sdiv(input, other)
    else:
      return builder.create_udiv(input, other)
    
  
  assert False

def fdiv(input, other, ieee_rounding, builder):
  input_scalar_ty = input.type.scalar
  other_scalar_ty = other.type.scalar
  if not input_scalar_ty.is_floating() or not other_scalar_ty.is_floating():
    raise ValueError("both operands of fdiv must have floating poscalar type")
  input, other = binary_op_type_checking_impl(input, other, builder, False, False, False, True)
  ret = builder.create_fdiv(input, other)
  if isinstance(ret, ir.binary_operator):
    ret.set_fdiv_ieee_rounding(ieee_rounding.value)
  return ret

def mod(input, other, builder):
  input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
  scalar_ty = input.type.scalar
  other_scalar_ty = other.type.scalar
  # float % int
  if scalar_ty.is_floating():
    return builder.create_frem(input, other)
  # % int
  elif scalar_ty.is_int():
    if scalar_ty.int_signedness != other_scalar_ty.int_signedness:
      raise ValueError("Cannot mod " + scalar_ty.repr() + " by " + other_scalar_ty.repr() + " because they have different signedness;" 
                          "this is unlikely to result in a useful answer. Cast them to the same signedness.")
    
    if scalar_ty.is_int_signed():
      return builder.create_srem(input, other)
    else:
      return builder.create_urem(input, other)
    
  
  assert False


def bitwise_op_type_checking_impl(input, other, builder):
  input, other = binary_op_type_checking_impl(input, other, builder, False, False, False)
  input_sca_ty = input.type.scalar
  other_sca_ty = other.type.scalar
  if not input_sca_ty.is_int() or not other_sca_ty.is_int():
    raise IncompatibleTypeErrorimpl(input_sca_ty, other_sca_ty)
  ret_sca_ty = integer_promote_impl(input_sca_ty, other_sca_ty)
  if ret_sca_ty != input_sca_ty:
    input = cast_impl(input, ret_sca_ty, builder)
  if ret_sca_ty != other_sca_ty:
    other = cast_impl(other, ret_sca_ty, builder)
  return input, other

def and_(input, other, builder):
  input, other = bitwise_op_type_checking_impl(input, other, builder)
  return builder.create_and(input, other)

def or_(input, other, builder):
  input, other = bitwise_op_type_checking_impl(input, other, builder)
  return builder.create_or(input, other)


def xor_(input, other, builder):
  input, other = bitwise_op_type_checking_impl(input, other, builder)
  return builder.create_xor(input, other)


def lshr(input, other, builder):
  input, other = bitwise_op_type_checking_impl(input, other, builder)
  return builder.create_lshr(input, other)


def shl(input, other, builder):
  input, other = bitwise_op_type_checking_impl(input, other, builder)
  return builder.create_shl(input, other)

#===----------------------------------------------------------------------===//
#                               Unary Operators
#===----------------------------------------------------------------------===//

def plus(input, ):
  return input

def minus(input, builder):
  input_sca_ty = input.type.scalar
  if input_sca_ty.is_ptr():
    raise ValueError("wrong type argument to unary minus (" + input_sca_ty.repr() + ")")
  _0 = ir.constant.get_null_value(input_sca_ty)
  return sub(_0, input, builder)

def invert(input, builder):
  input_sca_ty = input.type.scalar
  if input_sca_ty.is_ptr() or input_sca_ty.is_floating():
    raise ValueError("wrong type argument to unary invert (" + input_sca_ty.repr() + ")")
  _1 = ir.constant.get_all_ones_value(input_sca_ty)
  return xor_(input, _1, builder)


#===----------------------------------------------------------------------===//
#                               Comparison Operators
#===----------------------------------------------------------------------===//

def greater_than(input, other, builder):
  input, other = binary_op_type_checking_impl(input, other, builder)
  scalar_ty = input.type.scalar
  # float > float
  if scalar_ty.is_floating():
    return builder.create_fcmpOGT(input, other)
  # > int
  elif scalar_ty.is_int():
    if scalar_ty.is_int_signed():
      return builder.create_icmpSGT(input, other)
    else:
      return builder.create_icmpUGT(input, other)
    
  
  assert False

def greater_equal(input, other, builder):
  input, other = binary_op_type_checking_impl(input, other, builder)
  scalar_ty = input.type.scalar
  # float >= float
  if scalar_ty.is_floating():
    return builder.create_fcmpOGE(input, other)
  # >= int
  elif scalar_ty.is_int():
    if scalar_ty.is_int_signed():
      return builder.create_icmpSGE(input, other)
    else:
      return builder.create_icmpUGE(input, other)
    
  
  assert False

def less_than(input, other, builder):
  input, other = binary_op_type_checking_impl(input, other, builder)
  scalar_ty = input.type.scalar
  # float < float
  if scalar_ty.is_floating():
    return builder.create_fcmpOLT(input, other)
  # < int
  elif scalar_ty.is_int():
    if scalar_ty.is_int_signed():
      return builder.create_icmpSLT(input, other)
    else:
      return builder.create_icmpULT(input, other)
    
  
  assert False

def less_equal(input, other, builder):
  input, other = binary_op_type_checking_impl(input, other, builder)
  scalar_ty = input.type.scalar
  # float < float
  if scalar_ty.is_floating():
    return builder.create_fcmpOLE(input, other)
  # < int
  elif scalar_ty.is_int():
    if scalar_ty.is_int_signed():
      return builder.create_icmpSLE(input, other)
    else:
      return builder.create_icmpULE(input, other)
    
  
  assert False

def equal(input, other, builder):
  input, other = binary_op_type_checking_impl(input, other, builder)
  scalar_ty = input.type.scalar
  # float == float
  if scalar_ty.is_floating():
    return builder.create_fcmpOEQ(input, other)
  # == int
  elif scalar_ty.is_int():
    return builder.create_icmpEQ(input, other)
  assert False

def not_equal(input, other, builder):
  input, other = binary_op_type_checking_impl(input, other, builder)
  scalar_ty = input.type.scalar
  # float == float
  if scalar_ty.is_floating():
    return builder.create_fcmpUNE(input, other)
  # == int
  elif scalar_ty.is_int():
    return builder.create_icmpNE(input, other)
  assert False

#===----------------------------------------------------------------------===//
#                               Block Creation
#===----------------------------------------------------------------------===//

def arange(start, end, builder):
  return builder.get_range(start, end)

def zeros(shape, dtype, builder):
  _0 = ir.constant.get_null_value(dtype)
  return builder.create_splat(_0, shape)

#===----------------------------------------------------------------------===//
#                               Shape Manipulation
#===----------------------------------------------------------------------===//


def reshape(input, dst_shape, builder):
  numel = 1
  for s in dst_shape: 
    numel *= s
  if input.type.numel != numel:
    raise ValueError("cannot reshape block of different shape")
  return builder.create_reshape(input, dst_shape)

def cat(lhs, rhs, builder):
  return builder.create_cat(lhs, rhs)

def broadcast_impl1(input, shape, builder):
  if not input.type.is_block():
    return builder.create_splat(input, shape)
  src_shape = input.type.get_block_shapes()
  if len(src_shape) != len(shape):
    raise ValueError("Cannot broadcast")
  if shape == src_shape:
    return input
  return builder.create_broadcast(input, shape)

def broadcast_impl2(lhs,  rhs, builder):
  lhs_ty = lhs.type
  rhs_ty = rhs.type

  # make_shape_compatible(block, scalar)
  if lhs_ty.is_block() and not rhs_ty.is_block():
    rhs = builder.create_splat(rhs, lhs_ty.get_block_shapes())
  # make_shape_compatible(scalar, block)
  elif not lhs_ty.is_block() and rhs_ty.is_block():
    lhs = builder.create_splat(lhs, rhs_ty.get_block_shapes())
  # make_shape_compatible(block, block)
  elif lhs_ty.is_block() and rhs_ty.is_block():
    lhs_shape = lhs_ty.get_block_shapes()
    rhs_shape = rhs_ty.get_block_shapes()
    if len(lhs_shape) != len(rhs_shape):
      raise ValueError("Cannot make_shape_compatible: blocks must have the same rank")
    ret_shape = []
    for i in range(len(lhs_shape)):
      left = lhs_shape[i]
      right = rhs_shape[i]
      if left == 1:
        ret_shape.append(right)
      elif right == 1:
        ret_shape.append(left)
      elif left == right:
        ret_shape.append(left)
      else:
        raise ValueError("Cannot make_shape_compatible: incompatible dimensions at index " + str(i) +
                                 ": " + str(left) + " and " + str(right))
    if lhs_shape != ret_shape:
      lhs = builder.create_broadcast(lhs, ret_shape)
    if rhs_shape != ret_shape:
      rhs = builder.create_broadcast(rhs, ret_shape)
  return lhs, rhs

def broadcast_impl(lhs, rhs, builder):
  if isinstance(rhs, list):
    return broadcast_impl1(lhs, rhs, builder)
  return broadcast_impl2(lhs, rhs, builder)


# temporary until typesystem is properly merged too
def bitcast_impl(input, dst_ty, builder):
  return bitcast(input, dst_ty, builder).handle

def bitcast(input, dst_ty, builder):
  src_ty = input.type
  if src_ty.is_block():
    dst_ty = ir.type.make_block(dst_ty, input.type.get_block_shapes())
  if src_ty == dst_ty:
    return input
  src_sca_ty = src_ty.scalar
  dst_sca_ty = dst_ty.scalar
  if src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
    return cast_impl(input, dst_ty, builder)
  # Bitcast
  src_bits = src_sca_ty.primitive_bitwidth
  dst_bits = dst_sca_ty.primitive_bitwidth
  if  src_bits!= dst_bits:
    raise ValueError("Cannot bitcast data-type of size " + str(src_bits) +
                             "to data-type of size " + str(dst_bits))
  return builder.create_bitcast(input, dst_ty)

# temporary until typesystem is properly merged too
def cast_impl(input, dst_ty, builder):
  return cast(input, dst_ty, builder).handle

def cast(input, dst_ty, builder):
  src_ty = input.type
  if src_ty.is_block():
    dst_ty = ir.type.make_block(dst_ty, input.type.get_block_shapes())
  if src_ty == dst_ty:
    return input
  src_sca_ty = src_ty.scalar
  dst_sca_ty = dst_ty.scalar
  #
  if (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()) or\
     (dst_sca_ty.is_bf16() and not src_sca_ty.is_fp32()):
    return cast_impl(cast_impl(input, builder.get_float_ty(), builder), dst_sca_ty, builder)
  
  # FP Truncation
  truncate_fp = src_sca_ty.is_floating() and\
                dst_sca_ty.is_floating() and\
                src_sca_ty.get_fp_mantissa_width() > dst_sca_ty.get_fp_mantissa_width()
  if truncate_fp:
    return builder.create_fp_trunc(input, dst_ty)
  # FP Extension
  ext_fp = src_sca_ty.is_floating() and\
                dst_sca_ty.is_floating() and\
                src_sca_ty.get_fp_mantissa_width() < dst_sca_ty.get_fp_mantissa_width()
  if ext_fp:
    return builder.create_fp_ext(input, dst_ty)

  # Int cast
  if src_sca_ty.is_int() and dst_sca_ty.is_int() and\
    (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or
     src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
    sign_extend = src_sca_ty.is_int_signed() and src_sca_ty != builder.get_int1_ty()
    return builder.create_int_cast(input, dst_ty, sign_extend)
  
  # Float to Int
  if src_sca_ty.is_floating() and dst_sca_ty.is_int():
    if dst_sca_ty.is_bool():
      return builder.create_fp_to_ui(input, dst_ty)
    else:
      return builder.create_fp_to_si(input, dst_ty)
  
  # . Float
  if src_sca_ty.is_int() and dst_sca_ty.is_floating():
    if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
      return builder.create_ui_to_fp(input, dst_ty)
    else:
      return builder.create_si_to_fp(input, dst_ty)
  
  if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
    bitwidth = dst_sca_ty.int_bitwidth
    if bitwidth == 64:
      return builder.create_cast(ir.PtrToInt, input, dst_ty)
    if bitwidth == 1:
      return not_equal(cast_impl(input, builder.get_int64_ty(), builder),
                                 builder.get_int64(0),
                                 builder)
  
  if not src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
    return builder.create_int_to_ptr(input, dst_ty)
  # Ptr . Ptr
  if src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
    return builder.create_bitcast(input, dst_ty)
  # * . Bool
  if dst_sca_ty.is_bool():
    if src_sca_ty.is_ptr():
      input = cast_impl(input, builder.get_int64_ty(), builder)
    other = builder.get_int64(0)
    if src_ty.is_bool():
      other = builder.create_splat(other, src_ty.get_block_shapes())
    return builder.create_icmpNE(input, other)
  
  assert False

#===----------------------------------------------------------------------===//
#                               Memory Operators
#===----------------------------------------------------------------------===//

def load( ptr,  mask,  other, cache_modifier, eviction_policy, is_volatile,  builder):
  is_volatile = is_volatile.value

  if not ptr.type.scalar.is_ptr():
    raise ValueError("Pointer argument of load instruction is " + ptr.type.repr())
  if ptr.type.is_block():
    if mask:
      mask = broadcast_impl(mask, ptr.type.get_block_shapes(), builder)
    if other:
      other = broadcast_impl(other, ptr.type.get_block_shapes(), builder)
  
  if other:
    other = cast_impl(other, ptr.type.scalar.element, builder)
  ptr_ty = ptr.type.scalar
  elt_ty = ptr_ty.element
  # treat bool* as int8*
  if elt_ty == builder.get_int1_ty():
    elt_ty = builder.get_int8_ty()
    ptr_ty = ir.type.make_ptr(elt_ty, ptr_ty.address_space)
    ptr = cast_impl(ptr, ptr_ty, builder)
  
  # cache modifier
  cache = ir.CACHE_MODIFIER.NONE; # default
  if cache_modifier:
    if cache_modifier == ".ca":
      cache = ir.CACHE_MODIFIER.CA
    elif cache_modifier == ".cg":
      cache = ir.CACHE_MODIFIER.CG
    else:
      raise ValueError(f"Cache modifier {cache_modifier} not supported")
  
  # eviction policy
  eviction = ir.EVICTION_POLICY.NORMAL; #default
  if eviction_policy:
    if eviction_policy == "evict_last":
        eviction = ir.EVICTION_POLICY.EVICT_LAST
    elif eviction_policy == "evict_first":
        eviction = ir.EVICTION_POLICY.EVICT_FIRST
    else:
        raise ValueError(f"Eviction policy {eviction_policy} not supported")


  if not mask and not other:
    return builder.create_load(ptr, cache, eviction, is_volatile)
  if not mask:
    raise ValueError("`other` cannot be provided without `mask`")
  shape = ptr.type.get_block_shapes()
  if not other:
    other = ir.undef.get(elt_ty)
    if ptr.type.is_block():
      other = builder.create_splat(other, ptr.type.get_block_shapes())
  
  return builder.create_masked_load(ptr, mask, other, cache, eviction, is_volatile)

def store( ptr, val,  mask, builder):
  if not ptr.type.scalar.is_ptr():
    raise ValueError("Pointer argument of store instruction is " + ptr.type.repr())
  if ptr.type.is_block():
    val = broadcast_impl(val, ptr.type.get_block_shapes(), builder)
  if mask:
    mask = broadcast_impl(mask, ptr.type.get_block_shapes(), builder)
  ptr_ty = ptr.type.scalar
  elt_ty = ptr_ty.element
  # treat bool* as int8*
  if elt_ty == builder.get_int1_ty():
    elt_ty = builder.get_int8_ty()
    ptr_ty = ir.type.make_ptr(elt_ty, ptr_ty.address_space)
    ptr = cast_impl(ptr, ptr_ty, builder)
  
  # cast to target data-type
  val = cast_impl(val, elt_ty, builder)
  if not mask:
    return builder.create_store(ptr, val)
  if not mask.type.scalar.is_bool():
    raise ValueError("Mask must have boolean scalar type")
  return builder.create_masked_store(ptr, val, mask)

def atomic_cas( ptr, cmp, val, builder):
  return builder.create_atomic_cas(ptr, cmp, val)

def atom_red_typechecking_impl( ptr, val, mask, builder):
  if not ptr.type.scalar.is_ptr():
    raise ValueError("Pointer argument of store instruction is " + ptr.type.repr())
  if ptr.type.is_block():
    if mask:
      mask = broadcast_impl(mask, ptr.type.get_block_shapes(), builder)
    if val:
      val = broadcast_impl(val, ptr.type.get_block_shapes(), builder)
  val = cast_impl(val, ptr.type.scalar.element, builder)
  if not mask:
    mask = builder.get_int1(True)
    if ptr.type.is_block():
      mask = builder.create_splat(mask, ptr.type.get_block_shapes())
  return ptr, val, mask
  

def atomic_max( ptr, val, mask, builder):
  ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
  sca_ty = val.type.scalar
  # direct call to atomic_max for integers
  if sca_ty.is_int():
    if sca_ty.is_int_signed():
      return builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, ptr, val, mask)
    else:
      return builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX, ptr, val, mask)
  # for float
  # return atomic_smax(i_ptr, i_val) if val >= 0
  # return atomic_umin(i_ptr, i_val) if val < 0
  i_val = bitcast_impl(val, builder.get_int32_ty(), builder)
  i_ptr = bitcast_impl(ptr, ir.type.make_ptr(builder.get_int32_ty(), 1), builder)
  pos = greater_equal(val, ir.constant_float.get(sca_ty, 0), builder)
  neg = less_than(val, ir.constant_float.get(sca_ty, 0), builder)
  pos_ret = builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, i_ptr, i_val, and_(mask, pos, builder).handle)
  neg_ret = builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN, i_ptr, i_val, and_(mask, neg, builder).handle)
  return where(pos, pos_ret, neg_ret, builder)

def atomic_min( ptr, val, mask, builder):
  ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
  sca_ty = val.type.scalar
  # direct call to atomic_min for integers
  if sca_ty.is_int():
    if sca_ty.is_int_signed():
      return builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, ptr, val, mask)
    else:
      return builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN, ptr, val, mask)
  # for float
  # return atomic_smin(i_ptr, i_val) if val >= 0
  # return atomic_umax(i_ptr, i_val) if val < 0
  i_val = bitcast_impl(val, builder.get_int32_ty(), builder)
  i_ptr = bitcast_impl(ptr, ir.type.make_ptr(builder.get_int32_ty(), 1), builder)
  pos = greater_equal(val, ir.constant_float.get(sca_ty, 0), builder)
  neg = less_than(val, ir.constant_float.get(sca_ty, 0), builder)
  pos_ret = builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, i_ptr, i_val, and_(mask, pos, builder).handle)
  neg_ret = builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX, i_ptr, i_val, and_(mask, neg, builder).handle)
  return where(pos, pos_ret, neg_ret, builder)

def atomic_add( ptr, val, mask, builder):
  ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
  sca_ty = val.type.scalar
  op = ir.ATOMIC_OP.FADD if sca_ty.is_floating() else ir.ATOMIC_OP.ADD
  return builder.create_atomic_rmw(op, ptr, val, mask)

def atomic_and( ptr, val, mask, builder):
  ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
  return builder.create_atomic_rmw(ir.ATOMIC_OP.AND, ptr, val, mask)

def atomic_or( ptr, val, mask, builder):
  ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
  return builder.create_atomic_rmw(ir.ATOMIC_OP.OR, ptr, val, mask)

def atomic_xor( ptr, val, mask, builder):
  ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
  return builder.create_atomic_rmw(ir.ATOMIC_OP.XOR, ptr, val, mask)

def atomic_xchg( ptr, val, mask, builder):
  ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
  return builder.create_atomic_rmw(ir.ATOMIC_OP.XCHG, ptr, val, mask)

#===----------------------------------------------------------------------===//
#                               Linear Algebra
#===----------------------------------------------------------------------===//

def dot(lhs, rhs, allow_tf32, builder):
  _0 = None
  if lhs.type.is_int_or_tileint():
    _0 = builder.get_int32(0)
  else:
    _0 = builder.get_float32(0)
  M = lhs.type.shape[0]
  N = rhs.type.shape[1]
  _0 = builder.create_splat(_0, [M, N])
  _allow_tf32 = allow_tf32.value != 0
  return builder.create_dot(lhs, rhs, _0, _allow_tf32)


#===----------------------------------------------------------------------===//
#                               Indexing
#===----------------------------------------------------------------------===//

def where( condition, x, y, builder):
  condition = cast_impl(condition, builder.get_int1_ty(), builder)
  if condition.type.is_block():
    x = broadcast_impl(x, condition.type.get_block_shapes(), builder)
    y = broadcast_impl(y, condition.type.get_block_shapes(), builder)
  
  x_ty = x.type.scalar
  y_ty = y.type.scalar
  ty = computation_type_impl(x_ty, y_ty, div_or_mod=False)
  x = cast_impl(x, ty, builder)
  y = cast_impl(y, ty, builder)
  return builder.create_select(condition, x, y)


#===----------------------------------------------------------------------===//
#                               Reductions
#===----------------------------------------------------------------------===//

def reduce_impl(input, axis, builder, name,
                FLOAT_OP, INT_OP):
  scalar_ty = input.type.scalar
  # input is extended to 32-bits if necessary
  # this increases numerical accuracy and can be done pretty much for free
  # on GPUs
  if scalar_ty.is_int() and scalar_ty.int_bitwidth <= 32:
    input = cast_impl(input, ir.type.get_int32(scalar_ty.context), builder)
  if scalar_ty.is_floating():
    return builder.create_reduce(input, FLOAT_OP, axis)
  elif scalar_ty.is_int():
    return builder.create_reduce(input, INT_OP, axis)
  assert False

def min(input, axis, builder):
  return reduce_impl(input, axis, builder, "min", ir.REDUCE_OP.FMIN, ir.REDUCE_OP.MIN)

def max(input, axis, builder):
  return reduce_impl(input, axis, builder, "max", ir.REDUCE_OP.FMAX, ir.REDUCE_OP.MAX)

def sum(input, axis, builder):
  return reduce_impl(input, axis, builder, "sum", ir.REDUCE_OP.FADD, ir.REDUCE_OP.ADD)

def xor_sum(input, axis, builder):
  scalar_ty = input.type.scalar
  if not scalar_ty.is_int():
    raise ValueError("xor_sum only supported for integers")
  return reduce_impl(input, axis, builder, "sum", ir.REDUCE_OP.XOR, ir.REDUCE_OP.XOR)


#===----------------------------------------------------------------------===//
#                               Math
#===----------------------------------------------------------------------===//

def umulhi(x,  y, builder):
  binary_op_type_checking_impl(x, y, builder)
  return builder.insert(ir.umulhi_inst.create(x, y))

def exp(x, builder):
  return builder.create_exp(x)

def log(x, builder):
  return builder.create_log(x)

def cos(x, builder):
  return builder.create_cos(x)

def sin(x, builder):
  return builder.create_sin(x)

def sqrt(x, builder):
  return builder.create_sqrt(x)


##

def multiple_of(x, value):
  i = x
  if not i:
    assert False
  i.set_metadata(ir.metadata.multiple_of, value)
  return i

def max_contiguous(x, value):
  i = x
  if not i:
    assert False
  i.set_metadata(ir.metadata.max_contiguous, value)
  return i

def debug_barrier(builder):
  return builder.create_barrier()


