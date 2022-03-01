from triton._C.libtriton.triton import frontend, ir

class SemanticError(Exception):
    pass

class IncompatibleTypeError(Exception):
    def __init__(self, a_ty, b_ty):
      pass


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
class Signedness:
  UNSIGNED = 0
  SIGNED = 1


def integer_promote(a_ty, b_ty):
  a_rank = a_ty.get_integer_bitwidth()
  b_rank = b_ty.get_integer_bitwidth()
  a_sn = a_ty.get_integer_signedness()
  b_sn = b_ty.get_integer_signedness()
  # Rules for signedness taken from "Usual arithmetic conversions" on
  # https://en.cppreference.com/w/c/language/conversion.
  if a_sn == b_sn:
    return a_ty if a_rank > b_rank else b_ty
  elif a_sn == Signedness.UNSIGNED:
    return a_ty if a_rank >= b_rank else b_ty
  elif b_sn == Signedness.UNSIGNED:
    return b_ty if b_rank >= a_rank else a_ty
  else:
    assert False
  

def computation_type(a_ty, b_ty, div_or_mod):
  ctx = a_ty.get_context()
  # 1) if one operand is double, the other is implicitly
  #    converted to double
  if a_ty.is_fp64_ty() or b_ty.is_fp64_ty():
    return ir.type.get_fp64_ty(ctx)
  # 2) if one operand is float, the other is implicitly
  #    converted to float
  if a_ty.is_fp32_ty() or b_ty.is_fp32_ty():
    return ir.type.get_fp32_ty(ctx)
  # 3 ) if one operand is half, the other is implicitly converted to half
  #     unless we're doing / or %, which do not exist natively in PTX for fp16.
  if a_ty.is_fp16_ty() or b_ty.is_fp16_ty():
    if div_or_mod:
      return ir.type.get_fp32_ty(ctx)
    else:
      return ir.type.get_fp16_ty(ctx)
  if not a_ty.is_integer_ty() or not b_ty.is_integer_ty():
    assert False
  # 4 ) both operands are integer and undergo
  #    integer promotion
  if div_or_mod and a_ty.get_integer_signedness() != b_ty.get_integer_signedness():
    raise SemanticError("Cannot use /, #, or % with " + a_ty.repr() + " and " + b_ty.repr() + " because they have different signedness;" 
                        "this is unlikely to result in a useful answer. Cast them to the same signedness.")
  
  return integer_promote(a_ty, b_ty)

#===----------------------------------------------------------------------===//
#                               Binary Operators
#===----------------------------------------------------------------------===//

def check_ptr_type(type_a, type_b, allow_ptr_a):
  if type_a.is_pointer_ty():
    if not allow_ptr_a:
      raise IncompatibleTypeError(type_a, type_b)
    # T* + U* with T != U
    if type_b.is_pointer_ty() and (type_a != type_b):
      raise IncompatibleTypeError(type_a, type_b)
    # T* + float
    if type_b.is_floating_point_ty():
      raise IncompatibleTypeError(type_a, type_b)
  

def binary_op_type_checking(lhs,  rhs,  builder,
                            allow_lhs_ptr = False, allow_rhs_ptr = False,
                            arithmetic_check = True, div_or_mod = False):
  # implicit broadcasting
  lhs, rhs = broadcast(lhs, rhs, builder)
  # implicit typecasting
  lhs_sca_ty = lhs.get_type().get_scalar_ty()
  rhs_sca_ty = rhs.get_type().get_scalar_ty()
  check_ptr_type(lhs_sca_ty, rhs_sca_ty, allow_lhs_ptr)
  check_ptr_type(rhs_sca_ty, lhs_sca_ty, allow_rhs_ptr)
  if arithmetic_check and not lhs_sca_ty.is_pointer_ty() and not rhs_sca_ty.is_pointer_ty():
    ret_sca_ty = computation_type(lhs_sca_ty, rhs_sca_ty, div_or_mod)
    lhs = cast(lhs, ret_sca_ty, builder)
    rhs = cast(rhs, ret_sca_ty, builder)
  

def add(input, other, builder):
  binary_op_type_checking(input, other, builder, True, True)
  input_scalar_ty = input.get_type().get_scalar_ty()
  other_scalar_ty = other.get_type().get_scalar_ty()
  # offset + ptr
  # ptr + offset
  if other_scalar_ty.is_pointer_ty() and not input_scalar_ty.is_pointer_ty():
    input, other = other, input
  if input_scalar_ty.is_pointer_ty():
    return builder.create_gep(input, [other])
  # float + float
  elif input_scalar_ty.is_floating_point_ty():
    return builder.create_fadd(input, other)
  # + int
  elif input_scalar_ty.is_integer_ty():
    return builder.create_add(input, other)
  assert False

def sub(input, other, builder):
  binary_op_type_checking(input, other, builder, True, False)
  scalar_ty = input.get_type().get_scalar_ty()
  # ptr - offset
  if scalar_ty.is_pointer_ty():
    return builder.create_gep(input, [minus(other, builder)])
  # float + float
  if scalar_ty.is_floating_point_ty():
    return builder.create_fsub(input, other)
  # + int
  elif scalar_ty.is_integer_ty():
    return builder.create_sub(input, other)
  assert False

def mul(input, other, builder):
  binary_op_type_checking(input, other, builder)
  scalar_ty = input.get_type().get_scalar_ty()
  # float * float
  if scalar_ty.is_floating_point_ty():
    return builder.create_fmul(input, other)
  # * int
  elif scalar_ty.is_integer_ty():
    return builder.create_mul(input, other)
  assert False

def Truediv(input, other, builder):
  binary_op_type_checking(input, other, builder, False, False, True, True)
  input_scalar_ty = input.get_type().get_scalar_ty()
  other_scalar_ty = other.get_type().get_scalar_ty()
  # float / int
  if input_scalar_ty.is_floating_point_ty() and other_scalar_ty.is_integer_ty():
    other = cast(other, input_scalar_ty, builder)
  # / float
  elif input_scalar_ty.is_integer_ty() and other_scalar_ty.is_floating_point_ty():
    input = cast(input, other_scalar_ty, builder)
  # / (cast to float32)
  elif input_scalar_ty.is_integer_ty() and other_scalar_ty.is_integer_ty():
    input = cast(input, builder.get_float_ty(), builder)
    other = cast(other, builder.get_float_ty(), builder)
  
  # float / float (cast to highest exponent type)
  elif input_scalar_ty.is_floating_point_ty() and other_scalar_ty.is_floating_point_ty():
    if input_scalar_ty.get_fp_mantissa_width() > other_scalar_ty.get_fp_mantissa_width():
      other = cast(other, input_scalar_ty, builder)
    else:
      input = cast(input, other_scalar_ty, builder)
  
  # unreachable
  else:
    assert False
  return builder.create_fdiv(input, other)

def floordiv(input, other, builder):
  binary_op_type_checking(input, other, builder, False, False, True, True)
  input_scalar_ty = input.get_type().get_scalar_ty()
  other_scalar_ty = other.get_type().get_scalar_ty()
  if input_scalar_ty.is_integer_ty() and other_scalar_ty.is_integer_ty():
    ret_ty = integer_promote(input_scalar_ty, other_scalar_ty)
    input = cast(input, ret_ty, builder)
    other = cast(other, ret_ty, builder)
    if ret_ty.is_integer_signed():
      return builder.create_sdiv(input, other)
    else:
      return builder.create_udiv(input, other)
    
  
  assert False

def fdiv(input, other, ieee_rounding, builder):
  input_scalar_ty = input.get_type().get_scalar_ty()
  other_scalar_ty = other.get_type().get_scalar_ty()
  if not input_scalar_ty.is_floating_point_ty() or not other_scalar_ty.is_floating_point_ty():
    raise SemanticError("both operands of fdiv must have floating poscalar type")
  binary_op_type_checking(input, other, builder, False, False, False, True)
  ret = builder.create_fdiv(input, other)
  if isinstance(ret, ir.binary_operator):
    ret.set_fdiv_ieee_rounding(ieee_rounding.get_value())
  return ret

def mod(input, other, builder):
  binary_op_type_checking(input, other, builder, False, False, True, True)
  scalar_ty = input.get_type().get_scalar_ty()
  other_scalar_ty = other.get_type().get_scalar_ty()
  # float % int
  if scalar_ty.is_floating_point_ty():
    return builder.create_frem(input, other)
  # % int
  elif scalar_ty.is_integer_ty():
    if scalar_ty.get_integer_signedness() != other_scalar_ty.get_integer_signedness():
      raise SemanticError("Cannot mod " + scalar_ty.repr() + " by " + other_scalar_ty.repr() + " because they have different signedness;" 
                          "this is unlikely to result in a useful answer. Cast them to the same signedness.")
    
    if scalar_ty.is_integer_signed():
      return builder.create_srem(input, other)
    else:
      return builder.create_urem(input, other)
    
  
  assert False


def bitwise_op_type_checking(input, other, builder):
  binary_op_type_checking(input, other, builder, False, False, False)
  input_sca_ty = input.get_type().get_scalar_ty()
  other_sca_ty = other.get_type().get_scalar_ty()
  if not input_sca_ty.is_integer_ty() or not other_sca_ty.is_integer_ty():
    throw_incompatible_types(input_sca_ty, other_sca_ty)
  ret_sca_ty = integer_promote(input_sca_ty, other_sca_ty)
  if ret_sca_ty != input_sca_ty:
    input = cast(input, ret_sca_ty, builder)
  if ret_sca_ty != other_sca_ty:
    other = cast(other, ret_sca_ty, builder)

def and_(input, other, builder):
  bitwise_op_type_checking(input, other, builder)
  return builder.create_and(input, other)

def or_(input, other, builder):
  bitwise_op_type_checking(input, other, builder)
  return builder.create_or(input, other)


def xor_(input, other, builder):
  bitwise_op_type_checking(input, other, builder)
  return builder.create_xor(input, other)


def lshr(input, other, builder):
  bitwise_op_type_checking(input, other, builder)
  return builder.create_lshr(input, other)


def shl(input, other, builder):
  bitwise_op_type_checking(input, other, builder)
  return builder.create_shl(input, other)

#===----------------------------------------------------------------------===//
#                               Unary Operators
#===----------------------------------------------------------------------===//

def plus(input, ):
  return input

def minus(input, builder):
  input_sca_ty = input.get_type().get_scalar_ty()
  if input_sca_ty.is_pointer_ty():
    raise SemanticError("wrong type argument to unary minus (" + input_sca_ty.repr() + ")")
  _0 = ir.constant.get_null_value(input_sca_ty)
  return sub(_0, input, builder)

def invert(input, builder):
  input_sca_ty = input.get_type().get_scalar_ty()
  if input_sca_ty.is_pointer_ty() or input_sca_ty.is_floating_point_ty():
    raise SemanticError("wrong type argument to unary invert (" + input_sca_ty.repr() + ")")
  _1 = ir.constant.get_all_ones_value(input_sca_ty)
  return xor_(input, _1, builder)


#===----------------------------------------------------------------------===//
#                               Comparison Operators
#===----------------------------------------------------------------------===//

def greater_than(input, other, builder):
  binary_op_type_checking(input, other, builder)
  scalar_ty = input.get_type().get_scalar_ty()
  # float > float
  if scalar_ty.is_floating_point_ty():
    return builder.create_fcmpOGT(input, other)
  # > int
  elif scalar_ty.is_integer_ty():
    if scalar_ty.is_integer_signed():
      return builder.create_icmpSGT(input, other)
    else:
      return builder.create_icmpUGT(input, other)
    
  
  assert False

def greater_equal(input, other, builder):
  binary_op_type_checking(input, other, builder)
  scalar_ty = input.get_type().get_scalar_ty()
  # float >= float
  if scalar_ty.is_floating_point_ty():
    return builder.create_fcmpOGE(input, other)
  # >= int
  elif scalar_ty.is_integer_ty():
    if scalar_ty.is_integer_signed():
      return builder.create_icmpSGE(input, other)
    else:
      return builder.create_icmpUGE(input, other)
    
  
  assert False

def less_than(input, other, builder):
  binary_op_type_checking(input, other, builder)
  scalar_ty = input.get_type().get_scalar_ty()
  # float < float
  if scalar_ty.is_floating_point_ty():
    return builder.create_fcmpOLT(input, other)
  # < int
  elif scalar_ty.is_integer_ty():
    if scalar_ty.is_integer_signed():
      return builder.create_icmpSLT(input, other)
    else:
      return builder.create_icmpULT(input, other)
    
  
  assert False

def less_equal(input, other, builder):
  binary_op_type_checking(input, other, builder)
  scalar_ty = input.get_type().get_scalar_ty()
  # float < float
  if scalar_ty.is_floating_point_ty():
    return builder.create_fcmpOLE(input, other)
  # < int
  elif scalar_ty.is_integer_ty():
    if scalar_ty.is_integer_signed():
      return builder.create_icmpSLE(input, other)
    else:
      return builder.create_icmpULE(input, other)
    
  
  assert False

def equal(input, other, builder):
  binary_op_type_checking(input, other, builder)
  scalar_ty = input.get_type().get_scalar_ty()
  # float == float
  if scalar_ty.is_floating_point_ty():
    return builder.create_fcmpOEQ(input, other)
  # == int
  elif scalar_ty.is_integer_ty():
    return builder.create_icmpEQ(input, other)
  assert False

def not_equal(input, other, builder):
  binary_op_type_checking(input, other, builder)
  scalar_ty = input.get_type().get_scalar_ty()
  # float == float
  if scalar_ty.is_floating_point_ty():
    return builder.create_fcmpUNE(input, other)
  # == int
  elif scalar_ty.is_integer_ty():
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
  if input.get_type().get_tile_num_elements() != numel:
    raise SemanticError("cannot reshape block of different shape")
  return builder.create_reshape(input, dst_shape)

def cat(lhs, rhs, builder):
  return builder.create_cat(lhs, rhs)

def broadcast(input, shape, builder):
  if not input.get_type().is_block_ty():
    return builder.create_splat(input, shape)
  src_shape = input.get_type().get_block_shapes()
  if src_shape.size() != shape.size():
    raise SemanticError("Cannot broadcast")
  if shape == src_shape:
    return input
  return builder.create_broadcast(input, shape)

def broadcast(lhs,  rhs, builder):
  lhs_ty = lhs.get_type()
  rhs_ty = rhs.get_type()

  # make_shape_compatible(block, scalar)
  if lhs_ty.is_block_ty() and not rhs_ty.is_block_ty():
    rhs = builder.create_splat(rhs, lhs_ty.get_block_shapes())
  # make_shape_compatible(scalar, block)
  elif not lhs_ty.is_block_ty() and rhs_ty.is_block_ty():
    lhs = builder.create_splat(lhs, rhs_ty.get_block_shapes())
  # make_shape_compatible(block, block)
  elif lhs_ty.is_block_ty() and rhs_ty.is_block_ty():
    lhs_shape = lhs_ty.get_block_shapes()
    rhs_shape = rhs_ty.get_block_shapes()
    if lhs_shape.size() != rhs_shape.size():
      raise SemanticError("Cannot make_shape_compatible: blocks must have the same rank")
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
        raise SemanticError("Cannot make_shape_compatible: incompatible dimensions at index " + str(i) +
                                 ": " + str(left) + " and " + str(right))
    if lhs_shape != ret_shape:
      lhs = builder.create_broadcast(lhs, ret_shape)
    if rhs_shape != ret_shape:
      rhs = builder.create_broadcast(rhs, ret_shape)
  return lhs, rhs

def bitcast(input, dst_ty, builder):
  src_ty = input.get_type()
  if src_ty.is_block_ty():
    dst_ty = ir.block_type.get(dst_ty, input.get_type().get_block_shapes())
  if src_ty == dst_ty:
    return input
  src_sca_ty = src_ty.get_scalar_ty()
  dst_sca_ty = dst_ty.get_scalar_ty()
  if src_sca_ty.is_pointer_ty() or dst_sca_ty.is_pointer_ty():
    return cast(input, dst_ty, builder)
  # Bitcast
  src_bits = src_sca_ty.get_primitive_size_in_bits()
  dst_bits = dst_sca_ty.get_primitive_size_in_bits()
  if  src_bits!= dst_bits:
    raise SemanticError("Cannot bitcast data-type of size " + str(src_bits) +
                             "to data-type of size " + str(dst_bits))
  return builder.create_cast(ir.BitCast, input, dst_ty)

def cast(input, dst_ty, builder):
  src_ty = input.get_type()
  if src_ty.is_block_ty():
    dst_ty = ir.block_type.get(dst_ty, input.get_type().get_block_shapes())
  if src_ty == dst_ty:
    return input
  src_sca_ty = src_ty.get_scalar_ty()
  dst_sca_ty = dst_ty.get_scalar_ty()
  #
  if (src_sca_ty.is_bf16_ty() and not dst_sca_ty.is_fp32_ty()) or\
     (dst_sca_ty.is_bf16_ty() and not src_sca_ty.is_fp32_ty()):
    return cast(cast(input, builder.get_float_ty(), builder), dst_sca_ty, builder)
  
  # FP Truncation
  truncate_fp = src_sca_ty.is_floating_point_ty() and\
                dst_sca_ty.is_floating_point_ty() and\
                src_sca_ty.get_fp_mantissa_width() > dst_sca_ty.get_fp_mantissa_width()
  if truncate_fp:
    return builder.create_fp_trunc(input, dst_ty)
  # FP Extension
  ext_fp = src_sca_ty.is_floating_point_ty() and\
                dst_sca_ty.is_floating_point_ty() and\
                src_sca_ty.get_fp_mantissa_width() < dst_sca_ty.get_fp_mantissa_width()
  if ext_fp:
    return builder.create_fp_ext(input, dst_ty)

  # Int cast
  if src_sca_ty.is_integer_ty() and dst_sca_ty.is_integer_ty() and\
      (src_sca_ty.get_integer_bitwidth() != dst_sca_ty.get_integer_bitwidth() or
       src_sca_ty.get_integer_signedness() != dst_sca_ty.get_integer_signedness()):
    sign_extend = src_sca_ty.is_integer_signed() and src_sca_ty != builder.get_int1_ty()
    return builder.create_int_cast(input, dst_ty, sign_extend)
  
  # Float to Int
  if src_sca_ty.is_floating_point_ty() and dst_sca_ty.is_integer_ty():
    if dst_sca_ty.is_bool_ty():
      return builder.create_fp_to_ui(input, dst_ty)
    else:
      return builder.create_fp_to_si(input, dst_ty)
  
  # . Float
  if src_sca_ty.is_integer_ty() and dst_sca_ty.is_floating_point_ty():
    if src_sca_ty.is_bool_ty() or not src_sca_ty.is_integer_signed():
      return builder.create_ui_to_fp(input, dst_ty)
    else:
      return builder.create_si_to_fp(input, dst_ty)
  
  if src_sca_ty.is_pointer_ty() and dst_sca_ty.is_integer_ty():
    bitwidth = dst_sca_ty.get_integer_bitwidth()
    if bitwidth == 64:
      return builder.create_cast(ir.PtrToInt, input, dst_ty)
    if bitwidth == 1:
      return not_equal(cast(input, builder.get_int64_ty(), builder),
                                 builder.get_int64(0),
                                 builder)
  
  if not src_sca_ty.is_pointer_ty() and dst_sca_ty.is_pointer_ty():
    return builder.create_cast(ir.IntToPtr, input, dst_ty)
  # Ptr . Ptr
  if src_sca_ty.is_pointer_ty() and dst_sca_ty.is_pointer_ty():
    return builder.create_cast(ir.BitCast, input, dst_ty)
  # * . Bool
  if dst_sca_ty.is_bool_ty():
    if src_sca_ty.is_pointer_ty():
      input = cast(input, builder.get_int64_ty(), builder)
    other = builder.get_int64(0)
    if src_ty.is_bool_ty():
      other = builder.create_splat(other, src_ty.get_block_shapes())
    return builder.create_icmpNE(input, other)
  
  assert False

#===----------------------------------------------------------------------===//
#                               Memory Operators
#===----------------------------------------------------------------------===//

def load( ptr,  mask,  other, cache_modifier, eviction_policy, is_volatile,  builder):
  if not ptr.get_type().get_scalar_ty().is_pointer_ty():
    raise SemanticError("Pointer argument of load instruction is " + ptr.get_type().repr())
  if ptr.get_type().is_block_ty():
    if mask:
      mask = broadcast(mask, ptr.get_type().get_block_shapes(), builder)
    if other:
      other = broadcast(other, ptr.get_type().get_block_shapes(), builder)
  
  if other:
    other = cast(other, ptr.get_type().get_scalar_ty().get_pointer_element_ty(), builder)
  ptr_ty = ptr.get_type().get_scalar_ty()
  elt_ty = ptr_ty.get_pointer_element_ty()
  # treat bool* as int8*
  if elt_ty == builder.get_int1_ty():
    elt_ty = builder.get_int8_ty()
    ptr_ty = pointer_ir.type.get(elt_ty, ptr_ty.get_pointer_address_space())
    ptr = cast(ptr, ptr_ty, builder)
  
  # cache modifier
  cache = ir.load_inst.NONE; # default
  if not cache_modifier.empty():
    if cache_modifier == ".ca":
      cache = load_inst.CA
    elif cache_modifier == ".cg":
      cache = load_inst.CG
    else:
      raise SemanticError(f"Cache modifier {cache_modifier} not supported")
  
  # eviction policy
  eviction = load_inst.NORMAL; #default
  if not eviction_policy.empty():
    if eviction_policy == "evict_last":
        eviction = load_inst.EVICT_LAST
    elif eviction_policy == "evict_first":
        eviction = load_inst.EVICT_FIRST
    else:
        raise SemanticError(f"Eviction policy {eviction_policy} not supported")


  if not mask and not other:
    return builder.create_load(ptr, cache, eviction, is_volatile)
  if not mask:
    raise SemanticError("`other` cannot be provided without `mask`")
  shape = ptr.get_type().get_block_shapes()
  if not other:
    other = ir.undef_value.get(elt_ty)
    if ptr.get_type().is_block_ty():
      other = builder.create_splat(other, ptr.get_type().get_block_shapes())
  
  return builder.create_masked_load(ptr, mask, other, cache, eviction, is_volatile)

def store( ptr, val,  mask, builder):
  if not ptr.get_type().get_scalar_ty().is_pointer_ty():
    raise SemanticError("Pointer argument of store instruction is " + ptr.get_type().repr())
  if ptr.get_type().is_block_ty():
    val = broadcast(val, ptr.get_type().get_block_shapes(), builder)
  if mask:
    mask = broadcast(mask, ptr.get_type().get_block_shapes(), builder)
  ptr_ty = ptr.get_type().get_scalar_ty()
  elt_ty = ptr_ty.get_pointer_element_ty()
  # treat bool* as int8*
  if elt_ty == builder.get_int1_ty():
    elt_ty = builder.get_int8_ty()
    ptr_ty = pointer_ir.type.get(elt_ty, ptr_ty.get_pointer_address_space())
    ptr = cast(ptr, ptr_ty, builder)
  
  # cast to target data-type
  val = cast(val, elt_ty, builder)
  if not mask:
    return builder.create_store(ptr, val)
  if not mask.get_type().get_scalar_ty().is_bool_ty():
    raise SemanticError("Mask must have boolean scalar type")
  return builder.create_masked_store(ptr, val, mask)

def atomic_cas( ptr, cmp, val, builder):
  return builder.create_atomic_cas(ptr, cmp, val)

def atom_red_typechecking( ptr, val, mask, builder):
  if not ptr.get_type().get_scalar_ty().is_pointer_ty():
    raise SemanticError("Pointer argument of store instruction is " + ptr.get_type().repr())
  if ptr.get_type().is_block_ty():
    if mask:
      mask = broadcast(mask, ptr.get_type().get_block_shapes(), builder)
    
    if val:
      val = broadcast(val, ptr.get_type().get_block_shapes(), builder)
    
  
  val = cast(val, ptr.get_type().get_scalar_ty().get_pointer_element_ty(), builder)
  if not mask:
    mask = builder.get_int1(True)
    if ptr.get_type().is_block_ty():
      mask = builder.create_splat(mask, ptr.get_type().get_block_shapes())
  

def atomic_max( ptr, val, mask, builder):
  atom_red_typechecking(ptr, val, mask, builder)
  sca_ty = val.get_type().get_scalar_ty()
  # direct call to atomic_max for integers
  if sca_ty.is_integer_ty():
    if sca_ty.is_integer_signed():
      return builder.create_atomic_rmw(ir.atomic_rmw_op_t.Max, ptr, val, mask)
    else:
      return builder.create_atomic_rmw(ir.atomic_rmw_op_t.UMax, ptr, val, mask)
    
  
  # for float
  # return atomic_smax(i_ptr, i_val) if val >= 0
  # return atomic_umin(i_ptr, i_val) if val < 0
  i_val = bitcast(val, builder.get_int32_ty(), builder)
  i_ptr = bitcast(ptr, pointer_ir.type.get(builder.get_int32_ty(), 1), builder)
  pos = greater_equal(val, constant_fp.get(sca_ty, 0), builder)
  neg = less_than(val, constant_fp.get(sca_ty, 0), builder)
  pos_ret = builder.create_atomic_rmw(ir.atomic_rmw_op_t.Max, i_ptr, i_val, and_(mask, pos, builder))
  neg_ret = builder.create_atomic_rmw(ir.atomic_rmw_op_t.UMin, i_ptr, i_val, and_(mask, neg, builder))
  return where(pos, pos_ret, neg_ret, builder)

def atomic_min( ptr, val, mask, builder):
  atom_red_typechecking(ptr, val, mask, builder)
  sca_ty = val.get_type().get_scalar_ty()
  # direct call to atomic_min for integers
  if sca_ty.is_integer_ty():
    if sca_ty.is_integer_signed():
      return builder.create_atomic_rmw(ir.atomic_rmw_op_t.Min, ptr, val, mask)
    else:
      return builder.create_atomic_rmw(ir.atomic_rmw_op_t.UMin, ptr, val, mask)
    
  
  # for float
  # return atomic_smin(i_ptr, i_val) if val >= 0
  # return atomic_umax(i_ptr, i_val) if val < 0
  i_val = bitcast(val, builder.get_int32_ty(), builder)
  i_ptr = bitcast(ptr, pointer_ir.type.get(builder.get_int32_ty(), 1), builder)
  pos = greater_equal(val, constant_fp.get(sca_ty, 0), builder)
  neg = less_than(val, constant_fp.get(sca_ty, 0), builder)
  pos_ret = builder.create_atomic_rmw(ir.atomic_rmw_op_t.Min, i_ptr, i_val, and_(mask, pos, builder))
  neg_ret = builder.create_atomic_rmw(ir.atomic_rmw_op_t.UMax, i_ptr, i_val, and_(mask, neg, builder))
  return where(pos, pos_ret, neg_ret, builder)

def atomic_add( ptr, val, mask, builder):
  atom_red_typechecking(ptr, val, mask, builder)
  sca_ty = val.get_type().get_scalar_ty()
  op = ir.atomic_rmw_op_t.FAdd if sca_ty.is_floating_point_ty() else ir.atomic_rmw_op_t.Add
  return builder.create_atomic_rmw(op, ptr, val, mask)

def atomic_and( ptr, val, mask, builder):
  atom_red_typechecking(ptr, val, mask, builder)
  return builder.create_atomic_rmw(ir.atomic_rmw_op_t.And, ptr, val, mask)

def atomic_or( ptr, val, mask, builder):
  atom_red_typechecking(ptr, val, mask, builder)
  return builder.create_atomic_rmw(ir.atomic_rmw_op_t.Or, ptr, val, mask)

def atomic_xor( ptr, val, mask, builder):
  atom_red_typechecking(ptr, val, mask, builder)
  return builder.create_atomic_rmw(ir.atomic_rmw_op_t.Xor, ptr, val, mask)

def atomic_xchg( ptr, val, mask, builder):
  atom_red_typechecking(ptr, val, mask, builder)
  sca_ty = val.get_type().get_scalar_ty()
  return builder.create_atomic_rmw(ir.atomic_rmw_op_t.Xchg, ptr, val, mask)

#===----------------------------------------------------------------------===//
#                               Linear Algebra
#===----------------------------------------------------------------------===//

def dot(lhs, rhs, allow_tf32, builder):
  _0 = None
  if lhs.get_type().is_int_or_tileint_ty():
    _0 = builder.get_int32(0)
  else:
    _0 = builder.get_float32(0)
  M = lhs.get_type().get_block_shapes()[0]
  N = rhs.get_type().get_block_shapes()[1]
  _0 = builder.create_splat(_0, [M, N])
  _allow_tf32 = allow_tf32.get_value() != 0
  return builder.create_dot(lhs, rhs, _0, _allow_tf32)


#===----------------------------------------------------------------------===//
#                               Indexing
#===----------------------------------------------------------------------===//

def where( condition, x, y, builder):
  condition = cast(condition, builder.get_int1_ty(), builder)
  if condition.get_type().is_block_ty():
    x = broadcast(x, condition.get_type().get_block_shapes(), builder)
    y = broadcast(y, condition.get_type().get_block_shapes(), builder)
  
  x_ty = x.get_type().get_scalar_ty()
  y_ty = y.get_type().get_scalar_ty()
  ty = computation_type(x_ty, y_ty, div_or_mod=False)
  x = cast(x, ty, builder)
  y = cast(y, ty, builder)
  return builder.create_select(condition, x, y)


#===----------------------------------------------------------------------===//
#                               Reductions
#===----------------------------------------------------------------------===//

def reduce_impl(input, axis, builder, name,
                FLOAT_OP, INT_OP):
  scalar_ty = input.get_type().get_scalar_ty()
  # input is extended to 32-bits if necessary
  # this increases numerical accuracy and can be done pretty much for free
  # on GPUs
  if scalar_ty.is_integer_ty() and scalar_ty.get_integer_bitwidth() <= 32:
    input = cast(input, ir.type.get_int32_ty(scalar_ty.get_context()), builder)
  if scalar_ty.is_floating_point_ty():
    return builder.create_reduce(input, FLOAT_OP, axis)
  elif scalar_ty.is_integer_ty():
    return builder.create_reduce(input, INT_OP, axis)
  assert False

def min(input, axis, builder):
  return reduce_impl(input, axis, builder, "min", ir.reduce_inst.FMIN, ir.reduce_inst.MIN)

def max(input, axis, builder):
  return reduce_impl(input, axis, builder, "max", ir.reduce_inst.FMAX, ir.reduce_inst.MAX)

def sum(input, axis, builder):
  return reduce_impl(input, axis, builder, "sum", ir.reduce_inst.FADD, ir.reduce_inst.ADD)

def xor_sum(input, axis, builder):
  scalar_ty = input.get_type().get_scalar_ty()
  if not scalar_ty.is_integer_ty():
    raise SemanticError("xor_sum only supported for integers")
  return reduce_impl(input, axis, builder, "sum", ir.reduce_inst.XOR, ir.reduce_inst.XOR)


#===----------------------------------------------------------------------===//
#                               Math
#===----------------------------------------------------------------------===//

def umulhi(x,  y, builder):
  binary_op_type_checking(x, y, builder)
  return builder.insert(umulhi_inst.create(x, y))

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


