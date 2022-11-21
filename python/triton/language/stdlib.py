from __future__ import division, annotations

from typing import TypeVar, Callable, List, Tuple, Optional

import triton
import triton.language as tl

CallableT = TypeVar("CallableT", bound=Callable)


def _globaltimer(builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_globaltimer, tl.int64)


@tl.builtin
def globaltimer(_builder: tl.ir.builder = None) -> tl.tensor:
    return _globaltimer(_builder)


def _clock(builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_clock(), tl.int64)


@tl.builtin
def clock(_builder: tl.ir.builder = None) -> tl.tensor:
    return _clock(_builder)


def _debug_barrier(builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_barrier(""), tl.void)


@tl.builtin
def debug_barrier(_builder: tl.ir.builder = None) -> tl.tensor:
    return _debug_barrier(_builder)


def _program_id(axis: int, builder: tl.ir.builder) -> tl.tensor:
    axis = tl._constexpr_to_value(axis)
    return tl.tensor(builder.create_get_program_id(axis), tl.int32)


@tl.builtin
def program_id(axis, _builder: tl.ir.builder = None) -> tl.tensor:
    """
    Returns the id of the current program instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
    """
    # if axis == -1:
    #     pid0 = program_id(0, _builder)
    #     pid1 = program_id(1, _builder)
    #     pid2 = program_id(2, _builder)
    #     npg0 = num_programs(0, _builder)
    #     npg1 = num_programs(0, _builder)
    #     return pid0 + pid1*npg0 + pid2*npg0*npg1
    axis = tl._constexpr_to_value(axis)
    return _program_id(axis, _builder)


def _num_programs(axis: int, builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_get_num_programs(axis), tl.int32)


@tl.builtin
def num_programs(axis, _builder: tl.ir.builder = None) -> tl.tensor:
    """
    Returns the number of program instances launched along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
    """
    axis = tl._constexpr_to_value(axis)
    return tl._num_programs(axis, _builder)


def _arange(start: int, end: int, builder: tl.ir.builder) -> tl.tensor:
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("arange's arguments must be of type tl.constexpr")

    shape = [end - start]
    ret_ty = tl.block_type(tl.int32, shape)
    return tl.tensor(builder.get_range(start, end), ret_ty)


@tl.builtin
def arange(start, end, _builder: tl.ir.builder = None) -> tl.tensor:
    """
    Returns contiguous values within the open interval [:code:`start`, :code:`end`).

    :param start: Start of the interval. Must be a power of two.
    :type start: int
    :param stop: End of the interval. Must be a power of two >= start.
    :type stop: int
    """
    start = tl._constexpr_to_value(start)
    end = tl._constexpr_to_value(end)
    return _arange(start, end, _builder)


def _zeros(
    shape: List[int],
    dtype: tl.dtype,
    builder: tl.ir.builder,
) -> tl.tensor:
    _0 = tl.ir.constant.get_null_value(dtype.to_ir(builder))
    ret_ty = tl.block_type(dtype, shape)
    return tl.tensor(builder.create_splat(_0, shape), ret_ty)


@tl.builtin
def zeros(
    shape,
    dtype: tl.dtype,
    _builder: tl.ir.builder = None,
) -> tl.tensor:
    """
    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`float16`
    :type dtype: DType
    """
    for i, d in enumerate(shape):
        if not isinstance(d, tl.constexpr):
            raise TypeError(f"Shape element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(
                f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]"
            )
    shape = [x.value for x in shape]
    dtype = tl._constexpr_to_value(dtype)
    return tl._zeros(shape, dtype, _builder)


@triton.jit
def zeros_like(input: tl.tensor) -> tl.tensor:
    return zeros(input.shape, input.dtype)


def _dequantize(
    input: tl.tensor,
    scale: tl.tensor,
    shift: tl.tensor,
    nbit: int,
    dst_ty: tl.dtype,
    builder: tl.ir.builder,
) -> tl.tensor:
    input_ty = input.type
    assert isinstance(input_ty, tl.block_type)
    assert input_ty.element_ty.is_int32() or input_ty.element_ty.is_int16()
    assert nbit in [2, 4, 8]
    assert dst_ty == tl.float16

    shape = input_ty.get_block_shapes()
    factor = input_ty.element_ty.primitive_bitwidth // nbit
    dst_shape = shape[:-1] + (factor * shape[-1],)

    dst_ty = tl.block_type(dst_ty, dst_shape)
    return tl.tensor(
        builder.create_dequantize(
            input.handle,
            scale.handle,
            shift.handle,
            dst_ty.to_ir(builder),
        ),
        dst_ty,
    )


@tl.builtin
def dequantize(
    input: tl.tensor,
    scale: tl.tensor,
    shift: tl.tensor,
    nbit: int,
    dst_ty: tl.dtype = tl.float16,
    _builder: tl.ir.builder = None,
):
    """
    Tries to dequantize the input to given dtype
    """
    nbit = tl._constexpr_to_value(nbit)
    return _dequantize(input, scale, shift, nbit, dst_ty, _builder)


@tl.builtin
def broadcast(
    input: tl.tensor,
    other: tl.tensor,
    _builder: tl.ir.builder = None,
) -> Tuple[tl.tensor, tl.tensor]:
    """
    Tries to broadcast the two given blocks to a common compatible shape.

    :param input: The first input tensor.
    :type input: Block
    :param other: The second input tensor.
    :type other: Block
    """
    return tl._broadcast_impl_value(input, other, _builder)


@tl.builtin
def broadcast_to(input: tl.tensor, shape, _builder: tl.ir.builder = None) -> tl.tensor:
    """
    Tries to broadcast the given tensor to a new :code:`shape`.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.
    :type shape: Tuple[int]
    """
    return tl._broadcast_impl_shape(input, shape, _builder)


def _cat(
    lhs: tl.tensor,
    rhs: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    assert lhs.type.is_block() and rhs.type.is_block()
    assert lhs.type.shape[1:] == rhs.type.shape[1:]
    ret_shape = [lhs.type.shape[0] + rhs.type.shape[0]]
    ret_ty = tl.block_type(lhs.type.scalar, ret_shape)
    return tl.tensor(
        builder.create_cat(lhs.handle, rhs.handle),
        ret_ty,
    )


@tl.builtin
def cat(
    input: tl.tensor,
    other: tl.tensor,
    _builder: tl.ir.builder = None,
) -> tl.tensor:
    """
    Concatenate the given blocks

    :param input: The first input tensor.
    :type input:
    :param other: The second input tensor.
    :type other:
    """
    return _cat(input, other, _builder)


@tl.builtin
def reshape(
    input: tl.tensor,
    shape,
    _builder: tl.ir.builder = None,
) -> tl.tensor:
    """
    Tries to reshape the given tensor to a new shape.

    :param input: The input tensor.
    :type input:
    :param shape: The desired shape.
    :type shape: Tuple[int]

    """
    shape = [x.value for x in shape]
    return tl._reshape(input, shape, _builder)


def _dot(
    a: tl.tensor,
    b: tl.tensor,
    trans_a: bool,
    trans_b: bool,
    allow_tf32: bool,
    builder: tl.ir.builder,
) -> tl.tensor:
    in_a = 1 if not trans_a else 0
    in_b = 1 if trans_b else 0
    assert a.type.is_block() and b.type.is_block()
    assert len(a.shape) == 2 and len(b.shape) == 2
    assert a.shape[in_a] == b.shape[in_b]
    assert (
        a.shape[0] >= 16 and a.shape[1] >= 16 and b.shape[1] >= 16
    ), "small blocks not supported!"
    if a.type.scalar.is_int():
        _0 = builder.get_int32(0)
        ret_scalar_ty = tl.int32
    else:
        _0 = builder.get_float32(0)
        ret_scalar_ty = tl.float32
    M = a.type.shape[in_a ^ 1]
    N = b.type.shape[in_b ^ 1]
    _0 = builder.create_splat(_0, [M, N])
    ret_ty = tl.block_type(ret_scalar_ty, [M, N])
    ret = builder.create_dot(
        a.handle,
        b.handle,
        _0,
        trans_a,
        trans_b,
        allow_tf32,
    )
    return tl.tensor(ret, ret_ty)


@tl.builtin
def dot(
    input,
    other,
    trans_a=False,
    trans_b=False,
    allow_tf32=True,
    _builder: tl.ir.builder = None,
):
    """
    Returns the matrix product of two blocks.

    The two blocks must be two dimensionals and have compatible inner dimensions.

    :param input: The first tensor to be multiplied.
    :type input: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param other: The second tensor to be multiplied.
    :type other: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    """
    allow_tf32 = tl._constexpr_to_value(allow_tf32)
    return _dot(input, other, trans_a, trans_b, allow_tf32, _builder)


def _parse_eviction_policy(eviction_policy) -> tl.ir.EVICTION_POLICY:
    eviction = tl.ir.EVICTION_POLICY.NORMAL  # default
    if eviction_policy:
        if eviction_policy == "evict_last":
            eviction = tl.ir.EVICTION_POLICY.EVICT_LAST
        elif eviction_policy == "evict_first":
            eviction = tl.ir.EVICTION_POLICY.EVICT_FIRST
        else:
            raise ValueError(f"Eviction policy {eviction_policy} not supported")
    return eviction


def _load(
    ptr: tl.tensor,
    mask: Optional[tl.tensor],
    other: Optional[tl.tensor],
    cache_modifier: str,
    eviction_policy: str,
    is_volatile: bool,
    builder: tl.ir.builder,
) -> tl.tensor:
    if not ptr.type.scalar.is_ptr():
        raise ValueError(
            "Pointer argument of load instruction is " + ptr.type.__repr__()
        )
    if ptr.type.is_block():
        if mask:
            mask = tl._broadcast_impl_shape(
                mask,
                ptr.type.get_block_shapes(),
                builder,
            )
        if other:
            other = tl._broadcast_impl_shape(
                other,
                ptr.type.get_block_shapes(),
                builder,
            )

    if other:
        other = tl._cast(other, ptr.type.scalar.element_ty, builder)
    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty
    # treat bool* as int8*
    if elt_ty == tl.int1:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = tl._cast(ptr, ptr_ty, builder)

    # cache modifier
    cache = tl.ir.CACHE_MODIFIER.NONE  # default
    if cache_modifier:
        if cache_modifier == ".ca":
            cache = tl.ir.CACHE_MODIFIER.CA
        elif cache_modifier == ".cg":
            cache = tl.ir.CACHE_MODIFIER.CG
        else:
            raise ValueError(f"Cache modifier {cache_modifier} not supported")

    # eviction policy
    eviction = _parse_eviction_policy(eviction_policy)

    if ptr.type.is_block():
        shape = ptr.type.get_block_shapes()
        dst_ty = tl.block_type(elt_ty, shape)
    else:
        dst_ty = elt_ty

    if not mask and not other:
        return tl.tensor(
            builder.create_load(
                ptr.handle,
                cache,
                eviction,
                is_volatile,
            ),
            dst_ty,
        )
    if not mask:
        raise ValueError("`other` cannot be provided without `mask`")

    if not other:
        other_ir = tl.ir.undef.get(elt_ty.to_ir(builder))
        if ptr.type.is_block():
            other_ir = builder.create_splat(
                other_ir,
                ptr.type.get_block_shapes(),
            )
        other = tl.tensor(other_ir, dst_ty)

    return tl.tensor(
        builder.create_masked_load(
            ptr.handle,
            mask.handle,
            other.handle,
            cache,
            eviction,
            is_volatile,
        ),
        dst_ty,
    )


@tl.builtin
def load(
    pointer,
    mask=None,
    other=None,
    cache_modifier="",
    eviction_policy="",
    volatile=False,
    _builder: tl.ir.builder = None,
) -> tl.tensor:
    """
    Return a tensor of data whose values are, elementwise, loaded from memory at location defined by :code:`pointer`.

    :param *:
    :code:`mask` and :code:`other` are implicitly broadcast to :code:`pointer.shape`.

    :code:`other` is implicitly typecast to :code:`pointer.dtype.element_ty`.

    :param pointer: Pointers to the data to be loaded.
    :type pointer: Block of dtype=tr.PointerDType
    :param mask: if mask[idx] is false, do not load the data at address :code:`pointer[idx]`.
    :type mask: Block of tr.int1, optional
    :param other: if mask[idx] is false, return other[idx]
    :type other: Block, optional
    :param cache_modifier: changes cache option in nvidia ptx
    'type cache_modifier: str, optional
    """
    # mask, other can be tl.constexpr
    if mask is not None:
        mask = tl._to_tensor(mask, _builder)
    if other is not None:
        other = tl._to_tensor(other, _builder)
    cache_modifier = tl._constexpr_to_value(cache_modifier)
    eviction_policy = tl._constexpr_to_value(eviction_policy)
    volatile = tl._constexpr_to_value(volatile)
    return _load(
        pointer,
        mask,
        other,
        cache_modifier,
        eviction_policy,
        volatile,
        _builder,
    )


def _store(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: Optional[tl.tensor],
    eviction_policy: str,
    builder: tl.ir.builder,
) -> tl.tensor:
    if not ptr.type.scalar.is_ptr():
        raise ValueError(
            "Pointer argument of store instruction is " + ptr.type.__repr__()
        )
    if ptr.type.is_block():
        val = tl._broadcast_impl_shape(
            val,
            ptr.type.get_block_shapes(),
            builder,
        )
    if mask:
        mask = tl._broadcast_impl_shape(
            mask,
            ptr.type.get_block_shapes(),
            builder,
        )
    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty
    # treat bool* as int8*
    if elt_ty == tl.int1:
        # convert to bool first and then store as int8
        val = tl._cast(val, tl.int1, builder)
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = tl._cast(ptr, ptr_ty, builder)
    # eviction policy
    eviction = _parse_eviction_policy(eviction_policy)
    # cast to target data-type
    val = tl._cast(val, elt_ty, builder)
    if not mask:
        return tl.tensor(
            builder.create_store(ptr.handle, val.handle, eviction),
            tl.void,
        )
    if not mask.type.scalar.is_bool():
        raise ValueError("Mask must have boolean scalar type")
    return tl.tensor(
        builder.create_masked_store(
            ptr.handle,
            val.handle,
            mask.handle,
            eviction,
        ),
        tl.void,
    )


@tl.builtin
def store(
    pointer,
    value,
    mask=None,
    eviction_policy="",
    _builder: tl.ir.builder = None,
) -> tl.tensor:
    """
    Stores :code:`value` tensor of elements in memory, element-wise, at the memory locations specified by :code:`pointer`.

    :param *:
    :code:`value` is implicitly broadcast to :code:`pointer.shape` and typecast to :code:`pointer.dtype.element_ty`.

    :param pointer: The memory locations where the elements of :code:`value` are stored.
    :type pointer: Block of dtype=tr.PointerDType
    :param value: The tensor of elements to be stored.
    :type value: Block
    :param mask: If mask[idx] is false, do not store :code:`value[idx]` at :code:`pointer[idx]`.
    :type mask: Block of tr.int1, optional
    """
    # value can be tl.constexpr
    value = tl._to_tensor(value, _builder)
    if mask is not None:
        mask = tl._to_tensor(mask, _builder)
    return tl._store(pointer, value, mask, eviction_policy, _builder)


def _atom_red_typechecking_impl(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: tl.tensor,
    op: str,
    builder: tl.ir.builder,
) -> Tuple[tl.tensor, tl.tensor, tl.tensor]:
    if not ptr.type.scalar.is_ptr():
        raise ValueError(
            "Pointer argument of store instruction is " + ptr.type.__repr__()
        )

    element_ty = ptr.type.scalar.element_ty
    if element_ty is tl.float16 and op != "add":
        raise ValueError("atomic_" + op + " does not support fp16")
    if element_ty in [tl.int1, tl.int8, tl.int16, tl.bfloat16]:
        raise ValueError("atomic_" + op + " does not support " + element_ty)
    if ptr.type.is_block():
        if mask:
            mask = tl._broadcast_impl_shape(
                mask,
                ptr.type.get_block_shapes(),
                builder,
            )
        if val:
            val = tl._broadcast_impl_shape(
                val,
                ptr.type.get_block_shapes(),
                builder,
            )
    val = tl._cast(val, ptr.type.scalar.element_ty, builder)
    if not mask:
        mask_ir = builder.get_int1(True)
        mask_ty = tl.int1
        if ptr.type.is_block():
            mask_ir = builder.create_splat(mask_ir, ptr.type.get_block_shapes())
            mask_ty = tl.block_type(tl.int1, ptr.type.get_block_shapes())
        mask = tl.tensor(mask_ir, mask_ty)
    return ptr, val, mask


def _atomic_cas(
    ptr: tl.tensor,
    cmp: tl.tensor,
    val: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    element_ty = ptr.type.scalar.element_ty
    if element_ty.primitive_bitwidth not in [16, 32, 64]:
        raise ValueError("atomic_cas only supports elements with width {16, 32, 64}")
    return tl.tensor(
        builder.create_atomic_cas(ptr.handle, cmp.handle, val.handle),
        val.type,
    )


@tl.builtin
def atomic_cas(pointer, cmp, val, _builder: tl.ir.builder = None) -> tl.tensor:
    """
    Performs an atomic compare-and-swap at the memory location specified by :code:`pointer`.

    Return the data stored at :code:`pointer` before the atomic operation.

    :param pointer: The memory locations to compare-and-swap.
    :type pointer: Block of dtype=tr.PointerDType
    :param cmp: The values expected to be found in the atomic object
    :type cmp: Block of dtype=`pointer.dtype.element_ty`
    :param val: The values to copy in case the expected value matches the contained value.
    :type val: Block of dtype=`pointer.dtype.element_ty`
    """
    cmp = tl._to_tensor(cmp, _builder)
    val = tl._to_tensor(val, _builder)
    return _atomic_cas(pointer, cmp, val, _builder)


def _add_atomic_docstr(name: str) -> Callable[[CallableT], CallableT]:
    def _decorator(func: CallableT) -> CallableT:
        docstr = """
    Performs an atomic {name} at the memory location specified by :code:`pointer`.

    Return the data stored at :code:`pointer` before the atomic operation.

    :param pointer: The memory locations to apply {name}.
    :type pointer: Block of dtype=tr.PointerDType
    :param val: The values to {name} in the atomic object.
    :type val: Block of dtype=`pointer.dtype.element_ty`
    :param mask: If mask[idx] is false, do not apply {name}.
    :type mask: Block of tr.int1, optional
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _atomic_xchg(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    ptr, val, mask = _atom_red_typechecking_impl(
        ptr,
        val,
        mask,
        "xchg",
        builder,
    )
    return tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.XCHG,
            ptr.handle,
            val.handle,
            mask.handle,
        ),
        val.type,
    )


@tl.builtin
@_add_atomic_docstr("exchange")
def atomic_xchg(
    pointer,
    val,
    mask=None,
    _builder: tl.ir.builder = None,
) -> tl.tensor:
    val = tl._to_tensor(val, _builder)
    return _atomic_xchg(pointer, val, mask, _builder)


def _atomic_add(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "add", builder)
    sca_ty = val.type.scalar
    op = tl.ir.ATOMIC_OP.FADD if sca_ty.is_floating() else tl.ir.ATOMIC_OP.ADD
    return tl.tensor(
        builder.create_atomic_rmw(
            op,
            ptr.handle,
            val.handle,
            mask.handle,
        ),
        val.type,
    )


@tl.builtin
@_add_atomic_docstr("add")
def atomic_add(pointer, val, mask=None, _builder: tl.ir.builder = None) -> tl.tensor:
    val = tl._to_tensor(val, _builder)
    return _atomic_add(pointer, val, mask, _builder)


def _atomic_max(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "max", builder)
    sca_ty = val.type.scalar
    # direct call to atomic_max for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(
                builder.create_atomic_rmw(
                    tl.ir.ATOMIC_OP.MAX,
                    ptr.handle,
                    val.handle,
                    mask.handle,
                ),
                val.type,
            )
        else:
            return tl.tensor(
                builder.create_atomic_rmw(
                    tl.ir.ATOMIC_OP.UMAX,
                    ptr.handle,
                    val.handle,
                    mask.handle,
                ),
                val.type,
            )
    # for float
    # return atomic_smax(i_ptr, i_val) if val >= 0
    # return atomic_umin(i_ptr, i_val) if val < 0
    i_val = tl._bitcast(val, tl.int32, builder)
    i_ptr = tl._bitcast(
        ptr,
        tl.pointer_type(tl.int32, 1),
        builder,
    )
    pos = tl._greater_equal(
        val,
        tl.tensor(
            tl.ir.constant_float.get(sca_ty.to_ir(builder), 0),
            sca_ty,
        ),
        builder,
    )
    neg = tl._less_than(
        val,
        tl.tensor(
            tl.ir.constant_float.get(sca_ty.to_ir(builder), 0),
            sca_ty,
        ),
        builder,
    )
    pos_ret = tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.MAX,
            i_ptr.handle,
            i_val.handle,
            tl._and_(mask, pos, builder).handle,
        ),
        i_val.type,
    )
    neg_ret = tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.UMIN,
            i_ptr.handle,
            i_val.handle,
            tl._and_(mask, neg, builder).handle,
        ),
        i_val.type,
    )
    return tl._where(pos, pos_ret, neg_ret, builder)


@tl.builtin
@_add_atomic_docstr("max")
def atomic_max(pointer, val, mask=None, _builder: tl.ir.builder = None) -> tl.tensor:
    val = tl._to_tensor(val, _builder)
    return _atomic_max(pointer, val, mask, _builder)


def _atomic_min(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "min", builder)
    sca_ty = val.type.scalar
    # direct call to atomic_min for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(
                builder.create_atomic_rmw(
                    tl.ir.ATOMIC_OP.MIN,
                    ptr.handle,
                    val.handle,
                    mask.handle,
                ),
                val.type,
            )
        else:
            return tl.tensor(
                builder.create_atomic_rmw(
                    tl.ir.ATOMIC_OP.UMIN,
                    ptr.handle,
                    val.handle,
                    mask.handle,
                ),
                val.type,
            )
    # for float
    # return atomic_smin(i_ptr, i_val) if val >= 0
    # return atomic_umax(i_ptr, i_val) if val < 0
    i_val = tl._bitcast(val, tl.int32, builder)
    i_ptr = tl._bitcast(
        ptr,
        tl.pointer_type(tl.int32, 1),
        builder,
    )
    pos = tl._greater_equal(
        val,
        tl.tensor(
            tl.ir.constant_float.get(sca_ty.to_ir(builder), 0),
            sca_ty,
        ),
        builder,
    )
    neg = tl._less_than(
        val,
        tl.tensor(
            tl.ir.constant_float.get(sca_ty.to_ir(builder), 0),
            sca_ty,
        ),
        builder,
    )
    pos_ret = tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.MIN,
            i_ptr.handle,
            i_val.handle,
            tl._and_(mask, pos, builder).handle,
        ),
        i_val.type,
    )
    neg_ret = tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.UMAX,
            i_ptr.handle,
            i_val.handle,
            tl._and_(mask, neg, builder).handle,
        ),
        i_val.type,
    )
    return tl._where(pos, pos_ret, neg_ret, builder)


@tl.builtin
@_add_atomic_docstr("min")
def atomic_min(pointer, val, mask=None, _builder: tl.ir.builder = None) -> tl.tensor:
    val = tl._to_tensor(val, _builder)
    return _atomic_min(pointer, val, mask, _builder)


def _atomic_and(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "and", builder)
    return tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.AND,
            ptr.handle,
            val.handle,
            mask.handle,
        ),
        val.type,
    )


@tl.builtin
@_add_atomic_docstr("logical and")
def atomic_and(pointer, val, mask=None, _builder: tl.ir.builder = None) -> tl.tensor:
    val = tl._to_tensor(val, _builder)
    return _atomic_and(pointer, val, mask, _builder)


def _atomic_or(
    ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, builder: tl.ir.builder
) -> tl.tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "or", builder)
    return tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.OR,
            ptr.handle,
            val.handle,
            mask.handle,
        ),
        val.type,
    )


@tl.builtin
@_add_atomic_docstr("logical or")
def atomic_or(pointer, val, mask=None, _builder: tl.ir.builder = None) -> tl.tensor:
    val = tl._to_tensor(val, _builder)
    return _atomic_or(pointer, val, mask, _builder)


def _atomic_xor(
    ptr: tl.tensor,
    val: tl.tensor,
    mask: tl.tensor,
    builder: tl.ir.builder,
) -> tl.tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "xor", builder)
    return tl.tensor(
        builder.create_atomic_rmw(
            tl.ir.ATOMIC_OP.XOR,
            ptr.handle,
            val.handle,
            mask.handle,
        ),
        val.type,
    )


@tl.builtin
@_add_atomic_docstr("logical xor")
def atomic_xor(pointer, val, mask=None, _builder: tl.ir.builder = None) -> tl.tensor:
    val = tl._to_tensor(val, _builder)
    return _atomic_xor(pointer, val, mask, _builder)


def _umulhi(x: tl.tensor, y: tl.tensor, builder: tl.ir.builder) -> tl.tensor:
    x, y = tl._binary_op_type_checking_impl(x, y, builder=builder)
    return tl.tensor(
        builder.create_umulhi(x.handle, y.handle),
        x.type,
    )


@tl.builtin
def umulhi(x, y, _builder: tl.ir.builder = None) -> tl.tensor:
    x = tl._to_tensor(x, _builder)
    y = tl._to_tensor(y, _builder)
    return _umulhi(x, y, _builder)


@tl.builtin
def fdiv(x, y, ieee_rounding=False, _builder: tl.ir.builder = None) -> tl.tensor:
    ieee_rounding = tl._constexpr_to_value(ieee_rounding)
    return tl._fdiv(x, y, ieee_rounding, _builder)


def _add_math_1arg_docstr(name: str) -> Callable[[CallableT], CallableT]:
    def _decorator(func: CallableT) -> CallableT:
        docstr = """
    Computes the element-wise {name} of :code:`x`

    :param x: the input values
    :type x: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _exp(x: tl.tensor, builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_exp(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("exponential")
def exp(x, _builder: tl.ir.builder = None) -> tl.tensor:
    return _exp(x, _builder)


def _log(x: tl.tensor, builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_log(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("natural logarithm")
def log(x, _builder: tl.ir.builder = None) -> tl.tensor:
    return _log(x, _builder)


def _cos(x: tl.tensor, builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_cos(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("cosine")
def cos(x, _builder: tl.ir.builder = None) -> tl.tensor:
    return _cos(x, _builder)


def _sin(x: tl.tensor, builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_sin(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("sine")
def sin(x, _builder: tl.ir.builder = None) -> tl.tensor:
    return _sin(x, _builder)


def _sqrt(x: tl.tensor, builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_sqrt(x.handle), x.type)


@tl.builtin
@_add_math_1arg_docstr("square root")
def sqrt(x, _builder: tl.ir.builder = None) -> tl.tensor:
    return _sqrt(x, _builder)


def _add_reduction_docstr(name: str) -> Callable[[CallableT], CallableT]:
    def _decorator(func: CallableT) -> CallableT:
        docstr = """
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    :param input: the input values
    :param axis: the dimension along which the reduction should be done
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _reduce_impl(
    input: tl.tensor,
    axis: int,
    builder: tl.ir.builder,
    name: str,
    FLOAT_OP: tl.ir.REDUCE_OP,
    INT_OP: tl.ir.REDUCE_OP,
) -> tl.tensor:
    scalar_ty = input.type.scalar
    # input is extended to 32-bits if necessary
    # this increases numerical accuracy and can be done pretty much for free
    # on GPUs
    if scalar_ty.is_int() and scalar_ty.int_bitwidth <= 32:
        input = tl._cast(input, tl.int32, builder)

    # hardware doesn't support FMAX, FMIN, CMP for bfloat16
    if scalar_ty is tl.bfloat16:
        input = tl._cast(input, tl.float32, builder)

    # choose the right unsigned operation
    if scalar_ty.is_int_unsigned():
        int_op_to_unit = {
            tl.ir.REDUCE_OP.MIN: tl.ir.REDUCE_OP.UMIN,
            tl.ir.REDUCE_OP.MAX: tl.ir.REDUCE_OP.UMAX,
            tl.ir.REDUCE_OP.ARGMIN: tl.ir.REDUCE_OP.ARGUMIN,
            tl.ir.REDUCE_OP.ARGMAX: tl.ir.REDUCE_OP.ARGUMAX,
        }
        if INT_OP in int_op_to_unit:
            INT_OP = int_op_to_unit[INT_OP]

    # get result type
    shape = input.type.shape
    ret_shape = []
    for i, s in enumerate(shape):
        if i != axis:
            ret_shape.append(s)
    if len(ret_shape) == 0:
        res_ty = scalar_ty
    else:
        res_ty = tl.block_type(scalar_ty, ret_shape)

    if scalar_ty.is_floating():
        return tl.tensor(
            builder.create_reduce(
                input.handle,
                FLOAT_OP,
                axis,
            ),
            res_ty,
        )
    elif scalar_ty.is_int():
        return tl.tensor(
            builder.create_reduce(
                input.handle,
                INT_OP,
                axis,
            ),
            res_ty,
        )
    assert False


def _max(input: tl.tensor, axis: int, builder: tl.ir.builder) -> tl.tensor:
    return _reduce_impl(
        input,
        axis,
        builder,
        "max",
        tl.ir.REDUCE_OP.FMAX,
        tl.ir.REDUCE_OP.MAX,
    )


@tl.builtin
@_add_reduction_docstr("maximum")
def max(input, axis, _builder: tl.ir.builder = None) -> tl.tensor:
    axis = tl._constexpr_to_value(axis)
    return _max(input, axis, _builder)


def _argmax(input: tl.tensor, axis: int, builder: tl.ir.builder) -> tl.tensor:
    return _reduce_impl(
        input,
        axis,
        builder,
        "argmax",
        tl.ir.REDUCE_OP.ARGFMAX,
        tl.ir.REDUCE_OP.ARGMAX,
    )


@tl.builtin
@_add_reduction_docstr("maximum index")
def argmax(input, axis, _builder: tl.ir.builder = None) -> tl.tensor:
    axis = tl._constexpr_to_value(axis)
    return _argmax(input, axis, _builder)


def _min(input: tl.tensor, axis: int, builder: tl.ir.builder) -> tl.tensor:
    return _reduce_impl(
        input,
        axis,
        builder,
        "min",
        tl.ir.REDUCE_OP.FMIN,
        tl.ir.REDUCE_OP.MIN,
    )


@tl.builtin
@_add_reduction_docstr("minimum")
def min(input, axis, _builder: tl.ir.builder = None) -> tl.tensor:
    axis = tl._constexpr_to_value(axis)
    return _min(input, axis, _builder)


def _argmin(input: tl.tensor, axis: int, builder: tl.ir.builder) -> tl.tensor:
    return _reduce_impl(
        input,
        axis,
        builder,
        "argmin",
        tl.ir.REDUCE_OP.ARGFMIN,
        tl.ir.REDUCE_OP.ARGMIN,
    )


@tl.builtin
@_add_reduction_docstr("minimum index")
def argmin(input, axis, _builder: tl.ir.builder = None) -> tl.tensor:
    axis = tl._constexpr_to_value(axis)
    return _argmin(input, axis, _builder)


def _sum(input: tl.tensor, axis: int, builder: tl.ir.builder) -> tl.tensor:
    return _reduce_impl(
        input,
        axis,
        builder,
        "sum",
        tl.ir.REDUCE_OP.FADD,
        tl.ir.REDUCE_OP.ADD,
    )


@tl.builtin
@_add_reduction_docstr("sum")
def sum(input, axis, _builder: tl.ir.builder = None) -> tl.tensor:
    axis = tl._constexpr_to_value(axis)
    return _sum(input, axis, _builder)


def _xor_sum(input: tl.tensor, axis: int, builder: tl.ir.builder) -> tl.tensor:
    scalar_ty = input.type.scalar
    if not scalar_ty.is_int():
        raise ValueError("xor_sum only supported for integers")
    return _reduce_impl(
        input,
        axis,
        builder,
        "sum",
        tl.ir.REDUCE_OP.XOR,
        tl.ir.REDUCE_OP.XOR,
    )


@tl.builtin
@_add_reduction_docstr("xor sum")
def xor_sum(input, axis, _builder: tl.ir.builder = None) -> tl.tensor:
    axis = tl._constexpr_to_value(axis)
    return _xor_sum(input, axis, _builder)


def _multiple_of(x: tl.tensor, values: List[int]) -> tl.tensor:
    if len(x.shape) != len(values):
        raise ValueError(
            "Shape of input to multiple_of does not match the length of values"
        )
    x.handle.multiple_of(values)
    return x


@tl.builtin
def multiple_of(input, values, _builder: tl.ir.builder = None) -> tl.tensor:
    """
    Let the compiler knows that the values in :code:`input` are all multiples of :code:`value`.
    """
    if isinstance(values, tl.constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, tl.constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(
                f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]"
            )
    values = [x.value for x in values]
    return _multiple_of(input, values)


def _max_contiguous(x: tl.tensor, values: List[int]) -> tl.tensor:
    if len(x.shape) != len(values):
        raise ValueError(
            "Shape of input to max_contiguous does not match the length of values"
        )
    x.handle.max_contiguous(values)
    return x


@tl.builtin
def max_contiguous(input, values, _builder: tl.ir.builder = None) -> tl.tensor:
    """
    Let the compiler knows that the `value` first values in :code:`input` are contiguous.
    """
    if isinstance(values, tl.constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, tl.constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(
                f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]"
            )
    values = [x.value for x in values]
    return _max_contiguous(input, values)


@triton.jit
def abs(x) -> tl.tensor:
    return tl.where(x >= 0, x, -x)


@triton.jit
def cdiv(x, div) -> tl.tensor:
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type input: Block
    :param div: the divisor
    :param div: Block
    """
    return (x + div - 1) // div


@triton.jit
def maximum(x, y) -> tl.tensor:
    """
    Computes the element-wise maximum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return tl.where(x > y, x, y)


# minimum is in trition.core


@triton.jit
@_add_math_1arg_docstr("sigmoid")
def sigmoid(x) -> tl.tensor:
    return 1 / (1 + exp(-x))


@triton.jit
@_add_math_1arg_docstr("softmax")
def softmax(x, ieee_rounding: tl.constexpr = False) -> tl.tensor:
    z = x - max(x, 0)
    num = exp(z)
    den = sum(num, 0)
    return fdiv(num, den, ieee_rounding)


@triton.jit
def ravel(x: tl.tensor) -> tl.tensor:
    """
    Returns a contiguous flattened view of :code:`x`

    :param x: the input tensor
    :type x: Block
    """
    return reshape(x, [x.numel])


@triton.jit
def swizzle2d(
    i: int,
    j: int,
    size_i: int,
    size_j: int,
    size_g: int,
) -> Tuple[int, int]:
    """
    transformes indices of a row-major size_i*size_j matrix into those
    of one where indices are row major for each group of size_j rows.
    For example, for size_i = size_j = 4 and size_g = 2, it will transform
    [[0 , 1 , 2 , 3 ],
     [4 , 5 , 6 , 7 ],
     [8 , 9 , 10, 11],
     [12, 13, 14, 15]]
    into
    [[0, 2,  4 , 6 ],
     [1, 3,  5 , 7 ],
     [8, 10, 12, 14],
     [9, 11, 13, 15]]
    """
    # "unrolled index in array"
    ij = i * size_j + j
    # number of elements in `size_g` groups
    # of `size_j` columns
    size_gj = size_g * size_j
    # index of the group in which (i,j) is
    group_id = ij // size_gj
    # row-index of the first element of this group
    off_i = group_id * size_g
    # last group may have fewer rows
    size_g = tl.minimum(size_i - off_i, size_g)
    # new row and column indices
    new_i = off_i + (ij % size_g)
    new_j = (ij % size_gj) // size_g
    return new_i, new_j
