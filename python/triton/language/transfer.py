from __future__ import division, annotations

from typing import Optional

from triton import language as tl


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
