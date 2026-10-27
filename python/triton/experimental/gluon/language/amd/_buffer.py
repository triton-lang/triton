"""Shared implementation of the descriptor-based AMD buffer ops.

Architecture modules (``amd.cdna3.buffer``, ``amd.cdna4.buffer``,
``amd.gfx1250.buffer``) expose these as builtins with their own cache policy
types.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from triton._C.libtriton import ir
from triton.experimental.gluon.language import _core as ttgl
from .._core import _unwrap_if_constexpr
from .._layouts import DistributedLayout
from .._semantic import _check


@dataclass(eq=True)
class buffer_descriptor_type(ttgl.base_type):
    """The type of a buffer resource descriptor accessing ``element_ty`` elements."""

    element_ty: ttgl.dtype

    def __str__(self) -> str:
        return f"buffer_descriptor<{self.element_ty}>"

    def _to_ir(self, builder: ir.builder) -> ir.type:
        return builder.get_buffer_desc_type(self.element_ty.to_ir(builder))

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[buffer_descriptor, int]:
        return buffer_descriptor(handles[cursor], self), cursor + 1

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self._to_ir(builder))

    def mangle(self) -> str:
        return f"BD{self.element_ty.mangle()}BD"


@dataclass
class buffer_descriptor(ttgl.base_value):
    """
    A buffer resource descriptor (V#), made by ``make_buffer_descriptor``.

    It is a scalar, program-uniform value held in SGPRs, accepted by the ops of
    the ``buffer`` module.
    """

    handle: ir.value
    type: buffer_descriptor_type

    def _set_name(self, builder: ir.builder, name: str) -> None:
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    @property
    def dtype(self):
        return self.type.element_ty


def _scalar_operand(value, name, semantic, max_bits=32):
    """Convert an optional scalar integer operand to an i32 handle, or to an i64 one for a 64-bit value when
    `max_bits` allows it."""
    value = _unwrap_if_constexpr(value)
    if value is None:
        return ir.value()
    value = semantic.to_tensor(value)
    _check(not value.type.is_block(), lambda: f"{name} must be a scalar, got {value.type}")
    _check(value.dtype.is_int() and value.dtype.primitive_bitwidth <= max_bits,
           lambda: f"{name} must be an integer of at most {max_bits} bits, got {value.dtype}")
    if value.dtype.primitive_bitwidth > 32:
        return semantic.cast(value, ttgl.int64).handle
    return semantic.cast(value, ttgl.int32).handle


def _soffset_operand(soffset, offsets, semantic):
    """Convert an optional scalar or per-wave `soffset` to an i32 handle.

    A tensor must be broadcastable to `offsets` and hold the same value in every lane of a wave, so it
    can go in the instruction's scalar offset register.
    """
    soffset = _unwrap_if_constexpr(soffset)
    if soffset is None or not isinstance(soffset, ttgl.tensor) or not soffset.type.is_block():
        return _scalar_operand(soffset, "soffset", semantic)
    _check(soffset.dtype.is_int() and soffset.dtype.primitive_bitwidth <= 32,
           lambda: f"soffset must be an integer of at most 32 bits, got {soffset.dtype}")
    shape = offsets.shape
    _check(
        len(soffset.shape) == len(shape) and all(s in (1, n) for s, n in zip(soffset.shape, shape)),
        lambda: f"soffset of shape {soffset.shape} must broadcast to the offsets shape {shape}")
    # In soffset's own layout, all-zero lane bases mean every lane of a wave
    # holds the same value. Registers and warps may differ: each register element
    # is its own access, and each wave gets its own scalar offset.
    lane_bases = _unwrap_if_constexpr(semantic.to_linear_layout(soffset.type.layout, soffset.shape)).lane_bases
    _check(
        all(not any(basis) for basis in lane_bases),
        lambda: f"soffset must be the same across the lanes of a wave: give it size 1 in every dimension the "
        f"lanes of its layout span, got shape {soffset.shape}")
    soffset = semantic.cast(soffset, ttgl.int32)
    _, soffset = semantic.broadcast_impl_value(offsets, soffset)
    return soffset.handle


def _cache_policy(cache_policy, cache_policy_type, scope, semantic):
    """Return the `cachePolicy` attribute and the optional scope of an access."""
    cache_policy = _unwrap_if_constexpr(cache_policy)
    scope = _unwrap_if_constexpr(scope)
    if cache_policy is None:
        policy = semantic.builder.get_cache_policy("none", "evict_normal")
    else:
        _check(isinstance(cache_policy, cache_policy_type),
               lambda: f"cache_policy must be a {cache_policy_type.__qualname__}, got {cache_policy!r}")
        policy = cache_policy._to_ir(semantic.builder)
    if scope is not None:
        _check(scope in ("cta", "gpu", "sys"), lambda: f"scope must be 'cta', 'gpu' or 'sys', got {scope!r}")
        scope = semantic._str_to_scope(scope)
    return policy, scope


def _verify(desc, offsets):
    _check(isinstance(desc, buffer_descriptor), lambda: f"expected a buffer descriptor, got {desc!r}")
    _check(isinstance(offsets.type, ttgl.distributed_type), lambda: "expected offsets type to be a distributed_type")
    _check(offsets.dtype.is_int32() or offsets.dtype.is_uint32(),
           lambda: f"offsets element type must be int32 or uint32, got {offsets.dtype}")


def _broadcast_mask_other(offsets, mask, other, elem_ty, semantic):
    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    _check(other is None or mask is not None, lambda: "when other is not None, mask should not be None")
    if mask is not None:
        offsets, mask = semantic.broadcast_impl_value(offsets, mask)
    if other is not None:
        other = semantic.to_tensor(other)
        other = semantic.cast(other, elem_ty)
        offsets, other = semantic.broadcast_impl_value(offsets, other)
    return offsets, mask, other


def make_buffer_descriptor(base, num_records, semantic, num_records_bits=32):
    _check(base.type.is_ptr() and not base.type.is_block(), lambda: f"base must be a scalar pointer, got {base.type}")
    num_records = _scalar_operand(num_records, "num_records", semantic, max_bits=num_records_bits)
    handle = semantic.builder.create_make_buffer_desc(base.handle, num_records)
    return buffer_descriptor(handle, buffer_descriptor_type(base.type.element_ty))


def load(desc, offsets, mask, other, soffset, scope, cache_policy, cache_policy_type, semantic):
    _verify(desc, offsets)
    offsets, mask, other = _broadcast_mask_other(offsets, mask, other, desc.dtype, semantic)
    policy, scope = _cache_policy(cache_policy, cache_policy_type, scope, semantic)
    ret_ty = offsets.type.with_element_ty(desc.dtype)
    builder = semantic.builder
    handle = builder.create_buffer_desc_load(ret_ty.to_ir(builder), desc.handle, offsets.handle,
                                             mask.handle if mask is not None else ir.value(),
                                             other.handle if other is not None else ir.value(), policy,
                                             _soffset_operand(soffset, offsets, semantic), scope)
    return ttgl.tensor(handle, ret_ty)


def store(value, desc, offsets, mask, soffset, scope, cache_policy, cache_policy_type, semantic):
    _verify(desc, offsets)
    offsets_shape = offsets.shape
    mask = _unwrap_if_constexpr(mask)
    if mask is None:
        offsets, value = semantic.broadcast_tensors(offsets, value)
    else:
        offsets, value, mask = semantic.broadcast_tensors(offsets, value, mask)
    _check(offsets_shape == offsets.shape,
           lambda: f"Expected `offsets` argument to have shape {offsets.shape} but got {offsets_shape}")
    value = semantic.cast(value, desc.dtype)
    policy, scope = _cache_policy(cache_policy, cache_policy_type, scope, semantic)
    semantic.builder.create_buffer_desc_store(value.handle, desc.handle, offsets.handle,
                                              mask.handle if mask is not None else ir.value(), policy,
                                              _soffset_operand(soffset, offsets, semantic), scope)


def load_to_shared(dest, desc, offsets, mask, soffset, scope, cache_policy, cache_policy_type, semantic):
    _verify(desc, offsets)
    _check(isinstance(offsets.type.layout, DistributedLayout),
           lambda: "expected offsets type layout to be a DistributedLayout")
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        offsets, mask = semantic.broadcast_impl_value(offsets, mask)
    policy, scope = _cache_policy(cache_policy, cache_policy_type, scope, semantic)
    semantic.builder.create_buffer_desc_load_to_local(dest.handle, desc.handle,
                                                      offsets.handle, mask.handle if mask is not None else ir.value(),
                                                      ir.value(), ir.value(), policy,
                                                      _scalar_operand(soffset, "soffset", semantic), scope)


def _atomic_semantic(sem, default, allowed, op_name, semantic):
    sem = _unwrap_if_constexpr(sem)
    _check(sem is None or sem in allowed,
           lambda: f"{op_name} only supports {' and '.join(allowed)} semantics, got {sem!r}")
    return semantic._str_to_sem(sem, default=default)


def _atomic_scope(scope, semantic):
    scope = _unwrap_if_constexpr(scope)
    _check(scope is None or scope in ("cta", "gpu", "sys"),
           lambda: f"scope must be 'cta', 'gpu' or 'sys', got {scope!r}")
    return semantic._str_to_scope(scope)


def atomic_load(desc, offsets, mask, sem, scope, soffset, semantic):
    _verify(desc, offsets)
    offsets, mask, _ = _broadcast_mask_other(offsets, mask, None, desc.dtype, semantic)
    sem = _atomic_semantic(sem, ir.MEM_SEMANTIC.ACQUIRE, ("acquire", "relaxed"), "atomic_load", semantic)
    ret_ty = offsets.type.with_element_ty(desc.dtype)
    builder = semantic.builder
    handle = builder.create_buffer_atomic_load(ret_ty.to_ir(builder), desc.handle, offsets.handle,
                                               mask.handle if mask is not None else ir.value(),
                                               _soffset_operand(soffset, offsets, semantic), sem,
                                               _atomic_scope(scope, semantic))
    return ttgl.tensor(handle, ret_ty)


def atomic_store(value, desc, offsets, mask, sem, scope, soffset, semantic):
    _verify(desc, offsets)
    offsets_shape = offsets.shape
    mask = _unwrap_if_constexpr(mask)
    if mask is None:
        offsets, value = semantic.broadcast_tensors(offsets, value)
    else:
        offsets, value, mask = semantic.broadcast_tensors(offsets, value, mask)
    _check(offsets_shape == offsets.shape,
           lambda: f"Expected `offsets` argument to have shape {offsets.shape} but got {offsets_shape}")
    value = semantic.cast(value, desc.dtype)
    sem = _atomic_semantic(sem, ir.MEM_SEMANTIC.RELEASE, ("release", "relaxed"), "atomic_store", semantic)
    semantic.builder.create_buffer_atomic_store(value.handle, desc.handle, offsets.handle,
                                                mask.handle if mask is not None else ir.value(),
                                                _soffset_operand(soffset, offsets, semantic), sem,
                                                _atomic_scope(scope, semantic))
