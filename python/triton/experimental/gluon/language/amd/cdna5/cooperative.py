"""
Cooperative 128-byte atomic memory ops for CDNA5.

A line is a naturally aligned, contiguous 128-byte block of global memory. A group of 32, 16, or 8 consecutive
lanes of a wave32 loads or stores one line as a single atomic access, so a line is never observed torn. Like
:func:`~triton.experimental.gluon.language.amd.cdna5.buffer_load`, the ops address global memory through a scalar
base pointer and a tensor of element offsets.
"""

from __future__ import annotations

from triton._C.libtriton import ir
from ..._core import builtin, distributed_type, tensor, _unwrap_if_constexpr
from ..._semantic import _check
from .._buffer import _atomic_scope, _atomic_semantic

__all__ = ["load", "store"]

_ACCESS_DOC = """The offsets layout decides the access: each lane accesses its register-contiguous elements, up
    to 16 bytes, and 128 / that many consecutive lanes form each 128-byte line, in the dense row-major order of
    ``offsets``. Each lane is addressed by its first offset in register order; the remaining offsets are assumed
    to follow contiguously and are not read. Each line must be contiguous, 128-byte
    aligned and backed by storage, and byte offsets must be below 2**32; these address requirements are not
    checked.

    ``mask`` must be provably constant over every line; a masked-off line is not accessed.

    A line synchronizes only with cooperative atomics of the same 128-byte size, and is only valid in global
    memory reached over a fabric that supports 128-byte requests (not host memory over PCIe)."""


def _format_doc(fn):
    """Fill the shared access description into `fn`'s docstring."""
    fn.__doc__ = fn.__doc__.format(access=_ACCESS_DOC)
    return fn


def _verify(ptr, offsets):
    _check(ptr.type.is_ptr() and not ptr.type.is_block(), lambda: f"ptr must be a scalar pointer, got {ptr.type}")
    _check(isinstance(offsets.type, distributed_type), lambda: "expected offsets type to be a distributed_type")
    _check(offsets.dtype.is_int32() or offsets.dtype.is_uint32(),
           lambda: f"offsets element type must be int32 or uint32, got {offsets.dtype}")


@builtin
@_format_doc
def load(ptr, offsets, mask=None, other=None, *, sem=None, scope=None, _semantic=None):
    """
    Atomically load 128-byte lines from global memory into registers.

    Each 128-byte line is one atomic snapshot, consistent with every cooperative store to it; different lines
    are independent snapshots. Masked-off elements read ``other``, or zero without it.

    To poll a flag, use ``sem="relaxed"``: the flag and the data in its line arrive in one snapshot, while
    ``"acquire"`` adds a fence and CTA barriers on every call. If later reads of other memory depend on the
    flag, the producer must store the flag's line with :func:`store` and ``sem="release"``, and the consumer
    reloads it once with ``"acquire"`` after the loop.

    {access}

    Args:
        ptr (pointer to scalar): Global memory base pointer.
        offsets (tensor): int32 offsets of the elements to load, in elements.
        mask (tensor, optional): Mask, constant over each line. Defaults to None.
        other (tensor or scalar, optional): Values for masked-off elements. Defaults to None.
        sem (str, optional): ``"acquire"`` or ``"relaxed"``; ``"acquire"`` orders the whole operation as for
            :func:`~triton.language.atomic_load`. Defaults to None (``"acquire"``).
        scope (str, optional): ``"cta"``, ``"gpu"`` or ``"sys"``. Defaults to None (``"gpu"``).

    Returns:
        tensor: The loaded values, with the shape and layout of ``offsets``.
    """
    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    _verify(ptr, offsets)
    _check(other is None or mask is not None, lambda: "when other is not None, mask should not be None")
    sem = _atomic_semantic(sem, ir.MEM_SEMANTIC.ACQUIRE, ("acquire", "relaxed"), "cooperative.load", _semantic)
    scope = _atomic_scope(scope, _semantic)
    if mask is not None:
        offsets, mask = _semantic.broadcast_impl_value(offsets, _semantic.to_tensor(mask))
        other = _semantic.cast(_semantic.to_tensor(0 if other is None else other), ptr.dtype.element_ty)
        offsets, other = _semantic.broadcast_impl_value(offsets, other)
    ret_ty = offsets.type.with_element_ty(ptr.dtype.element_ty)
    builder = _semantic.builder
    handle = builder.create_cooperative_atomic_load(ret_ty.to_ir(builder), ptr.handle, offsets.handle,
                                                    mask.handle if mask is not None else ir.value(),
                                                    other.handle if other is not None else ir.value(), sem, scope)
    return tensor(handle, ret_ty)


@builtin
@_format_doc
def store(value, ptr, offsets, mask=None, *, sem=None, scope=None, _semantic=None):
    """
    Atomically store 128-byte lines from registers to global memory.

    Each 128-byte line becomes visible as a whole to cooperative loads; different lines may become visible in
    any order. Masked-off lines are not written.

    {access}

    Args:
        value (tensor or scalar): The values to store, converted to the pointee type.
        ptr (pointer to scalar): Global memory base pointer.
        offsets (tensor): int32 offsets of the elements to store, in elements.
        mask (tensor, optional): Mask, constant over each line. Defaults to None.
        sem (str, optional): ``"release"`` or ``"relaxed"``; ``"release"`` orders the whole operation as for
            :func:`~triton.language.atomic_store`. Defaults to None (``"release"``).
        scope (str, optional): ``"cta"``, ``"gpu"`` or ``"sys"``. Defaults to None (``"gpu"``).
    """
    mask = _unwrap_if_constexpr(mask)
    _verify(ptr, offsets)
    _check(not ptr.dtype.is_const(), lambda: "cannot store through a constant pointer")
    sem = _atomic_semantic(sem, ir.MEM_SEMANTIC.RELEASE, ("release", "relaxed"), "cooperative.store", _semantic)
    scope = _atomic_scope(scope, _semantic)
    offsets_shape = offsets.shape
    value = _semantic.to_tensor(value)
    if mask is None:
        offsets, value = _semantic.broadcast_tensors(offsets, value)
    else:
        offsets, value, mask = _semantic.broadcast_tensors(offsets, value, _semantic.to_tensor(mask))
    _check(offsets_shape == offsets.shape,
           lambda: f"Expected `offsets` argument to have shape {offsets.shape} but got {offsets_shape}")
    value = _semantic.cast(value, ptr.dtype.element_ty)
    _semantic.builder.create_cooperative_atomic_store(value.handle, ptr.handle, offsets.handle,
                                                      mask.handle if mask is not None else ir.value(), sem, scope)
