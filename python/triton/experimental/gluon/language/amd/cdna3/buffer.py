"""
Descriptor-based buffer memory ops for CDNA3.

These access global memory through an explicit buffer resource descriptor (:func:`make_buffer_descriptor`)
instead of a base pointer, with explicit bounds, a scalar offset and per-access coherence scope. The
pointer-based ``buffer_load`` / ``buffer_store`` in the parent module are unchanged but will be deprecated and
migrated to these ops soon.
"""

from __future__ import annotations

from ..._core import builtin
from .. import _buffer
from .._buffer import buffer_descriptor, buffer_descriptor_type
from ._cache_policy import CachePolicy

__all__ = ["CachePolicy", "buffer_descriptor", "buffer_descriptor_type", "make_buffer_descriptor", "load", "store"]


@builtin
def make_buffer_descriptor(base, num_records, _semantic=None) -> buffer_descriptor:
    """
    Make a buffer resource descriptor from the hardware fields the program controls.

    Accesses through the descriptor address ``base + soffset + offsets[i]``. The hardware checks every
    access against ``num_records``, counting ``soffset``: in bytes from ``base``, an access is in bounds
    only if ``soffset + offset + size <= num_records``. Out-of-bounds loads return zero and out-of-bounds
    stores are dropped, so ``num_records=0`` disables every access.

    The check applies per hardware access, and accesses wider than 4 bytes are checked in 4-byte pieces.
    For elements of 4 bytes or more with a ``num_records`` that is a multiple of the element size, it is
    exact per element. Otherwise, elements sharing an access or a 4-byte piece with out-of-bounds bytes
    are out of bounds too, and an element wider than 4 bytes can be partly in bounds.

    The descriptor is as aligned as ``base``: use :func:`~triton.experimental.gluon.language.multiple_of`
    on ``base`` to let accesses through it vectorize when the compiler can't prove the alignment.

    Args:
        base (pointer to scalar): Global memory base pointer. Its pointee type is the element type of
            the descriptor; offsets count these elements.
        num_records (int32 or uint32 scalar): Bytes accessible from ``base``, read as an unsigned 32-bit
            value: up to 2**32 - 1.

    Returns:
        buffer_descriptor: the descriptor.
    """
    return _buffer.make_buffer_descriptor(base, num_records, _semantic)


_ACCESS_ARGS_DOC = """
        soffset (int32 scalar or tensor, optional): Offset, in elements, added to every element's
            offset and counted by the bounds check. It is held in the instruction's scalar offset
            register, so a uniform advance costs no per-lane arithmetic. A tensor gives each wave its
            own value: it broadcasts to ``offsets`` and must have size 1 in every dimension the lanes
            of the offsets' layout span, e.g. shape ``[M, 1]`` for ``[M, N]`` offsets whose rows each
            sit in one wave. Defaults to None (0).
        cache_policy (CachePolicy, optional): Cache hints. Defaults to None."""


@builtin
def load(desc, offsets, mask=None, other=None, *, soffset=None, scope=None, cache_policy=None, _semantic=None):
    """
    Load from global memory through a buffer descriptor into registers.

    Element ``i`` is read from ``base + soffset + offsets[i]``. Out-of-bounds elements read zero, and
    masked-off elements read ``other``, or zero without it. See :func:`make_buffer_descriptor` for the
    bounds.

    Args:
        desc (buffer_descriptor): Descriptor of the global memory to load from.
        offsets (tensor): Offsets of the elements to load, in elements.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor or scalar, optional): Values for masked-off elements. Defaults to None.
        scope (str, optional): ``"cta"``, ``"gpu"`` or ``"sys"``. The load reads data visible at that
            scope. It is not atomic and orders no other memory operation; pair it with an acquire to
            consume a payload. To poll a flag, use :func:`~triton.experimental.gluon.language.atomic_load`
            instead, which the compiler never hoists. Defaults to None (no coherence beyond the
            wave).""" + _ACCESS_ARGS_DOC + """

    Returns:
        tensor: The loaded values, with the shape and layout of ``offsets``.
    """
    return _buffer.load(desc, offsets, mask, other, soffset, scope, cache_policy, CachePolicy, _semantic)


@builtin
def store(value, desc, offsets, mask=None, *, soffset=None, scope=None, cache_policy=None, _semantic=None):
    """
    Store a tensor to global memory through a buffer descriptor.

    Element ``i`` is written to ``base + soffset + offsets[i]``. Out-of-bounds and masked-off elements
    are dropped. See :func:`make_buffer_descriptor` for the bounds.

    Args:
        value (tensor): The values to store.
        desc (buffer_descriptor): Descriptor of the global memory to store to.
        offsets (tensor): Offsets of the elements to store, in elements.
        mask (tensor, optional): Mask tensor for predicated stores. Defaults to None.
        scope (str, optional): ``"cta"``, ``"gpu"`` or ``"sys"``. The store makes its data available at
            that scope. It is not atomic and orders no other memory operation; follow it with a release
            to publish. Defaults to None.""" + _ACCESS_ARGS_DOC + """
    """
    _buffer.store(value, desc, offsets, mask, soffset, scope, cache_policy, CachePolicy, _semantic)
