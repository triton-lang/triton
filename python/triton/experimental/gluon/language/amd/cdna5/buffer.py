"""
Descriptor-based buffer memory ops for CDNA5.

These access global memory through an explicit buffer resource descriptor (:func:`make_buffer_descriptor`)
instead of a base pointer, with explicit bounds, a scalar offset and per-access coherence scope. The
pointer-based ``buffer_load`` / ``buffer_store`` in the parent module are unchanged but will be deprecated and
migrated to these ops soon.
"""

from __future__ import annotations

from ..._core import builtin
from .. import _buffer
from .._buffer import buffer_descriptor, buffer_descriptor_type
from ..cdna3.buffer import _format_doc
from ._cache_policy import CachePolicy as _CachePolicy

__all__ = ["buffer_descriptor", "buffer_descriptor_type", "make_buffer_descriptor", "load", "store"]


@builtin
def make_buffer_descriptor(base, num_records, _semantic=None) -> buffer_descriptor:
    """
    Make a buffer resource descriptor from the hardware fields the program controls.

    Accesses through the descriptor address ``base + soffset + offsets[i]``. The hardware checks every
    access against ``num_records``, counting ``soffset``: in bytes from ``base``, an access is in bounds
    only if ``soffset + offset + size <= num_records``. Out-of-bounds loads return zero and out-of-bounds
    stores are dropped, so ``num_records=0`` disables every access.

    The check applies per hardware access. An access that ends within ``num_records`` is in bounds as a
    whole. One that crosses it is checked in 4-byte pieces, and a piece is in bounds only if it ends at
    least 4 bytes before ``num_records``: elements of that access near the bound are out of bounds too,
    even when they fit. So the check is exact per element when no access crosses ``num_records``, such
    as with accesses of one element, or with a ``num_records`` that is a multiple of the access size.

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


@builtin
@_format_doc
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
            instead, which the compiler never hoists. Defaults to None (coherent within the
            workgroup).
        {soffset_arg}
        cache_policy (:class:`~triton.experimental.gluon.language.amd.cdna5.CachePolicy`, optional): Cache
            hints. Defaults to None.

    Returns:
        tensor: The loaded values, with the shape and layout of ``offsets``.
    """
    return _buffer.load(desc, offsets, mask, other, soffset, scope, cache_policy, _CachePolicy, _semantic)


@builtin
@_format_doc
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
            to publish. Defaults to None.
        {soffset_arg}
        cache_policy (:class:`~triton.experimental.gluon.language.amd.cdna5.CachePolicy`, optional): Cache
            hints. Defaults to None.
    """
    _buffer.store(value, desc, offsets, mask, soffset, scope, cache_policy, _CachePolicy, _semantic)
