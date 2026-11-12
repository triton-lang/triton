"""
Descriptor-based buffer memory ops for RDNA4.

These access global memory through an explicit buffer resource descriptor (:func:`make_buffer_descriptor`)
instead of a base pointer, with explicit bounds, a scalar offset and per-access coherence scope.
"""

from __future__ import annotations

from ..._core import builtin
from .. import _buffer
from .._buffer import buffer_descriptor, buffer_descriptor_type
from ..cdna3.buffer import _format_doc, make_buffer_descriptor
from ..cdna5._cache_policy import CachePolicy as _CachePolicy

__all__ = ["buffer_descriptor", "buffer_descriptor_type", "make_buffer_descriptor", "load", "store"]


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
            instead, which the compiler never hoists. Defaults to None, the CU scope: the same as ``"cta"``
            with ``wgp_cu_mode="cu"``, but narrower in the default WGP mode, where a workgroup can span
            the two CUs of a WGP.
        {soffset_arg}
        cache_policy (:class:`~triton.experimental.gluon.language.amd.rdna4.CachePolicy`, optional): Cache
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
            to publish. Defaults to None, the CU scope: the same as ``"cta"``
            with ``wgp_cu_mode="cu"``, but narrower in the default WGP mode, where a workgroup can span
            the two CUs of a WGP.
        {soffset_arg}
        cache_policy (:class:`~triton.experimental.gluon.language.amd.rdna4.CachePolicy`, optional): Cache
            hints. Defaults to None.
    """
    _buffer.store(value, desc, offsets, mask, soffset, scope, cache_policy, _CachePolicy, _semantic)
