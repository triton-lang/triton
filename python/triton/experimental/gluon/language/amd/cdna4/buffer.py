"""
Descriptor-based buffer memory ops for CDNA4.

The same ops as :mod:`~triton.experimental.gluon.language.amd.cdna3.buffer`, plus :func:`load_to_shared`.
"""

from __future__ import annotations

from ..._core import builtin
from .. import _buffer
from ..cdna3.buffer import (CachePolicy, buffer_descriptor, buffer_descriptor_type, make_buffer_descriptor, load, store)

__all__ = [
    "CachePolicy", "buffer_descriptor", "buffer_descriptor_type", "make_buffer_descriptor", "load", "store",
    "load_to_shared"
]


@builtin
def load_to_shared(dest, desc, offsets, mask=None, *, soffset=None, scope=None, cache_policy=None, _semantic=None):
    """
    Load from global memory through a buffer descriptor directly into shared memory.

    Like :func:`~triton.experimental.gluon.language.amd.cdna4.async_copy.buffer_load_to_shared`, the copy
    is asynchronous: commit it with :func:`~triton.experimental.gluon.language.amd.cdna4.async_copy.commit_group`
    and wait with :func:`~triton.experimental.gluon.language.amd.cdna4.async_copy.wait_group` before
    reading ``dest``. The same layout requirements apply to ``offsets``, ``mask`` and ``dest``.

    Element ``i`` is read from ``base + soffset + offsets[i]``. Masked-off and out-of-bounds elements write
    zero, at no extra cost; apply any other fill value after loading from ``dest``. See
    :func:`~triton.experimental.gluon.language.amd.cdna3.buffer.make_buffer_descriptor` for the bounds.

    Args:
        dest (shared_memory_descriptor): Destination shared memory descriptor.
        desc (buffer_descriptor): Descriptor of the global memory to load from.
        offsets (tensor): Offsets of the elements to load, in elements.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        soffset (int32 scalar, optional): Scalar offset, in elements, added to every element's offset and
            counted by the bounds check. Defaults to None (0).
        scope (str, optional): ``"cta"``, ``"gpu"`` or ``"sys"``. The load reads data visible at that
            scope. It is not atomic and orders nothing. Defaults to None.
        cache_policy (CachePolicy, optional): Cache hints. Defaults to None.
    """
    _buffer.load_to_shared(dest, desc, offsets, mask, soffset, scope, cache_policy, CachePolicy, _semantic)
