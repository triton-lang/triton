from __future__ import annotations
from typing import TYPE_CHECKING

from triton.experimental.gluon.language._core import builtin

if TYPE_CHECKING:
    from ..._semantic import GluonSemantic

__all__ = ["create_buffer_load", "create_buffer_store"]


@builtin
def create_buffer_load(ptr, element_type, offsets, cache, mask, layout, other, _semantic=None):
    cache_modifier = _semantic._str_to_load_cache_modifier(cache)
    return _semantic.create_buffer_load(ptr.handle, offsets.shape, element_type, offsets.handle, cache_modifier,
                                        mask.handle, other.handle, layout)


@builtin
def create_buffer_store(stored_value, ptr, offsets, cache, mask, _semantic: GluonSemantic = None):
    cache_modifier = _semantic._str_to_load_cache_modifier(cache)
    return _semantic.create_buffer_store(stored_value.handle, ptr.handle, offsets.handle, cache_modifier, mask.handle)
