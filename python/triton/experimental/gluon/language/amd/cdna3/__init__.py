from __future__ import annotations
from typing import TYPE_CHECKING

# import triton.language as tl
# from triton.language.core import _unwrap_if_constexpr, _unwrap_shape, constexpr_type
from triton.experimental.gluon.language._core import builtin

if TYPE_CHECKING:
    from ..._semantic import GluonSemantic

__all__ = ["get_amd_mfma_layout", "create_buffer_load", "create_buffer_store"]


@builtin
def get_amd_mfma_layout(version, warps_per_cta, tiles_per_warp, m_dim, n_dim, transposed, ctas_per_cga, cta_split_num,
                        cta_order, elem_type_width, _semantic: GluonSemantic = None):
    return _semantic.builder.get_amd_mfma_layout(version, warps_per_cta, tiles_per_warp, m_dim, n_dim, transposed,
                                                 ctas_per_cga, cta_split_num, cta_order, elem_type_width)


def isfloat64(x):
    if not isinstance(x, float):
        return False

    min_float32 = 2**-126
    max_float32 = (2 - 2**-23) * 2**127
    abs_x = __builtins__['abs'](x)
    if abs_x == float("inf") or\
       abs_x == 0.0 or \
       x != x or \
           min_float32 <= abs_x <= max_float32:
        return False
    else:
        return True


@builtin
def create_buffer_load(ptr, element_type, offsets, cache, mask, layout, other, _semantic=None):
    cache_modifier = _semantic._str_to_load_cache_modifier(cache)
    return _semantic.create_buffer_load(ptr.handle, offsets.shape, element_type, offsets.handle, cache_modifier,
                                        mask.handle, other.handle, layout)


@builtin
def create_buffer_store(stored_value, ptr, offsets, cache, mask, _semantic: GluonSemantic = None):
    cache_modifier = _semantic._str_to_load_cache_modifier(cache)

    return _semantic.create_buffer_store(stored_value.handle, ptr.handle, offsets.handle, cache_modifier, mask.handle)
