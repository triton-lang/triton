from __future__ import annotations

from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._semantic import _check

from .._core import builtin, _unwrap_if_constexpr

__all__ = ["slice"]


@builtin
def slice(source, shape, offsets, _semantic=None):
    """Extract a register only slice of a tensor.

    Returns a view of ``source`` of the requested ``shape`` starting at
    ``offsets``, keeping the source's distributed layout. Because the layout is
    preserved, the slice introduces no cross-thread data movement; the slice
    extent and offsets must align to the source layout's CTA tiling.

    Args:
        source (tensor): The tensor to slice. Must have a distributed layout.
        shape (List[int]): Shape of the extracted slice (same rank as source).
        offsets (List[int]): Per-dimension start offsets into ``source``.
    """
    _check(isinstance(source.type, ttgl.distributed_type),
           lambda: f"Expected source to have a distributed_type but got {source.type}")

    shape = [_unwrap_if_constexpr(s) for s in shape]
    offsets = [_unwrap_if_constexpr(o) for o in offsets]

    rank = len(source.type.shape)
    _check(len(shape) == rank, lambda: f"slice: shape must have rank {rank}, got {len(shape)}")
    _check(len(offsets) == rank, lambda: f"slice: offsets must have rank {rank}, got {len(offsets)}")

    src_shape = source.type.shape
    for i in range(rank):
        _check(shape[i] <= src_shape[i], lambda: f"slice: result shape {shape} cannot exceed source shape {src_shape}")
        _check(offsets[i] + shape[i] <= src_shape[i],
               lambda: f"slice: offset {offsets} + shape {shape} exceeds source shape {src_shape}")

    ret_ty = ttgl.distributed_type(source.dtype, shape, source.type.layout)
    handle = _semantic.builder.create_extract_slice(ret_ty.to_ir(_semantic.builder), source.handle, offsets)
    return ttgl.tensor(handle, ret_ty)
