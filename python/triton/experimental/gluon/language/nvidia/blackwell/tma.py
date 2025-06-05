from triton.experimental.gluon.language._core import builtin
import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language.nvidia.hopper.tma import (
    async_copy_global_to_shared,
    async_copy_shared_to_global,
    store_wait,
)

__all__ = [
    "async_gather",
    "async_scatter",
    "async_copy_global_to_shared",
    "async_copy_shared_to_global",
    "store_wait",
]


@builtin
def async_gather(tensor_desc, x_offsets, y_offset, barrier, result, pred=True, _builder=None):
    pred = ttgl.to_tensor(pred, _builder=_builder)
    y_offset = ttgl.to_tensor(y_offset, _builder=_builder)
    _builder.create_async_tma_gather(tensor_desc.handle, x_offsets.handle, y_offset.handle, barrier.handle,
                                     result.handle, pred.handle)


@builtin
def async_scatter(tensor_desc, x_offsets, y_offset, src, _builder=None):
    y_offset = ttgl.to_tensor(y_offset, _builder=_builder)
    _builder.create_async_tma_scatter(tensor_desc.handle, x_offsets.handle, y_offset.handle, src.handle)
