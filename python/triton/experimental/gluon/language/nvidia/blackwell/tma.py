from triton.experimental.gluon.language._core import builtin
from triton.experimental.gluon.language.nvidia.hopper.tma import _tensor_desc_to_tma_ptr

__all__ = ["async_gather", "async_scatter"]


@builtin
def async_gather(tensor_desc, x_offsets, y_offset, barrier, result, pred=None, _builder=None):
    x_offsets = [x.handle for x in x_offsets]
    if pred is None:
        pred = _builder.get_int1(True)
    else:
        pred = pred.handle

    tma_ptr = _tensor_desc_to_tma_ptr(tensor_desc, _builder)
    _builder.create_async_tma_gather(tma_ptr, x_offsets, y_offset.handle, barrier.handle, result.handle, pred)


@builtin
def async_scatter(tensor_desc, x_offsets, y_offset, src, _builder=None):
    x_offsets = [x.handle for x in x_offsets]
    tma_ptr = _tensor_desc_to_tma_ptr(tensor_desc, _builder)
    _builder.create_async_tma_scatter(tma_ptr, x_offsets, y_offset.handle, src.handle)
