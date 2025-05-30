from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

__all__ = ["async_copy_global_to_local", "async_copy_local_to_global", "async_reduce", "store_wait"]


def _tensor_desc_to_tma_ptr(tensor_desc, builder):
    return builder.create_tensor_desc_to_tma_ptr(tensor_desc.handle)


@builtin
def async_copy_global_to_local(tensor_desc, coord, barrier, result, pred=None, _builder=None):
    coord = [c.handle for c in coord]
    if pred is None:
        pred = _builder.get_int1(True)
    else:
        pred = pred.handle

    tma_ptr = _tensor_desc_to_tma_ptr(tensor_desc, _builder)
    _builder.create_async_tma_copy_global_to_local(tma_ptr, coord, barrier.handle, result.handle, pred)


@builtin
def async_copy_local_to_global(tensor_desc, coord, src, _builder=None):
    coord = [c.handle for c in coord]
    tma_ptr = _tensor_desc_to_tma_ptr(tensor_desc, _builder)
    _builder.create_async_tma_copy_local_to_global(tma_ptr, coord, src.handle)


@builtin
def async_reduce(tensor_desc, coord, src, kind, _builder=None):
    coord = [c.handle for c in coord]
    tma_ptr = _tensor_desc_to_tma_ptr(tensor_desc, _builder)
    _builder.create_async_tma_reduce(tma_ptr, coord, src.handle, kind)


@builtin
def store_wait(pendings, _builder=None):
    pendings = _unwrap_if_constexpr(pendings)
    _builder.create_async_tma_store_wait(pendings)
