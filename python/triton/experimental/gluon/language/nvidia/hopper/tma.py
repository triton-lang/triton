from triton.language.semantic import _convert_to_ir_values
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr
import triton.experimental.gluon.language._core as ttgl

__all__ = ["async_copy_global_to_shared", "async_copy_shared_to_global", "store_wait"]


@builtin
def async_copy_global_to_shared(tensor_desc, coord, barrier, result, pred=True, _builder=None):
    coord = _convert_to_ir_values(_builder, coord, require_i64=False)
    pred = ttgl.to_tensor(pred, _builder=_builder)
    _builder.create_async_tma_copy_global_to_local(tensor_desc.handle, coord, barrier.handle, result.handle,
                                                   pred.handle)


@builtin
def async_copy_shared_to_global(tensor_desc, coord, src, _builder=None):
    coord = _convert_to_ir_values(_builder, coord, require_i64=False)
    _builder.create_async_tma_copy_local_to_global(tensor_desc.handle, coord, src.handle)


@builtin
def store_wait(pendings, _builder=None):
    pendings = _unwrap_if_constexpr(pendings)
    _builder.create_async_tma_store_wait(pendings)
