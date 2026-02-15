import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._core import builtin
from triton.experimental.gluon.language.nvidia.hopper.tma import (
    async_copy_global_to_shared,
    async_copy_shared_to_global,
    store_wait,
    tensor_descriptor,
    tensor_descriptor_type,
    make_tensor_descriptor,
    _emit_alignment_check,
)

__all__ = [
    "async_gather",
    "async_scatter",
    "async_copy_global_to_shared",
    "async_copy_shared_to_global",
    "store_wait",
    "tensor_descriptor",
    "tensor_descriptor_type",
    "make_tensor_descriptor",
]


@builtin
def async_gather(tensor_desc, x_offsets, y_offset, barrier, result, pred=True, _semantic=None):
    """
    Asynchronously gather elements from global memory to shared memory using TMA.

    Args:
        tensor_desc (tensor_descriptor): The tensor descriptor.
        x_offsets (tensor): 1D tensor of X offsets.
        y_offset (int): Scalar Y offset.
        barrier (shared_memory_descriptor): Barrier that will be signaled when the operation is complete.
        result (tensor_memory_descriptor): Result shared memory, must have NVMMASharedLayout.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
    """
    if _semantic.builder.options.enable_iisan:
        _emit_alignment_check(tensor_desc, (y_offset, ), "async_gather", "y_offset", _semantic=_semantic)

    pred = _semantic.to_tensor(pred)
    y_offset = _semantic.to_tensor(y_offset)
    _semantic.builder.create_async_tma_gather(tensor_desc.handle, x_offsets.handle, y_offset.handle, barrier.handle,
                                              result.handle, pred.handle)


def _emit_scatter_nonnegative_check(x_offsets, y_offset, _semantic=None):
    y_offset = ttgl.to_tensor(y_offset, _semantic=_semantic)
    zero = ttgl.to_tensor(0, _semantic=_semantic)

    is_nonnegative = y_offset.__ge__(zero, _semantic=_semantic)
    ttgl.device_assert(is_nonnegative, "async_scatter y_offset cannot be negative", _semantic=_semantic)

    is_nonnegative = x_offsets.__ge__(zero, _semantic=_semantic)
    ttgl.device_assert(is_nonnegative, "async_scatter x_offsets cannot have any negative elements", _semantic=_semantic)


@builtin
def async_scatter(tensor_desc, x_offsets, y_offset, src, _semantic=None):
    """
    Asynchronously scatter elements from shared memory to global memory using TMA.

    Args:
        tensor_desc (tensor_descriptor): The tensor descriptor.
        x_offsets (tensor): 1D tensor of X offsets.
        y_offset (int): Scalar Y offset.
        src (tensor_memory_descriptor): The source data, must be in NVMMASharedLayout.
    """
    if _semantic.builder.options.enable_iisan:
        _emit_alignment_check(tensor_desc, (y_offset, ), "async_scatter", "y_offset", _semantic=_semantic)
        _emit_scatter_nonnegative_check(x_offsets, y_offset, _semantic=_semantic)

    y_offset = _semantic.to_tensor(y_offset)
    _semantic.builder.create_async_tma_scatter(tensor_desc.handle, x_offsets.handle, y_offset.handle, src.handle)
