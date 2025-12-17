from triton.experimental.gluon.language._core import builtin
from triton.experimental.gluon.language.nvidia.hopper.tma import (
    async_copy_global_to_shared,
    async_copy_shared_to_global,
    store_wait,
    tensor_descriptor,
    tensor_descriptor_type,
    make_tensor_descriptor,
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


def _check_gather_scatter(tensor_desc, x_offsets, smem, op_name, smem_name):
    # Tensor descriptor must be 2D and layout must match the shared memory layout.
    assert len(
        tensor_desc.block_shape
    ) == 2, f"async {op_name} requires a 2D tensor descriptor, but got one with rank {len(tensor_desc.block_shape)}"
    assert tensor_desc.layout == smem.layout, f"tensor descriptor layout {tensor_desc.layout} does not match {smem_name} shared memory layout {smem.layout}"
    # Row offsets must be 1D and have at least 8 rows.
    assert len(
        x_offsets.shape
    ) == 1, f"async {op_name} requires a 1D tensor of row offsets, but got one with rank {len(x_offsets.shape)}"
    assert x_offsets.shape[0] >= 8, f"async {op_name} requires at least 8 rows, but got {x_offsets.shape[0]}"
    # Block shape must be [1, Y] where Y >= min_cols.
    min_cols = 32 // tensor_desc.dtype.primitive_bitwidth * 8
    assert tensor_desc.block_shape[
        0] == 1, f"async {op_name} requires the tensor descriptor's block shape to have 1 row, but got {tensor_desc.block_shape}"
    assert tensor_desc.block_shape[
        1] >= min_cols, f"async {op_name} requires the tensor descriptor's block shape to have at least {min_cols} columns, but got {tensor_desc.block_shape[1]}"


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
    _check_gather_scatter(tensor_desc, x_offsets, result, "gather", "result")
    pred = _semantic.to_tensor(pred)
    y_offset = _semantic.to_tensor(y_offset)
    _semantic.builder.create_async_tma_gather(tensor_desc.handle, x_offsets.handle, y_offset.handle, barrier.handle,
                                              result.handle, pred.handle)


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
    _check_gather_scatter(tensor_desc, x_offsets, src, "scatter", "source")
    y_offset = _semantic.to_tensor(y_offset)
    _semantic.builder.create_async_tma_scatter(tensor_desc.handle, x_offsets.handle, y_offset.handle, src.handle)
