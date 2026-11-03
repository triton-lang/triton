# Adapted from https://github.com/triton-lang/triton/blob/d8cb93d979065c242472c8637827a20e72438409/python/triton_kernels/triton_kernels/testing.py

import torch

from triton_kernels.tensor import FP4
from triton_kernels.tensor import make_ragged_tensor_metadata, wrap_torch_tensor, convert_layout
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.testing import alloc_rand, make_slice_sizes


def make_random_tensor(shape, n_slices, ragged_dim, device, dtype, mxfp_dim, transpose, squeeze_batch_dim,
                       is_mx_rowmajor=False, scale_hbm_swizzling=None):
    # allocate buffer
    buffer_shape = ((n_slices, ) if ragged_dim is None else tuple()) + shape
    buffer_dtype = torch.bfloat16 if dtype.has_mx_scale else dtype.torch_dtype
    # FIXME: Took a long time with shape (10, 784, 400) on simulator.
    # buffer = alloc_rand(buffer_shape, device=device, dtype=buffer_dtype)
    buffer = alloc_rand(buffer_shape, device='cpu', dtype=buffer_dtype)
    buffer = buffer.to(device)
    if squeeze_batch_dim:
        buffer = buffer.squeeze(0)
    # handle raggedness
    ragged_metadata = None
    if ragged_dim is not None:
        slice_sizes = make_slice_sizes(n_slices, shape[ragged_dim], device=device)
        ragged_metadata = make_ragged_tensor_metadata(slice_sizes, shape[ragged_dim])
    # handle transpose
    if transpose:
        buffer = buffer.mT.contiguous().mT
    # handle mxfp
    scales = None
    if mxfp_dim is not None:
        assert dtype.has_mx_scale
        buffer_dtype = dtype.torch_dtype
        if is_mx_rowmajor:
            scales = downcast_to_mxfp(buffer, buffer_dtype, axis=mxfp_dim)[1]
            buffer = downcast_to_mxfp(buffer.mT.contiguous(), buffer_dtype, axis=mxfp_dim)[0].mT
        else:
            buffer, scales = downcast_to_mxfp(buffer, buffer_dtype, axis=mxfp_dim)
        buffer = wrap_torch_tensor(buffer, FP4 if dtype.is_mxfloat4 else None)
        scales = wrap_torch_tensor(scales)
        if scale_hbm_swizzling is not None:
            # convert scales to swizzled hbm layout
            if callable(scale_hbm_swizzling):
                scale_hbm_swizzling = scale_hbm_swizzling(ragged_metadata)
            scales = convert_layout(scales, scale_hbm_swizzling)
    return buffer, scales, ragged_metadata
