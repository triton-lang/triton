from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language._standard import _import_from_triton
from triton.experimental.gluon.language.nvidia.hopper import tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

import triton.tools.ragged_tma as tl_ragged

# fmt: off

def create_ragged_descriptor_host(T, block_shape, layout, ragged_dim=0):
    triton_desc = tl_ragged.create_ragged_descriptor(T, block_shape, ragged_dim)
    return TensorDescriptor(
        triton_desc.base,
        triton_desc.shape,
        triton_desc.strides,
        triton_desc.block_shape,
        layout,
        padding=triton_desc.padding
    )


_compute_ragged_descriptor_params_2d = _import_from_triton(tl_ragged._compute_ragged_descriptor_params_2d)
_compute_ragged_descriptor_params_3d = _import_from_triton(tl_ragged._compute_ragged_descriptor_params_3d)

@gluon.jit
def create_ragged_descriptor_device_2d(
    base_ptr,
    shape_0, shape_1,
    stride_0, stride_1: ttgl.constexpr,
    block_shape_0: ttgl.constexpr, block_shape_1: ttgl.constexpr,
    layout,
    ragged_dim: ttgl.constexpr
):
    shape, stride = _compute_ragged_descriptor_params_2d(
        shape_0, shape_1,
        stride_0, stride_1,
        ragged_dim
    )
    return tma.make_tensor_descriptor(
        base_ptr,
        shape=shape,
        strides=[stride[0], stride[1], stride[2], stride_1],
        block_shape=[1, 1, block_shape_0, block_shape_1],
        layout=layout,
    )


@gluon.jit
def create_ragged_descriptor_device_3d(
    base_ptr,
    shape_0, shape_1, shape_2,
    stride_0, stride_1, stride_2: ttgl.constexpr,
    block_shape_0: ttgl.constexpr, block_shape_1: ttgl.constexpr, block_shape_2: ttgl.constexpr,
    layout,
    ragged_dim: ttgl.constexpr
):
    shape, stride =  _compute_ragged_descriptor_params_3d(
        shape_0, shape_1, shape_2,
        stride_0, stride_1, stride_2,
        ragged_dim
    )
    return tma.make_tensor_descriptor(
        base_ptr,
        shape=shape,
        strides=[stride[0], stride[1], stride[2], stride[3], stride_2],
        block_shape=[1, 1, block_shape_0, block_shape_1, block_shape_2],
        layout=layout,
    )


_to_ragged_indices = _import_from_triton(tl_ragged.to_ragged_indices)


@gluon.jit
def to_ragged_coords(slice_off, slice_size, coords, ragged_dim: ttgl.constexpr):
    c0, c1, c2 = _to_ragged_indices(slice_off, slice_size, coords[ragged_dim])
    return [c0, c1] + coords[:ragged_dim] + [c2] + coords[ragged_dim + 1:]
