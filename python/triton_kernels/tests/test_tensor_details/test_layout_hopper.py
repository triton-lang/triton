import pytest
from triton._internal_testing import is_cuda
from triton_kernels.tensor import wrap_torch_tensor, convert_layout, FP4
from triton_kernels.tensor_details.layout import HopperMXScaleLayout, HopperMXValueLayout
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp, upcast_from_mxfp
from triton_kernels.tensor_details.layout_details.hopper_value import mxfp4_to_bf16_triton
from triton_kernels.tensor_details.layout_details.hopper_scale import unswizzle_mxfp4_scale_hopper
from triton_kernels.target_info import cuda_capability_geq
import triton.language as tl
import triton
import torch

# ------------------------------------------------------------
# Torch tests
# ------------------------------------------------------------


@pytest.mark.parametrize("shape", [(16, 32), (16, 64), (32, 32), (32, 64), (64, 128), (128, 128)])
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("mx_axis", [0, 1])
@pytest.mark.parametrize("mma_version", [2, 3])
def test_mxfp4_value_roundtrip(shape, trans, mx_axis, mma_version):
    x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    if trans:
        x = x.mT
    if x.shape[1 - mx_axis] < 32:
        pytest.skip("Not enough elements along non-mx axis")
    layout = HopperMXValueLayout(x.shape, mx_axis, mma_version)
    res = layout.unswizzle_data(layout.swizzle_data(x))
    assert (res == x).all()


@pytest.mark.parametrize("mx_axis", [0, 1])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("shape", [(256, 64), (256, 128), (256, 256)])
def test_mxfp4_scale_roundtrip(shape, mx_axis, num_warps):
    x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    layout = HopperMXScaleLayout(x.shape, mx_axis=mx_axis, num_warps=num_warps)
    res = layout.unswizzle_data(layout.swizzle_data(x))
    assert (res[:shape[0], :shape[1]] == x).all()


# ------------------------------------------------------------
# Triton tests
# ------------------------------------------------------------

# ------------------ upcast mxfp4 to bf16 --------------------


@triton.jit
def _upcast_mxfp4_to_bf16(Y, X, XScale, x_stride_m, x_stride_n, x_scale_stride_m, x_scale_stride_n, y_stride_m,
                          y_stride_n, X_BLOCK_M: tl.constexpr, X_BLOCK_N: tl.constexpr, Y_BLOCK_M: tl.constexpr,
                          Y_BLOCK_N: tl.constexpr, SCALE_BLOCK_M: tl.constexpr, SCALE_BLOCK_N: tl.constexpr,
                          mx_axis: tl.constexpr):
    offs_m_val = tl.arange(0, X_BLOCK_M)
    offs_n_val = tl.arange(0, X_BLOCK_N)
    offs_m_scale = tl.arange(0, SCALE_BLOCK_M)
    offs_n_scale = tl.arange(0, SCALE_BLOCK_N)
    # load values
    offs_x = offs_m_val[:, None] * x_stride_m + offs_n_val[None, :] * x_stride_n
    x = tl.load(X + offs_x)
    # load scales
    offs_x_scale = offs_m_scale[:, None] * x_scale_stride_m + offs_n_scale[None, :] * x_scale_stride_n
    x_scale = tl.load(XScale + offs_x_scale)
    x_scale = unswizzle_mxfp4_scale_hopper(x_scale, mx_axis=mx_axis, num_warps=tl.extra.cuda.num_warps())
    y = mxfp4_to_bf16_triton(x, x_scale, mx_axis=mx_axis)
    # write back output
    offs_m_val = tl.arange(0, Y_BLOCK_M)
    offs_n_val = tl.arange(0, Y_BLOCK_N)
    offs_y = offs_m_val[:, None] * y_stride_m + offs_n_val[None, :] * y_stride_n
    tl.store(Y + offs_y, y)


@pytest.mark.skipif(not is_cuda(), reason="Only supported on cuda")
@pytest.mark.skipif(not cuda_capability_geq(9), reason="Only supported for capability >= 9")
def test_upcast_mxfp4_to_bf16():
    mx_axis = 0
    num_warps = 4
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    shape = (256, 128)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    x_fp4_val, x_fp4_scale = downcast_to_mxfp(x, torch.uint8, axis=mx_axis)
    x_bf16 = upcast_from_mxfp(x_fp4_val, x_fp4_scale, x.dtype, axis=mx_axis)
    x_fp4_val = wrap_torch_tensor(x_fp4_val, dtype=FP4)
    x_fp4_scale = wrap_torch_tensor(x_fp4_scale)
    x_fp4_val = convert_layout(x_fp4_val, HopperMXValueLayout, mx_axis=mx_axis)
    x_fp4_scale = convert_layout(x_fp4_scale, HopperMXScaleLayout, mx_axis=mx_axis, num_warps=num_warps)
    y = torch.empty_like(x_bf16)
    _upcast_mxfp4_to_bf16[(1, )](
        y, x_fp4_val.storage.data, x_fp4_scale.storage.data,  #
        x_fp4_val.storage.data.stride(0), x_fp4_val.storage.data.stride(1),  #
        x_fp4_scale.storage.data.stride(0), x_fp4_scale.storage.data.stride(1),  #
        y.stride(0), y.stride(1),  #
        x_fp4_val.storage.data.shape[0], x_fp4_val.storage.data.shape[1],  #
        shape[0], shape[1],  #
        x_fp4_scale.storage.data.shape[0], x_fp4_scale.storage.data.shape[1],  #
        mx_axis=mx_axis, num_warps=num_warps)
    assert (y == x_bf16).all()
