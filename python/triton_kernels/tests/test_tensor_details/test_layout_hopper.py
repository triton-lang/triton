import pytest
from triton_kernels.tensor import wrap_torch_tensor, convert_layout, empty, FP4
from triton_kernels.tensor_details.layout import HopperMXScaleLayout, HopperMXValueLayout, StridedLayout
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

ZERO_SIZED_SHAPES = [(0, 64), (64, 0), (2, 0), (0, 2), (0, 64, 64)]


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
    layout = HopperMXValueLayout(mx_axis - 2, mma_version)
    logical_shape = list(x.shape)
    logical_shape[-1] *= 2
    transformation = layout.make_transformation(logical_shape, is_fp4=True)
    swizzled = transformation.swizzle_data(x)
    assert list(swizzled.shape) == transformation.storage_shape
    res = transformation.unswizzle_data(swizzled)
    assert (res == x).all()


@pytest.mark.parametrize("shape", [(64, 128), (130, 66), (2, 34, 18), (2, 3, 66, 34)])
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("mx_axis", [-2, -1])
@pytest.mark.parametrize("mma_version", [2, 3])
def test_mxfp4_value_swizzle_matches_torch(shape, step, trans, mx_axis, mma_version):
    data_cpu = torch.randint(0, 256, tuple(step * size for size in shape), dtype=torch.uint8,
                             generator=torch.Generator().manual_seed(0))
    data_cuda = data_cpu.cuda()
    index = (slice(step - 1, None, step), ) * len(shape)
    data_cpu, data_cuda = data_cpu[index], data_cuda[index]
    if trans:
        data_cpu, data_cuda = data_cpu.mT, data_cuda.mT
    logical_shape = list(data_cpu.shape)
    logical_shape[-1] *= 2
    layout = HopperMXValueLayout(mx_axis, mma_version)
    transformation = layout.make_transformation(logical_shape, is_fp4=True)

    expected = transformation.swizzle_data(data_cpu)
    actual = transformation.swizzle_data(data_cuda)

    assert actual.stride() == expected.stride()
    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(transformation.unswizzle_data(actual).cpu(), data_cpu)


def test_mxfp4_value_storage_shape_matches_swizzle():
    x = torch.randint(0, 256, (64, 128), dtype=torch.uint8)
    transformation = HopperMXValueLayout(-1, 3).make_transformation([64, 256], is_fp4=True)

    swizzled = transformation.swizzle_data(x)

    assert swizzled.shape == (64, 512)
    assert transformation.storage_shape == list(swizzled.shape)
    assert torch.equal(transformation.unswizzle_data(swizzled), x)


@pytest.mark.parametrize(("shape", "major_dim"),
                         [(shape[:-1] + (2 * shape[-1], ), -1) for shape in ZERO_SIZED_SHAPES] + [((2, 0, 6, 8), -2)])
@pytest.mark.parametrize("mx_axis", [-2, -1])
@pytest.mark.parametrize("mma_version", [2, 3])
@pytest.mark.parametrize("device", ["cpu", "meta", "cuda"])
def test_mxfp4_value_zero_sized_roundtrip(shape, major_dim, mx_axis, mma_version, device):
    src_layout = StridedLayout(major_dim)
    src = empty(shape, dtype=FP4, device=device, layout=src_layout)
    layout = HopperMXValueLayout(mx_axis=mx_axis, mma_version=mma_version)
    transformation = layout.make_transformation(list(shape), True)

    swizzled = convert_layout(src, layout)
    canonical = transformation.unswizzle_data(swizzled.data)
    roundtrip = convert_layout(swizzled, src_layout)

    assert src.shape == swizzled.shape == roundtrip.shape == list(shape)
    assert list(swizzled.data.shape) == transformation.storage_shape
    assert canonical.shape == (*shape[:-1], shape[-1] // 2)
    assert roundtrip.data.shape == src.data.shape


@pytest.mark.parametrize("mx_axis", [-2, -1])
def test_mxfp4_value_convert_layout_roundtrip(mx_axis):
    x = torch.randint(0, 256, (64, 64), dtype=torch.uint8)
    src = wrap_torch_tensor(x, dtype=FP4)
    layout = HopperMXValueLayout(mx_axis=mx_axis, mma_version=3)

    swizzled = convert_layout(src, layout)
    roundtrip = convert_layout(swizzled, src.storage.layout)

    assert torch.equal(roundtrip.storage.data, x)


@pytest.mark.parametrize("shape", [(64, 64), (2, 34, 18)])
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("mx_axis", [-2, -1])
@pytest.mark.parametrize("mma_version", [2, 3])
def test_mxfp4_value_convert_layout_matches_torch(shape, trans, mx_axis, mma_version):
    data_cpu = torch.randint(0, 256, shape, dtype=torch.uint8)
    data_cuda = data_cpu.cuda()
    if trans:
        data_cpu, data_cuda = data_cpu.mT, data_cuda.mT
    src_cpu = wrap_torch_tensor(data_cpu, dtype=FP4)
    src_cuda = wrap_torch_tensor(data_cuda, dtype=FP4)
    layout = HopperMXValueLayout(mx_axis, mma_version)

    expected = convert_layout(src_cpu, layout)
    actual = convert_layout(src_cuda, layout)
    roundtrip = convert_layout(actual, src_cuda.storage.layout)

    assert actual.storage.data.stride() == expected.storage.data.stride()
    assert torch.equal(actual.storage.data.cpu(), expected.storage.data)
    assert torch.equal(roundtrip.storage.data, data_cuda)


@pytest.mark.parametrize("mx_axis", [-2, -1])
def test_mxfp4_value_swizzle_peak_allocation(mx_axis):
    data = torch.empty((2048, 2048), dtype=torch.uint8, device="cuda")
    layout = HopperMXValueLayout(mx_axis, 3)
    transformation = layout.make_transformation([2048, 4096], is_fp4=True)
    warm = transformation.swizzle_data(data)
    torch.cuda.synchronize(data.device)
    del warm
    baseline = torch.cuda.memory_allocated(data.device)
    torch.cuda.reset_peak_memory_stats(data.device)

    swizzled = transformation.swizzle_data(data)
    torch.cuda.synchronize(data.device)
    peak = torch.cuda.max_memory_allocated(data.device) - baseline

    # Allow overlapping byte buffers, but not whole-tensor int32 intermediates.
    assert peak <= 2 * swizzled.nbytes + 1024**2


@pytest.mark.parametrize("mx_axis", [0, 1])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("shape", [(256, 64), (256, 128), (256, 256)])
def test_mxfp4_scale_roundtrip(shape, mx_axis, num_warps):
    x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    layout = HopperMXScaleLayout(mx_axis=mx_axis - 2, num_warps=num_warps)
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    res = transformation.unswizzle_data(transformation.swizzle_data(x))
    assert (res[:shape[0], :shape[1]] == x).all()


@pytest.mark.parametrize("shape", ZERO_SIZED_SHAPES)
@pytest.mark.parametrize("mx_axis", [-2, -1])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("device", ["cpu", "meta", "cuda"])
def test_mxfp4_scale_zero_sized_roundtrip(shape, mx_axis, num_warps, device):
    x = torch.empty(shape, dtype=torch.uint8, device=device)
    src = wrap_torch_tensor(x)
    layout = HopperMXScaleLayout(mx_axis=mx_axis, num_warps=num_warps)

    swizzled = convert_layout(src, layout)
    roundtrip = convert_layout(swizzled, StridedLayout(mx_axis))

    assert roundtrip.storage.data.shape == x.shape


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


@pytest.mark.skipif(not cuda_capability_geq(9), reason="Only supported for capability >= 9")
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("mx_axis", [0, 1])
def test_upcast_mxfp4_to_bf16(num_warps, mx_axis):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    shape = [64, 64]
    shape[1 - mx_axis] = 32 * num_warps
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    x_fp4_val, x_fp4_scale = downcast_to_mxfp(x, torch.uint8, axis=mx_axis)
    x_bf16 = upcast_from_mxfp(x_fp4_val, x_fp4_scale, x.dtype, axis=mx_axis)
    x_fp4_val = wrap_torch_tensor(x_fp4_val, dtype=FP4)
    x_fp4_scale = wrap_torch_tensor(x_fp4_scale)
    x_fp4_val = convert_layout(x_fp4_val, HopperMXValueLayout(mx_axis=mx_axis - 2, mma_version=3))
    x_fp4_scale = convert_layout(x_fp4_scale, HopperMXScaleLayout(mx_axis=mx_axis - 2, num_warps=num_warps))
    y = torch.empty_like(x_bf16)
    scale_block = [s // 32 if i == mx_axis else s for i, s in enumerate(shape)]
    scale_block = x_fp4_scale.storage.layout.swizzle_block_shape(scale_block)
    value_block = [s // 2 if i == mx_axis else s for i, s in enumerate(shape)]
    value_block = x_fp4_val.storage.layout.swizzle_block_shape(value_block)
    _upcast_mxfp4_to_bf16[(1, )](
        y, x_fp4_val.storage.data, x_fp4_scale.storage.data,  #
        x_fp4_val.storage.data.stride(0), x_fp4_val.storage.data.stride(1),  #
        x_fp4_scale.storage.data.stride(0), x_fp4_scale.storage.data.stride(1),  #
        y.stride(0), y.stride(1),  #
        *value_block, *shape,  #
        *scale_block, mx_axis=mx_axis, num_warps=num_warps)
    assert (y == x_bf16).all()
