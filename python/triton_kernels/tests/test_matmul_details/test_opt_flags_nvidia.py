import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_cuda

from triton_kernels.matmul import matmul, matmul_torch, PrecisionConfig
from triton_kernels.matmul_details._matmul import _compute_packed_n_w
from triton_kernels.matmul_details.opt_flags import InapplicableConstraint, scoped_opt_flags_constraints
from triton_kernels.matmul_details.opt_flags_details import opt_flags_nvidia
from triton_kernels.numerics_details.mxfp import MXFP_BLOCK_SIZE, downcast_to_mxfp
from triton_kernels.tensor import FP4, UINT8, Storage, Tensor, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details import layout
from triton_kernels.tensor_details.layout import BlackwellMX4ValueShuffledLayout
from triton_kernels.tensor_details.layout_details.blackwell_scale import BlackwellMXScaleLayout
from triton_kernels.tensor_details.ragged_tensor import make_ragged_tensor_metadata
from triton_kernels.testing import assert_close


def _make_blackwell_scale_tensor():
    scale_storage = Storage(torch.empty((1, 128), dtype=torch.uint8), BlackwellMXScaleLayout())
    return Tensor(scale_storage, dtype=UINT8)


def _make_blackwell_mxfp4_weight(device, k, n):
    weight_fp = torch.randn((n, k), device=device, dtype=torch.bfloat16).T
    weight_val, weight_scale = downcast_to_mxfp(weight_fp, torch.uint8, axis=-2)
    weight_val = wrap_torch_tensor(weight_val, dtype=FP4)
    weight_scale = wrap_torch_tensor(weight_scale, dtype=UINT8)
    weight_scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=-2)
    weight_scale = convert_layout(weight_scale, weight_scale_layout)
    return weight_val, weight_scale


def _make_batched_blackwell_mxfp4_weight(device, batch_size, k, n):
    weight_fp = torch.randn((batch_size, n, k), device=device, dtype=torch.bfloat16).transpose(-2, -1)
    weight_val, weight_scale = downcast_to_mxfp(weight_fp, torch.uint8, axis=-2)
    weight_val = wrap_torch_tensor(weight_val, dtype=FP4)
    weight_scale = wrap_torch_tensor(weight_scale, dtype=UINT8)
    weight_scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=-2)
    weight_scale = convert_layout(weight_scale, weight_scale_layout)
    return weight_val, weight_scale


def _shuffle_blackwell_mxfp4_weight(weight):
    shuffled_layout = BlackwellMX4ValueShuffledLayout()
    return convert_layout(weight, shuffled_layout)


@pytest.mark.parametrize(
    "constraints",
    [
        pytest.param({"is_persistent": False}, id="regular"),
        pytest.param({"is_persistent": True, "block_m": 128}, id="persistent"),
    ],
)
def test_matmul_hopper_mxfp4_rhs_scale_padding_is_masked(device, constraints):
    if device != "cuda" or not torch.cuda.is_available() or not is_cuda():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires Hopper")

    torch.manual_seed(0)
    # k=1504 gives 47 MXFP scale columns along K. Hopper scale swizzling pads
    # that to 48 columns, so dirtying swizzled zero bytes targets the K-tail
    # scale padding. n=256 is one full Hopper N tile, avoiding unrelated N padding.
    m, k, n = 64, 1504, 256
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    weight_fp = torch.randn((n, k), device=device, dtype=torch.bfloat16).T
    weight_val, weight_scale = downcast_to_mxfp(weight_fp, torch.uint8, axis=-2)

    value_layout = layout.make_default_matmul_mxfp4_w_layout(mx_axis=-2)
    scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=-2, num_warps=8)
    b = convert_layout(wrap_torch_tensor(weight_val, dtype=FP4), value_layout)
    b_scale = convert_layout(wrap_torch_tensor(weight_scale, dtype=UINT8), scale_layout)

    # Ones remain ones through the scale swizzle; zeros identify padded bytes.
    scale_padding = convert_layout(
        wrap_torch_tensor(torch.ones_like(weight_scale), dtype=UINT8),
        scale_layout,
    ).storage.data == 0
    assert bool(scale_padding.any().item())

    b_scale_dirty_padding = convert_layout(wrap_torch_tensor(weight_scale.clone(), dtype=UINT8), scale_layout)
    b_scale_dirty_padding.storage.data[scale_padding] = 0xFF

    precision_kwargs = {
        "b_microblock_size": MXFP_BLOCK_SIZE.value,
        "out_dtype": a.dtype,
    }
    try:
        with scoped_opt_flags_constraints(constraints):
            expected = matmul(
                a,
                b,
                None,
                precision_config=PrecisionConfig(b_mx_scale=b_scale, **precision_kwargs),
            )
            actual = matmul(
                a,
                b,
                None,
                precision_config=PrecisionConfig(b_mx_scale=b_scale_dirty_padding, **precision_kwargs),
            )
    except (InapplicableConstraint, NotImplementedError) as e:
        pytest.skip(f"inapplicable opt_flags constraint {e}")

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@triton.jit
def _hopper_rhs_packed_n_extent(out, n: tl.constexpr):
    tl.store(out, _compute_packed_n_w(n, 4, "HOPPER_VALUE"))


@pytest.mark.parametrize("n", [258, 320])
def test_matmul_hopper_mxfp4_rhs_packed_n_padding(device, n):
    if device != "cuda" or not torch.cuda.is_available() or not is_cuda():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires Hopper")

    torch.manual_seed(0)
    # Hopper MXFP4 RHS values are stored with N packed by 4 and then padded in
    # packed space. The generic kernel must ceil-divide before padding and wrap
    # using that padded packed width.
    packed_n = torch.empty((1, ), dtype=torch.int32, device=device)
    _hopper_rhs_packed_n_extent[(1, )](packed_n, n)
    assert packed_n.item() == 128

    m, k = 64, 2048
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    weight_fp = torch.randn((n, k), device=device, dtype=torch.bfloat16).T
    weight_val, weight_scale = downcast_to_mxfp(weight_fp, torch.uint8, axis=-2)
    value_layout = layout.make_default_matmul_mxfp4_w_layout(mx_axis=-2)
    scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=-2, num_warps=8)
    b = convert_layout(wrap_torch_tensor(weight_val, dtype=FP4), value_layout)
    b_scale = convert_layout(wrap_torch_tensor(weight_scale, dtype=UINT8), scale_layout)
    assert b.storage.data.shape[-1] == packed_n.item()
    precision_config = PrecisionConfig(
        b_mx_scale=b_scale,
        b_microblock_size=MXFP_BLOCK_SIZE.value,
        out_dtype=a.dtype,
    )

    with scoped_opt_flags_constraints({"is_persistent": False, "block_n": 256}):
        expected = matmul_torch(a, b, None, precision_config=precision_config)
        actual = matmul(a, b, None, precision_config=precision_config)

    assert torch.isfinite(actual).all()
    assert_close(expected, actual, maxtol=3e-2, rmstol=None)


@pytest.mark.parametrize("n, expected", [(64, 128), (200, 256)])
def test_compute_block_n_blackwell_scale_aligns_to_128(n, expected):
    precision_config = PrecisionConfig(
        b_mx_scale=_make_blackwell_scale_tensor(),
        b_microblock_size=MXFP_BLOCK_SIZE.value,
    )
    block_n, block_n_tma = opt_flags_nvidia.compute_block_n(n, None, precision_config)
    assert block_n == block_n_tma == expected


def test_compute_num_warps_uses_two_warp_floor():
    precision_config = PrecisionConfig()
    assert opt_flags_nvidia.compute_num_warps(16, 256, False, precision_config, {}) == 2
    assert opt_flags_nvidia.compute_num_warps(16, 256, False, precision_config, {"num_warps": 1}) == 1


def test_matmul_blackwell_scale_small_n(device):
    if device != "cuda" or not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell or newer")

    torch.manual_seed(0)
    m, n, k = 128, 64, 128
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b, b_scale = _make_blackwell_mxfp4_weight(device, k, n)
    precision_config = PrecisionConfig(
        b_mx_scale=b_scale,
        b_microblock_size=MXFP_BLOCK_SIZE.value,
        out_dtype=a.dtype,
    )
    tri_y = matmul(a, b, None, precision_config=precision_config)
    ref_y = matmul_torch(a.to(torch.bfloat16), b, None, precision_config=precision_config)
    assert_close(ref_y, tri_y, maxtol=3e-2, rmstol=None)


@pytest.mark.parametrize("split_k", [1, 2])
def test_matmul_clc(device, split_k):
    if device != "cuda" or not torch.cuda.is_available() or not is_cuda():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell or newer")

    torch.manual_seed(0)
    m, n, k = 4096, 1024, 128
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b = torch.randn((k, n), device=device, dtype=torch.bfloat16)
    with scoped_opt_flags_constraints({
            "clc": True,
            "block_m": 128,
            "block_n": 128,
            "block_k": 64,
            "split_k": split_k,
    }):
        tri_y = matmul(a, b, None)

    ref_y = matmul_torch(a, b, None, precision_config=PrecisionConfig())
    assert_close(ref_y, tri_y, maxtol=3e-2, rmstol=None)


def _assert_ragged_m_output(output, a, b, slice_sizes, sentinel):
    offset = 0
    for expert, size in enumerate(slice_sizes):
        expected = (a[offset:offset + size].float() @ b[expert].float()).to(output.dtype)
        torch.testing.assert_close(output[offset:offset + size], expected, rtol=0.02, atol=0.02)
        offset += size
    # Capacity-only tasks must not write output rows outside the useful range.
    tail = output[offset:]
    torch.testing.assert_close(tail, torch.full_like(tail, sentinel), rtol=0, atol=0)


@pytest.mark.parametrize("slice_sizes,capacity,split_k", [
    pytest.param((512, 256, 256, 0), 1024, 1, id="balanced-empty-expert"),
    pytest.param((0, 0, 1024, 0), 1024, 1, id="one-hot-expert"),
    pytest.param((1, 127, 129, 255), 1024, 1, id="partial-tiles-capacity-tail"),
    pytest.param((0, 0, 0, 0), 1024, 1, id="zero-useful-tiles"),
    pytest.param((16384, 8192, 4096, 0), 32768, 1, id="multiple-waves"),
    pytest.param((1, 127, 129, 255), 512, 2, id="split-k-capacity-tail"),
])
@pytest.mark.parametrize("transpose_rhs", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_matmul_clc_ragged_m_capacity(device, slice_sizes, capacity, split_k, transpose_rhs, dtype):
    if device != "cuda" or not torch.cuda.is_available() or not is_cuda():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell or newer")

    torch.manual_seed(0)
    n, k = 512, 256
    a = torch.randn((capacity, k), device=device, dtype=torch.bfloat16).to(dtype)
    weight_shape = (len(slice_sizes), n, k) if transpose_rhs else (len(slice_sizes), k, n)
    b = torch.randn(weight_shape, device=device, dtype=torch.bfloat16).to(dtype)
    if transpose_rhs:
        b = b.mT
    sizes = torch.tensor(slice_sizes, device=device, dtype=torch.int32)
    metadata = make_ragged_tensor_metadata(sizes, capacity)
    sentinel = 123.0
    precision = PrecisionConfig(out_dtype=torch.bfloat16)
    constraints = {
        "block_m": 128,
        "block_n": 128,
        "block_k": 64,
        "is_persistent": True,
        "split_k": split_k,
        "idle_sms": 0,
    }
    outputs = []
    for use_clc in (False, True):
        output = torch.full((capacity, n), sentinel, device=device, dtype=torch.bfloat16)
        with scoped_opt_flags_constraints({**constraints, "clc": use_clc}):
            matmul(a, b, None, a_ragged_metadata=metadata, c=output, precision_config=precision)
        outputs.append(output)

    # Keep tile shape and reduction order fixed while changing the scheduler.
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
    _assert_ragged_m_output(outputs[1], a, b, slice_sizes, sentinel)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_matmul_clc_ragged_m_graph_replay(device, dtype):
    if device != "cuda" or not torch.cuda.is_available() or not is_cuda():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell or newer")

    torch.manual_seed(0)
    capacity, n, k = 1024, 512, 256
    a = torch.randn((capacity, k), device=device, dtype=torch.bfloat16).to(dtype)
    b = torch.randn((4, k, n), device=device, dtype=torch.bfloat16).to(dtype)
    sizes = torch.tensor((256, 256, 256, 256), device=device, dtype=torch.int32)
    sentinel = 123.0
    output = torch.full((capacity, n), sentinel, device=device, dtype=torch.bfloat16)
    precision = PrecisionConfig(out_dtype=torch.bfloat16)
    constraints = {
        "block_m": 128,
        "block_n": 128,
        "block_k": 64,
        "is_persistent": True,
        "split_k": 1,
        "idle_sms": 0,
        "clc": True,
    }

    def run():
        metadata = make_ragged_tensor_metadata(sizes, capacity)
        matmul(a, b, None, a_ragged_metadata=metadata, c=output, precision_config=precision)

    with scoped_opt_flags_constraints(constraints):
        warmup = torch.cuda.Stream(device=device)
        warmup.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup):
            for _ in range(3):
                run()
        torch.cuda.current_stream(device).wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()

    # Reuse the same launch while the device-computed useful count changes.
    for values in ((1, 127, 129, 255), (0, 0, 0, 0), (0, 0, 1024, 0)):
        sizes.copy_(torch.tensor(values, device=device, dtype=torch.int32))
        a.copy_(torch.randn(a.shape, device=device, dtype=torch.bfloat16).to(dtype))
        output.fill_(sentinel)
        graph.replay()
        _assert_ragged_m_output(output, a, b, values, sentinel)


def test_matmul_blackwell_shuffled_mxfp4_weight(device):
    if device != "cuda" or not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell or newer")

    torch.manual_seed(0)
    batch_size, m, n, k = 2, 128, 128, 128
    a = torch.randn((batch_size, m, k), device=device, dtype=torch.bfloat16).to(torch.float8_e5m2)
    b, b_scale = _make_batched_blackwell_mxfp4_weight(device, batch_size, k, n)
    b_shuffled = _shuffle_blackwell_mxfp4_weight(b)

    # Sanity-check the host-side packing; this is the layout consumed by the
    # W_SHUFFLED TMA load path in _p_matmul.
    assert torch.equal(b.storage.data, convert_layout(b_shuffled, b.storage.layout).storage.data)

    precision_config = PrecisionConfig(
        b_mx_scale=b_scale,
        b_microblock_size=MXFP_BLOCK_SIZE.value,
        out_dtype=torch.bfloat16,
    )
    constraints = {
        "is_persistent": True,
        "block_m": 128,
    }
    with scoped_opt_flags_constraints(constraints):
        tri_y = matmul(a, b_shuffled, None, precision_config=precision_config)

    ref_y = matmul_torch(a.to(torch.bfloat16), b, None, precision_config=precision_config)
    assert_close(ref_y, tri_y, maxtol=3e-2, rmstol=None)
