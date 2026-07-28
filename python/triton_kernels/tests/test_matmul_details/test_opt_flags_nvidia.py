from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_cuda

from triton_kernels.matmul import matmul, matmul_torch, PrecisionConfig
from triton_kernels.matmul_details._matmul import _compute_packed_n_w
from triton_kernels.matmul_details import opt_flags
from triton_kernels.matmul_details.opt_flags import InapplicableConstraint, scoped_opt_flags_constraints
from triton_kernels.matmul_details.opt_flags_details import opt_flags_nvidia
from triton_kernels.numerics_details.mxfp import MXFP_BLOCK_SIZE, downcast_to_mxfp
from triton_kernels.tensor import BF16, FP4, UINT8, Storage, Tensor, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details import layout
from triton_kernels.tensor_details.layout import BlackwellMX4ValueShuffledLayout
from triton_kernels.tensor_details.layout_details.blackwell_scale import BlackwellMXScaleLayout
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


@pytest.mark.parametrize(
    "capability, rows, is_ragged, has_act_tensor_scale, has_weight_tensor_scale, "
    "epilogue_reduction_n, constraints, expected",
    [
        pytest.param(
            (10, 3), 65_536, True, True, True, 1, {}, (128, 256, 256, 4, 3, 2),
            id="gb300-large-ragged-nvfp4",
        ),
        pytest.param(
            (10, 3), 65_536, True, True, True, 2, {}, (128, 256, 256, 4, 2, 2),
            id="gb300-large-ragged-nvfp4-fused-activation",
        ),
        pytest.param(
            (10, 0), 65_536, True, True, True, 1, {}, (128, 128, 128, 8, 4, 1),
            id="gb200-keeps-default-tile",
        ),
        pytest.param(
            (10, 3), 65_535, True, True, True, 1, {}, (128, 128, 128, 8, 4, 1),
            id="small-ragged-nvfp4-keeps-default-tile",
        ),
        pytest.param(
            (10, 3), 65_536, False, True, True, 1, {}, (128, 128, 128, 8, 4, 1),
            id="dense-nvfp4-keeps-default-tile",
        ),
        pytest.param(
            (10, 3), 65_536, True, False, True, 1, {}, (128, 128, 128, 8, 4, 1),
            id="missing-activation-tensor-scale-keeps-default-tile",
        ),
        pytest.param(
            (10, 3), 65_536, True, True, False, 1, {}, (128, 128, 128, 8, 4, 1),
            id="missing-weight-tensor-scale-keeps-default-tile",
        ),
        pytest.param(
            (10, 3),
            65_536,
            True,
            True,
            True,
            1,
            {
                "block_m": 64,
                "block_n": 128,
                "block_k": 128,
                "num_warps": 8,
                "num_stages": 2,
                "epilogue_subtile": 1,
            },
            (64, 128, 128, 8, 2, 1),
            id="explicit-caller-constraints-win",
        ),
    ],
)
def test_make_default_opt_flags_nvidia_large_ragged_nvfp4_tile(
    monkeypatch,
    capability,
    rows,
    is_ragged,
    has_act_tensor_scale,
    has_weight_tensor_scale,
    epilogue_reduction_n,
    constraints,
    expected,
):
    monkeypatch.setattr(
        opt_flags,
        "cuda_capability_geq",
        lambda major, minor: capability >= (major, minor),
    )
    monkeypatch.setattr(
        opt_flags.torch.cuda,
        "get_device_properties",
        lambda *_args: SimpleNamespace(multi_processor_count=16),
    )
    monkeypatch.setattr(opt_flags.target_info, "has_native_mxfp", lambda: capability >= (10, 0))
    monkeypatch.setattr(opt_flags_nvidia, "compute_block_n", lambda *_args: (128, 128))
    monkeypatch.setattr(opt_flags_nvidia, "compute_grid_size", lambda *_args: 16)
    monkeypatch.setattr(opt_flags_nvidia, "compute_block_k", lambda *_args: 128)
    monkeypatch.setattr(opt_flags_nvidia, "compute_num_stages", lambda *_args, **_kwargs: 4)
    monkeypatch.setattr(
        opt_flags_nvidia,
        "compute_num_warps",
        lambda _block_m, _block_n, _is_persistent, _precision_config, caller_constraints:
            caller_constraints.get("num_warps", 8),
    )
    monkeypatch.setattr(opt_flags_nvidia, "is_blackwell_mx_lhs_dense_rhs", lambda *_args: False)

    precision_config = PrecisionConfig(
        a_mx_tensor_scale=torch.ones(1) if has_act_tensor_scale else None,
        b_mx_tensor_scale=torch.ones(1) if has_weight_tensor_scale else None,
    )
    routing_data = (
        SimpleNamespace(expected_slice_size=1024, n_slices=64, slice_sizes=torch.ones(64))
        if is_ragged
        else None
    )
    flags = opt_flags.make_default_opt_flags_nvidia(
        out_dtype=BF16,
        lhs_dtype=FP4,
        rhs_dtype=FP4,
        precision_config=precision_config,
        batch_size=1,
        m=rows,
        n=4096,
        k=4096,
        routing_data=routing_data,
        can_use_persistent_tma=True,
        can_use_split_k=False,
        enforce_bitwise_invariance=False,
        epilogue_effective_itemsize=6,
        x_transpose=False,
        has_y_acc_in=False,
        constraints=constraints,
        intermediate_out_dtype=torch.float32,
        epilogue_reduction_n=epilogue_reduction_n,
    )

    assert (
        flags.block_m,
        flags.block_n,
        flags.block_k,
        flags.num_warps,
        flags.num_stages,
        flags.epilogue_subtile,
    ) == expected


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
