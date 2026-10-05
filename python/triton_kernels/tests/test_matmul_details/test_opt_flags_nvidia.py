from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_cuda

from triton_kernels.matmul import Epilogue, FnName, FusedActivation, PrecisionConfig, matmul, matmul_torch
from triton_kernels.matmul_details import opt_flags
from triton_kernels.matmul_details._matmul import _compute_packed_n_w
from triton_kernels.matmul_details.opt_flags import InapplicableConstraint, scoped_opt_flags_constraints
from triton_kernels.matmul_details.opt_flags_details import opt_flags_nvidia
from triton_kernels.numerics_details.mxfp import MXFP_BLOCK_SIZE, NVFP_BLOCK_SIZE, downcast_to_mxfp, quantize_nvfp4_fn
from triton_kernels.specialize import FnSpecs
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.tensor import BF16, FP4, UINT8, Storage, Tensor, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details import layout
from triton_kernels.tensor_details.layout import BlackwellMX4ValueShuffledLayout
from triton_kernels.tensor_details.layout_details.blackwell_scale import BlackwellActMXScaleLayout, BlackwellMXScaleLayout
from triton_kernels.tensor_details.ragged_tensor import make_ragged_tensor_metadata_torch
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


@pytest.mark.parametrize("block_m,num_warps,epilogue_subtile,expected_stages", [
    (32, 4, 1, 4),
    (128, 8, 2, 1),
    (128, 4, 1, 3),
    (128, 4, 2, 3),
    (128, 4, 4, 3),
    (64, 4, 4, 4),
    (64, 8, 4, 3),
])
def test_fp4_reduction_stage_budget(monkeypatch, block_m, num_warps, epilogue_subtile, expected_stages):
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(shared_memory_per_block_optin=232448, multi_processor_count=148))
    monkeypatch.setattr(opt_flags_nvidia.target_info, "cuda_capability_geq", lambda *_: True)
    monkeypatch.setattr(opt_flags, "cuda_capability_geq", lambda *_: True)
    scale = torch.empty((1, ), dtype=torch.float8_e4m3fn)
    precision = PrecisionConfig(a_mx_scale=scale, b_mx_scale=scale)
    flags = opt_flags.make_default_opt_flags_nvidia(
        FP4,
        FP4,
        FP4,
        precision,
        1,
        256,
        512,
        1024,
        None,
        True,
        False,
        False,
        8,
        False,
        False,
        {
            "block_m": block_m, "block_n": 256, "block_k": 256, "num_warps": num_warps, "epilogue_subtile":
            epilogue_subtile, "is_persistent": True, "split_k": 1
        },
        torch.float32,
        mx_block_size=16,
        epilogue_reduction_n=2,
    )
    assert flags.num_stages == expected_stages


@pytest.mark.parametrize("overrides,expected_stages", [
    ({"epilogue_reduction_n": 1}, 1),
    ({"epilogue_effective_itemsize": 1}, 3),
    ({"out_dtype": BF16}, 1),
    ({"has_y_acc_in": True}, 2),
    ({"is_persistent": False}, 4),
])
def test_fp4_reduction_stage_budget_other_epilogues(monkeypatch, overrides, expected_stages):
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(shared_memory_per_block_optin=232448))
    monkeypatch.setattr(opt_flags_nvidia.target_info, "cuda_capability_geq", lambda *_: True)
    scale = torch.empty((1, ), dtype=torch.float8_e4m3fn)
    args = {
        "precision_config": PrecisionConfig(a_mx_scale=scale, b_mx_scale=scale),
        "is_persistent": True,
        "block_m": 128,
        "block_n": 256,
        "block_k": 256,
        "out_dtype": FP4,
        "lhs_dtype": FP4,
        "rhs_dtype": FP4,
        "x_transpose": False,
        "epilogue_effective_itemsize": 8,
        "has_y_acc_in": False,
        "mx_block_size": 16,
        "epilogue_reduction_n": 2,
        "epilogue_subtile": 2,
        "num_warps": 8,
        "occupancy_target": 1,
    }
    assert opt_flags_nvidia.compute_num_stages(**(args | overrides)) == expected_stages


@pytest.mark.parametrize("block_m,swizzle_lhs_scale", [(32, False), (64, False), (128, False), (128, True)])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("epilogue_subtile", [1, 2, 4])
def test_matmul_fp4_reduction_shared_memory(device, monkeypatch, block_m, num_warps, epilogue_subtile,
                                            swizzle_lhs_scale):
    if device != "cuda" or not torch.cuda.is_available() or not is_cuda():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires Blackwell")

    torch.manual_seed(0)
    m, n, k = 255, 512, 1024
    a, a_scale = downcast_to_mxfp(torch.randn((m, k), device=device, dtype=torch.bfloat16), torch.uint8, axis=-1,
                                  scale_dtype=torch.float8_e4m3fn, microblock_size=16)
    b, b_scale = downcast_to_mxfp(
        torch.randn((n, k), device=device, dtype=torch.bfloat16).T * 0.01, torch.uint8, axis=-2,
        scale_dtype=torch.float8_e4m3fn, microblock_size=16)
    a, b = wrap_torch_tensor(a, dtype=FP4), wrap_torch_tensor(b, dtype=FP4)
    a_scale, b_scale = wrap_torch_tensor(a_scale), wrap_torch_tensor(b_scale)
    if swizzle_lhs_scale:
        a_scale = convert_layout(a_scale, BlackwellActMXScaleLayout(None))
    b_scale = convert_layout(b_scale, BlackwellMXScaleLayout())
    c_scale = torch.empty((m, n // 2 // NVFP_BLOCK_SIZE.value), device=device, dtype=torch.float8_e4m3fn)
    precision = PrecisionConfig(
        a_mx_scale=a_scale,
        b_mx_scale=b_scale,
        c_mx_scale=c_scale,
        a_microblock_size=16,
        b_microblock_size=16,
        c_microblock_size=16,
        c_value_pack_factor=2,
        out_dtype=torch.uint8,
    )
    activation = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2), (1.0, 7.0))
    epilogue = Epilogue(FnSpecs(FnName.QUANTIZE_NVFP4.name, quantize_nvfp4_fn, (), ()), effective_itemsize=8)
    constraints = {
        "block_m": block_m, "block_n": 256, "block_k": 256, "num_warps": num_warps, "epilogue_subtile":
        epilogue_subtile, "is_persistent": True, "split_k": 1
    }
    with scoped_opt_flags_constraints(constraints | {"num_stages": 1}):
        expected = matmul(a, b, None, precision_config=precision, fused_activation=activation, epilogue=epilogue)
    expected_scale = c_scale.clone()
    c_scale.fill_(float("nan"))

    launches = []
    original_run = triton.runtime.JITFunction.run

    def capture(kernel, *args, **kwargs):
        result = original_run(kernel, *args, **kwargs)
        if result.name.startswith("_p_matmul"):
            launches.append(result.metadata)
        return result

    monkeypatch.setattr(triton.runtime.JITFunction, "run", capture)
    with scoped_opt_flags_constraints(constraints):
        actual = matmul(a, b, None, precision_config=precision, fused_activation=activation, epilogue=epilogue)
    torch.cuda.synchronize()
    assert launches
    assert all(launch.shared <= torch.cuda.get_device_properties(0).shared_memory_per_block_optin
               for launch in launches)
    assert torch.equal(actual, expected)
    assert torch.equal(c_scale.view(torch.uint8), expected_scale.view(torch.uint8))


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


def test_matmul_clc_rejects_ragged_m_grid(device):
    if device != "cuda" or not torch.cuda.is_available() or not is_cuda():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell or newer")

    slice_sizes = torch.tensor([64, 0, 64], dtype=torch.int32, device=device)
    metadata = make_ragged_tensor_metadata_torch(slice_sizes, 128)
    a = torch.randn((128, 64), device=device, dtype=torch.bfloat16)
    b = torch.randn((3, 64, 128), device=device, dtype=torch.bfloat16)
    with scoped_opt_flags_constraints({"clc": True}):
        with pytest.raises(InapplicableConstraint, match="host-known grid"):
            matmul(a, b, None, a_ragged_metadata=metadata)


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
