# isort: off
# fmt: off
from dataclasses import dataclass, fields, replace
import itertools
import pytest
import torch
from typing import Union
import triton
# routing utilities
from triton_kernels.routing import routing
# matmul utilities
import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
from triton_kernels.matmul_ogs import FlexCtx, InnerRoutingData, PrecisionConfig, FusedActivation, FnSpecs, FnName, Epilogue
from triton_kernels.matmul_ogs import matmul_ogs_set_idle_sms, matmul_ogs, matmul_ogs_torch
from triton_kernels.swiglu import swiglu, swiglu_fn, PrecisionConfig as SwiGLUPrecisionConfig
from triton_kernels.tensor import convert_layout, wrap_torch_tensor, FP4
from triton_kernels.tensor_details import layout
# numerics utilities
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp, upcast_from_mxfp, quantize_mxfp8_fn, downcast_to_mxfp_torch, upcast_from_mxfp_torch, MXFP_BLOCK_SIZE
# testing utilities
from triton_kernels.testing import assert_close, compute_actual_scale
# target-specific utilities
from triton_kernels.target_info import is_hip, is_hip_cdna3, is_cuda, is_hip_cdna4

# ---------------
# initialize data
# ---------------


def alloc_rand(shape, device, dtype, requires_grad=True):
    if dtype.itemsize == 1:
        tmp = 2**-(torch.randint(4, 8, shape, device=device, dtype=torch.float16))
        return tmp.to(dtype).requires_grad_(requires_grad)
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)


def alloc_rand_like(x):
    return alloc_rand(x.shape, x.device, x.dtype, x.requires_grad)


def mask_indx(idx, n_expts_act):
    idx.src_indx[idx.dst_indx[-n_expts_act:]] = -1
    idx.dst_indx[-n_expts_act:] = -1
    return idx


def init_routing_data(m, n_expts_tot, n_expts_act, do_gather, do_scatter, device="cuda"):
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device, requires_grad=True)
    routing_data, gather_idx, scatter_idx = routing(logits, n_expts_act)
    routing_data.gate_scal = None
    gather_idx = gather_idx if do_gather else None
    scatter_idx = scatter_idx if do_scatter else None
    return m, routing_data, gather_idx, scatter_idx


def init_compute_data(m, n, k, rdata, gindx, sindx, n_expts_tot, n_expts_act, mode, act_dtype, weight_dtype,
                      has_y_gammas, requires_grad=True, device="cuda",
                      inner_expt_opt=None, padding_block_k=None):
    torch.manual_seed(0)
    assert mode in {'batched', "plain", 'ragged'}
    if inner_expt_opt is not None:
        assert gindx is None and sindx is None
        # Rotate tensor shapes (dw = x.T @ dy)
        m, k = k, m * n_expts_act + padding_block_k * n_expts_tot
        in_m = m
    else:
        in_m = m * (n_expts_act if gindx is None else 1)
    shape_x = (n_expts_tot, in_m, k) if mode == 'batched' else (in_m, k)
    shape_batch = tuple() if (mode == "plain" or inner_expt_opt is not None) else (n_expts_tot, )
    x = alloc_rand(shape_x, device=device, dtype=act_dtype, requires_grad=requires_grad)
    w = alloc_rand(shape_batch + (k, n), device=device, dtype=weight_dtype, requires_grad=requires_grad)
    bias = alloc_rand(shape_batch + (n, ), device=device, dtype=torch.float32, requires_grad=requires_grad)
    gs0 = 2**torch.randint(-5, 0, (m * n_expts_act, ), device=device, dtype=torch.float32, requires_grad=requires_grad)
    gs1 = 2**torch.randint(-5, 0, (m * n_expts_act, ), device=device, dtype=torch.float32, requires_grad=requires_grad)
    gs0 = gs0.detach().requires_grad_(requires_grad)
    gs1 = gs1.detach().requires_grad_(requires_grad)
    if mode == 'batched' or (not has_y_gammas) or (has_y_gammas and (gindx is not None) and act_dtype.itemsize >= 2):
        gs0 = None
        gs1 = None
    if "float8" in str(weight_dtype) and torch.cuda.get_device_capability()[0] < 10:
        w = w.transpose(-1, -2).contiguous().transpose(-1, -2)

    def _apply_padding_and_fill_unused_part_with_nan(t, is_padded):
        nan_val = float("nan")
        if t.element_size() == 1:
            t = t.view(torch.int8)
            nan_val = 127

        start = 0
        if is_padded:
            for this_expt_nrows in rdata.expt_hist.tolist():
                end = start + this_expt_nrows
                padding_end = start + triton.cdiv(this_expt_nrows, padding_block_k) * padding_block_k
                t[end:padding_end, :] = 0
                start = padding_end
            assert start <= t.shape[0]
            t[start:, :] = nan_val
        else:
            n_actual_rows = rdata.expt_hist.sum().item()
            if n_actual_rows + padding_block_k < t.shape[0]:
                t[n_actual_rows+padding_block_k:, :] = nan_val

    if inner_expt_opt is not None:
        bias = None
        _apply_padding_and_fill_unused_part_with_nan(x.T, "pad_x" in inner_expt_opt)
        _apply_padding_and_fill_unused_part_with_nan(w, "pad_w" in inner_expt_opt)

    return x, w, bias, gs0, gs1


# ---------------
# numerics stuff
# ---------------


def init_precision(out_dtype, act_use_flexpoint, weight_dtype, weight_mxfp, mode, n_expts_tot=1, expt_is_inner=False, device="cuda"):
    weight_use_flexpoint = weight_dtype.itemsize == 1 and not weight_mxfp
    # flexpoint
    make_tensor = lambda val0, val1: torch.tensor([val0, val1] * (n_expts_tot // 2) +
                                                  ([val0]
                                                   if n_expts_tot % 2 else []), dtype=torch.float32, device=device)
    make_scalar = lambda val: torch.tensor([val], dtype=torch.float32, device=device)
    make = lambda val0, val1, is_tensor: make_tensor(val0, val1) if is_tensor else make_scalar(val0)

    in_flex_data = lambda scale, use_flex: InFlexData(dtype=out_dtype, scale=make_scalar(scale)
                                                      ) if use_flex else InFlexData()
    flex_ctx = FlexCtx(
        lhs_data=in_flex_data(1.25, act_use_flexpoint),
        rhs_data=InFlexData(
            dtype=weight_dtype,
            scale=make(1.50, 1.25, not expt_is_inner),
        ) if weight_use_flexpoint else InFlexData(),
        out_data=OutFlexData(
            dtype=out_dtype,
            expected_scale=make(4.00, 5.00, mode == "batched" or expt_is_inner),
            actual_scale=make(0, 0, mode == "batched" or expt_is_inner),
            checksum_scale=None,
        ) if act_use_flexpoint else OutFlexData(),
    )
    return PrecisionConfig(flex_ctx=flex_ctx, acc_scale=2.0 if act_use_flexpoint or weight_use_flexpoint else 1.0,
                           out_dtype=out_dtype)


def apply_precision(x_tri, w_tri, bias_tri, gs0_tri, gs1_tri, precision_config):
    flex_ctx = precision_config.flex_ctx

    def apply(x, scale):
        if scale is None:
            x = x.clone()
        elif scale.numel() == 1:
            x = x.float() * scale
        else:
            assert x.ndim == 3
            assert scale.numel() == x.shape[0]
            x = x.float() * scale[:, None, None]
        return x.detach().requires_grad_()

    return (
        apply(x_tri, flex_ctx.lhs_data.scale),
        apply(w_tri, flex_ctx.rhs_data.scale),
        None if bias_tri is None else apply(bias_tri, None),
        None if gs0_tri is None else apply(gs0_tri, None),
        None if gs1_tri is None else apply(gs1_tri, None),
    )


def dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    return torch.uint8 if dtype_str == "float4_e2m1" else getattr(torch, dtype_str)


# Scope to ensure that the opt_flags_constraints are reset after the test
@pytest.fixture
def opt_flags_scope(request):
    yield
    opt_flags.reset_opt_flags_constraints()


# ---------------
# unit tests
# ---------------


@dataclass
class Case:
    m: int
    n: int
    k: int
    mode: str
    act_dtype_str: str
    weight_dtype_str: str
    n_expts_tot: int = 1
    n_expts_act: int = 1
    split_k: int = 1
    hbm_swizzling: bool = False
    epilogue_subtile: Union[int, None] = None
    x_transpose: bool = False
    w_transpose: bool = False
    y_transpose: bool = False


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case)) for case in [
            # Zero-sized args:
            Case(0, 5, 7, "ragged", "float16", "float16"),
            Case(5, 0, 7, "ragged", "float16", "float16"),
            Case(5, 7, 0, "ragged", "float16", "float16"),
            Case(0, 5, 7, "batched", "float16", "float16"),
            Case(5, 0, 7, "batched", "float16", "float16"),
            Case(5, 7, 0, "batched", "float16", "float16"),
            # Non-mx types:
            Case(16, 256, 256, "ragged", "float16", "float16", 128, 4),
            Case(16, 256, 256, "ragged", "float16", "float16", 128, 4, split_k=3),
            Case(16, 256, 256, "ragged", "float16", "float16", 128, 4, split_k=3),
            Case(300, 400, 400, "batched", "float8_e5m2", "float8_e5m2", 5, 1),
            Case(16, 256, 256, "batched", "float16", "float16", 5, 1),
            Case(16, 256, 256, "ragged", "float16", "float16", 3, 1),
            Case(256, 256, 256, "ragged", "float16", "float16", 4, 1),
            Case(256, 256, 256, "ragged", "float16", "float16", 4, 1, split_k=3),
            Case(300, 400, 400, "batched", "float16", "float16", 5, 1),
            Case(300, 400, 400, "ragged", "float16", "float16"),
            Case(300, 400, 400, "ragged", "float8_e5m2", "float8_e5m2"),
            Case(1000, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 3, 1),
            Case(600, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 4, 2, epilogue_subtile=1),
            Case(600, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 4, 2, epilogue_subtile=2),
            Case(600, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 4, 2, epilogue_subtile=4),
            Case(600, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 4, 2),
            Case(600, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 4, 2, split_k=2),
            Case(1000, 400, 400, "ragged", "float16", "float16", 3, 1),
            Case(1000, 700, 700, "ragged", "float16", "float16", 8, 2),
            Case(1000, 700, 700, "ragged", "float16", "float16", 8, 2, split_k=9),
            Case(16, 16, 1000, "batched", "float16", "float16", 5, 1, split_k=None),
            Case(16, 16, 1000, "batched", "float8_e5m2", "float8_e5m2", 5, 1, split_k=None),
            Case(16, 16, 2048, "batched", "float8_e5m2", "float8_e5m2", 6, 1, split_k=5),
            # mx types:
            Case(16, 256, 256, "plain", "bfloat16", "mxfloat4_e2m1", 1, 1),
            Case(16, 256, 256, "plain", "bfloat16", "mxfloat4_e2m1", 1, 1, hbm_swizzling=True),
            Case(16, 256, 256, "ragged", "bfloat16", "mxfloat4_e2m1", 1, 1),
            Case(16, 256, 256, "ragged", "bfloat16", "mxfloat4_e2m1", 1, 1, hbm_swizzling=True),
            Case(1000, 700, 700, "batched", "bfloat16", "mxfloat4_e2m1", 8, 2),
            Case(1000, 700, 700, "batched", "bfloat16", "mxfloat4_e2m1", 8, 2, hbm_swizzling=True),
            Case(1000, 700, 700, "ragged", "bfloat16", "mxfloat4_e2m1", 8, 2, split_k=9),
            Case(1000, 512, 256, "ragged", "bfloat16", "mxfloat4_e2m1", 8, 2, split_k=9, hbm_swizzling=True),
            Case(300, 400, 400, "ragged", "bfloat16", "mxfloat8_e4m3fn", 8, 4),
            Case(300, 400, 400, "ragged", "bfloat16", "mxfloat8_e4m3fn", 8, 4, hbm_swizzling=True),
            Case(300, 400, 400, "batched", "bfloat16", "mxfloat8_e5m2", 32, 4),
            Case(1000, 700, 2, "batched", "bfloat16", "mxfloat4_e2m1", 8, 2),
            Case(1, 2880, 2880, "ragged", "bfloat16", "mxfloat4_e2m1", 128, 4),
            Case(16, 256, 256, "ragged", "float8_e5m2", "mxfloat4_e2m1", 128, 4, hbm_swizzling=True),
            Case(1000, 704, 832, "batched", "float8_e5m2", "mxfloat4_e2m1", 3, 1, hbm_swizzling=True),
            Case(1000, 704, 832, "batched", "float8_e5m2", "mxfloat4_e2m1", 3, 1, hbm_swizzling=True),
            Case(1000, 704, 832, "batched", "float8_e5m2", "mxfloat4_e2m1", 3, 1),
            Case(1000, 704, 800, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 2, split_k=9),
            Case(1000, 704, 800, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 2, split_k=9, hbm_swizzling=True),
            Case(1000, 704, 800, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 2),
            Case(1000, 704, 800, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 2, hbm_swizzling=True),
            Case(300, 400, 400, "ragged", "float8_e5m2", "mxfloat8_e4m3fn", 8, 4),
            Case(300, 400, 400, "ragged", "float8_e5m2", "mxfloat8_e4m3fn", 8, 4, hbm_swizzling=True),
            Case(300, 400, 832, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 4),
            Case(300, 400, 832, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 4, hbm_swizzling=True),
            Case(300, 400, 400, "batched", "float8_e5m2", "mxfloat8_e4m3fn", 32, 4),
            Case(300, 400, 400, "batched", "float8_e5m2", "mxfloat8_e4m3fn", 32, 4, hbm_swizzling=True),
            Case(256, 256, 256, "ragged", "float8_e5m2", "mxfloat4_e2m1", 128, 4, hbm_swizzling=True),
            Case(256, 256, 256, "ragged", "float8_e5m2", "mxfloat4_e2m1", 128, 4, hbm_swizzling=False),
            Case(16, 256, 256, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", 128, 4, hbm_swizzling=True),
            Case(1000, 704, 800, "batched", "mxfloat8_e4m3fn", "mxfloat4_e2m1", 3, 1, hbm_swizzling=True),
            Case(1000, 704, 800, "batched", "mxfloat8_e4m3fn", "mxfloat4_e2m1", 2, 1),
            Case(1000, 704, 800, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", 8, 2, split_k=9),
            Case(1000, 704, 800, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", 8, 2, split_k=9, hbm_swizzling=True),
            Case(1000, 704, 800, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", 8, 2),
            Case(1000, 704, 800, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", 8, 2, hbm_swizzling=True),
            Case(300, 400, 400, "ragged", "mxfloat8_e4m3fn", "mxfloat8_e4m3fn", 8, 4),
            Case(300, 512, 512, "ragged", "mxfloat8_e4m3fn", "mxfloat8_e4m3fn", 8, 4),
            Case(300, 400, 400, "ragged", "mxfloat8_e4m3fn", "mxfloat8_e4m3fn", 8, 4, hbm_swizzling=True),
            Case(300, 400, 800, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", 8, 4),
            Case(300, 400, 800, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", 8, 4, hbm_swizzling=True),
            Case(300, 400, 400, "batched", "mxfloat8_e4m3fn", "mxfloat8_e4m3fn", 32, 4),
            Case(300, 400, 400, "batched", "mxfloat8_e4m3fn", "mxfloat8_e4m3fn", 32, 4, hbm_swizzling=True),
            # AMD
            Case(300, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz"),
            Case(1000, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz", 3, 1),
            Case(600, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz", 4, 2),
            Case(600, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz", 4, 2, split_k=2),
            Case(300, 400, 400, "ragged", "float8_e4m3fn", "float8_e4m3fn"),
            Case(1000, 400, 400, "ragged", "float8_e4m3fn", "float8_e4m3fn", 3, 1),
            Case(600, 400, 400, "ragged", "float8_e4m3fn", "float8_e4m3fn", 4, 2),
        ] + [
            Case(320, 400, 400, mode, dtype, dtype, n_expts_tot, n_expts_act,
                 x_transpose=x_transpose, w_transpose=w_transpose, y_transpose=y_transpose)
            for (mode, n_expts_tot, n_expts_act) in (
                ("batched", 1, 1),
                ("ragged", 8, 4),
                ("ragged", 32, 4),
            )
            for dtype in ("float16", "float8_e5m2")
            for x_transpose in (False, True)
            for w_transpose in (False, True)
            for y_transpose in (False, True)
        ]
    ],
)
@pytest.mark.parametrize("block_m", [16, 128])
@pytest.mark.parametrize("do_gather, do_scatter, fused_scatter, inner_expt_opt", [
    (False, False, False, None),
    (True, False, False, None),
    (False, True, False, None),
    (False, True, True, None),
    (True, True, False, None),
    (True, True, True, None),
    (False, False, False, "pad_w"),
    (False, False, False, "pad_x"),
])
@pytest.mark.parametrize("has_y_gammas", [False, True])
@pytest.mark.parametrize("is_persistent", [False, True])
def test_op(m, n, k, split_k, do_gather, do_scatter, fused_scatter, inner_expt_opt, has_y_gammas, is_persistent, n_expts_tot,
            n_expts_act, mode, act_dtype_str, weight_dtype_str, block_m, hbm_swizzling, epilogue_subtile,
            x_transpose, w_transpose, y_transpose,
            device, opt_flags_scope, fresh_knobs):
    # TODO: remove when Triton FP8 supports proper RTNE
    if is_cuda():
        if "float8" in weight_dtype_str and torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("Float8 not tested on A100")
        if "float16" in act_dtype_str and "mx" in weight_dtype_str and torch.cuda.get_device_capability()[0] >= 10:
            pytest.skip("float16 x mx not supported with cuda capability >= 10")
        if weight_dtype_str.startswith("mx"):
            if "float8" in act_dtype_str and torch.cuda.get_device_capability()[0] < 10:
                pytest.skip("float8 x mx not supported with cuda capability < 10")
        if n == 2880 and k == 2880 and torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("Not enough memory on A100")

    elif is_hip():
        if "float8" in act_dtype_str and "mx" in weight_dtype_str and not is_hip_cdna4():
            pytest.skip("float8 x mx only supported on CDNA4")
        if "float8" in act_dtype_str and "mxfloat8" in weight_dtype_str:
            pytest.skip("NYI: float8 x mxfloat8 not tested on AMD GPU")
        if act_dtype_str.startswith("mx") and weight_dtype_str.startswith("mx"):
            pytest.skip("NYI: mx x mx not tested on AMD GPU")
        if is_persistent:
            pytest.skip("NYI: Persistent kernel not supported on AMD GPU")
        if split_k is not None and split_k > 1:
            pytest.skip("splitK hasn't been fully tested on AMD GPU.")

    if "float8_e4m3fnuz" in (weight_dtype_str, act_dtype_str) and not is_hip_cdna3():
        pytest.skip("float8_e4m3fnuz only tested on AMD CDNA3 Platform")

    if fused_scatter and split_k is not None and split_k > 1:
        pytest.skip("fused scatter scratchpad not supported with split_k")

    if hbm_swizzling:
        if is_hip():
            if not is_hip_cdna4():
                pytest.skip("Scale preshuffling on AMD GPU has not been emulated on non-CDNA4 arch yet.")
            if "mx" not in weight_dtype_str:
                pytest.skip("Non-scale swizzling not supported on CDNA4 yet")
            if n % 32 != 0 or k % (32 * 8) != 0:
                pytest.skip(f"Shape {m}x{n}x{k} is not supported for scale swizzling on AMD GPU")
        if torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("NYI. Ampere swizzling.")
        if torch.cuda.get_device_capability()[0] < 10:
            if "mxfloat4" not in weight_dtype_str:
                pytest.skip("NYI. Hopper swizzling just implemented for mxfp4.")
            if k % 64 != 0 or n % 64 != 0:
                # Automatic padding not implemented for Hopper swizzle
                pytest.skip("Hopper swizzling acts on a 64x64 tile (4x1 mma tiles).")

    expt_is_inner = (inner_expt_opt is not None)
    if expt_is_inner and (mode != "ragged" or "mx" in act_dtype_str or "mx" in weight_dtype_str):
        pytest.skip("Not supported yet")

    # launch metadata for batched / mx types may not work yet.
    torch.manual_seed(0)

    block_k = None
    if is_persistent and weight_dtype_str.startswith("mx") and torch.cuda.get_device_capability()[0] < 10:
        # Override block_k for testing correctness. The default is temporarily 128 for
        # performance reasons which doesn't work with persistent matmul.
        # TODO: revisit when Triton is better for H100 + MXFP4
        block_k = 256

    constraints = {
        "block_m": block_m,
        "block_k": block_k,
        "split_k": split_k,
        "fused_scatter": fused_scatter,
        "is_persistent": is_persistent,
        "epilogue_subtile": epilogue_subtile,
    }

    if is_hip() and hbm_swizzling and "float4" in weight_dtype_str:
        # Minimum block size to satisfy scale preshuffling
        constraints.update({
            "block_m": 32,
            "block_n": 32,
            "block_k": 256
        })

    opt_flags.update_opt_flags_constraints(constraints)

    weight_mxfp = weight_dtype_str.startswith("mx")
    if weight_mxfp:
        weight_dtype_str = weight_dtype_str[2:]
    act_mxfp8 = act_dtype_str.startswith("mx")
    act_is_float8 = act_dtype_str.startswith("float8")
    if act_mxfp8:
        act_dtype_str = act_dtype_str[2:]
        quantize_mxfp8_spec = FnSpecs(
            FnName.QUANTIZE_MXFP8.name, quantize_mxfp8_fn, (), ()
        )

    test_bwd = False
    weight_dtype = dtype_str_to_torch(weight_dtype_str)
    act_dtype = dtype_str_to_torch(act_dtype_str)
    precision_opt = init_precision(act_dtype, act_is_float8, weight_dtype, weight_mxfp,
                                   mode, n_expts_tot, expt_is_inner, device=device)
    # precision_opt.x_pad_trans_requires_flexpoint = False
    if mode == "ragged":
        m, rdata, gindx, sindx = init_routing_data(m, n_expts_tot, n_expts_act, do_gather, do_scatter,
                                                   device=device)
    else:
        rdata = gindx = sindx = None

    padding_block_k = 32
    x_tri, w_tri, bias_tri, gs0_tri, gs1_tri = init_compute_data(m, n, k, rdata, gindx, sindx, n_expts_tot, n_expts_act,
                                                                 mode, torch.bfloat16 if act_mxfp8 else act_dtype,  #
                                                                 torch.bfloat16 if weight_mxfp else weight_dtype,
                                                                 has_y_gammas, requires_grad=test_bwd, device=device,
                                                                 inner_expt_opt=inner_expt_opt, padding_block_k=padding_block_k)
    x_ref, w_ref, bias_ref, gs0_ref, gs1_ref = apply_precision(x_tri, w_tri, bias_tri, gs0_tri, gs1_tri, precision_opt)

    if x_transpose:
        x_tri = x_tri.detach().transpose(-1, -2).contiguous().transpose(-1, -2).requires_grad_(test_bwd)
    if w_transpose:
        w_tri = w_tri.detach().transpose(-1, -2).contiguous().transpose(-1, -2).requires_grad_(test_bwd)
    if y_transpose:
        if mode == "batched":
            yT_shape = (n_expts_tot, n, x_tri.shape[-2])
        elif expt_is_inner:
            yT_shape = (n_expts_tot, n, k)
        elif sindx is not None:
            yT_shape = (n, m)
        else:
            n_rows = x_tri.shape[-2] if gindx is None else gindx.dst_indx.shape[0]
            yT_shape = (n, n_rows)
        y_tri_in = torch.empty(yT_shape, dtype=act_dtype, device=device).transpose(-1, -2)
    else:
        y_tri_in = None

    if w_tri.shape[0] == 1 and mode != "batched":
        # Test the case when weight has dim 2, i.e., shape (K, N).
        w_tri = w_tri.squeeze(0).detach().requires_grad_(test_bwd)
        w_ref = w_ref.squeeze(0).detach().requires_grad_(test_bwd)

    if weight_mxfp:
        mx_axis = w_tri.ndim - 2
        # compute layouts
        w_layout, w_layout_opts = layout.StridedLayout, dict()
        w_scale_layout, w_scale_layout_opts = layout.StridedLayout, dict()
        if hbm_swizzling and "float4" in weight_dtype_str:
            w_layout, w_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=mx_axis)
            w_scale_layout, w_scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
                mx_axis=mx_axis, num_warps=8)
        # downcast to mxfp
        w_tri, w_scale_tri = downcast_to_mxfp(w_tri, weight_dtype, axis=mx_axis)
        w_ref = upcast_from_mxfp(w_tri, w_scale_tri, torch.bfloat16, axis=mx_axis)
        w_tri_dtype = FP4 if "float4" in weight_dtype_str else weight_dtype
        w_tri = wrap_torch_tensor(w_tri, w_tri_dtype)
        w_scale_tri = wrap_torch_tensor(w_scale_tri)
        # convert layouts
        w_tri = convert_layout(w_tri, w_layout, **w_layout_opts)
        w_scale_tri = convert_layout(w_scale_tri, w_scale_layout, **w_scale_layout_opts)
        precision_opt.weight_scale = w_scale_tri
    epilogue = None
    if act_mxfp8:
        x_tri, x_mx_scales_tri = downcast_to_mxfp(x_tri, act_dtype, axis=-1)
        x_ref = upcast_from_mxfp(x_tri, x_mx_scales_tri, torch.bfloat16, axis=-1)
        is_input_batched = x_tri.ndim == 3
        y_shape = x_tri.shape if is_input_batched else (1,) + x_tri.shape
        n_rows = y_shape[1] if gindx is None or mode == "batched" else gindx.dst_indx.shape[0]
        y_shape = (y_shape[0], n_rows, w_tri.shape[-1])
        if sindx is None or mode == "batched":
            if not is_input_batched:
                y_shape = (y_shape[1], y_shape[2])
        else:
            y_shape = (n_rows // rdata.n_expts_act, y_shape[-1])
        y_scale_shape = y_shape[:-1] + (triton.cdiv(y_shape[-1], MXFP_BLOCK_SIZE),)
        y_scale = torch.empty(y_scale_shape, dtype=torch.uint8, device=x_tri.device)
        precision_opt = replace(precision_opt, act_scale=x_mx_scales_tri, out_scale=y_scale)
        epilogue = Epilogue(quantize_mxfp8_spec, tuple(), tuple(), effective_itemsize=6.0)
    else:
        y_scale = None

    if mode == "batched":
        rdata, gindx, sindx = None, None, None
    flex = precision_opt.flex_ctx

    if expt_is_inner:
        inner_routing_data = InnerRoutingData(
            base=rdata, block_k=padding_block_k,
            x_is_padded="pad_x" in inner_expt_opt,
            w_is_padded="pad_w" in inner_expt_opt,
        )
        rdata = None
    else:
        inner_routing_data = None

    # triton
    try:
        tri_y = matmul_ogs(x_tri, w_tri, bias_tri, rdata, gindx, sindx, precision_opt,
                           gammas=gs1_ref, epilogue=epilogue, y=y_tri_in,
                           inner_routing_data=inner_routing_data)
    except (opt_flags.InapplicableConstraint, NotImplementedError):
        pytest.skip("inapplicable opt_flags constraint")
    if y_tri_in is not None:
        assert tri_y.data_ptr() == y_tri_in.data_ptr()
        assert tri_y.shape == y_tri_in.shape
        assert tri_y.stride() == y_tri_in.stride()
    # If split_k > 1, then the intermediate tensor is fp32.
    sep_gather = mode == "ragged" and do_gather and n_expts_act > 1 and split_k == 1
    sep_scatter = mode == "ragged" and do_scatter and n_expts_act > 1 and split_k == 1
    y_scale = flex.out_data.expected_scale if act_is_float8 else 1

    def round_x(x, idx):
        return x.to(act_dtype).to(torch.float32) if sep_gather else x

    round_y = lambda y: (y / y_scale).to(act_dtype).to(torch.float32) * y_scale if sep_scatter else y
    ref_y = matmul_ogs_torch(x_ref, w_ref, bias_ref,  #
                             rdata, gindx, sindx, round_x=round_x, round_y=round_y, gammas=gs1_ref,
                             inner_routing_data=inner_routing_data)

    def scale(val, scal):
        if scal is None:
            return val
        elif scal.numel() == 1:
            return val / scal
        else:
            assert val.ndim == 3
            return val / scal[:, None, None]

    if act_mxfp8:
        tri_y = upcast_from_mxfp(tri_y, precision_opt.out_scale, target_dtype=torch.bfloat16, axis=-1).to(ref_y.dtype)
        ref_y_quant, ref_y_scale = downcast_to_mxfp_torch(ref_y, act_dtype, axis=-1)
        ref_y = upcast_from_mxfp_torch(ref_y_quant, ref_y_scale, target_dtype=ref_y.dtype, axis=-1)
        maxtol = 4e-1
        rmstol = 4e-2
    elif weight_mxfp and "float4_e2m1" in weight_dtype_str:
        if act_is_float8:
            maxtol = 8e-2
        else:
            maxtol = 3e-2
        rmstol = None
    else:
        maxtol = None
        rmstol = None
    assert_close(scale(ref_y, flex.out_data.expected_scale), tri_y, maxtol=maxtol, rmstol=rmstol)

    if act_is_float8:
        tri_y_scale = flex.out_data.actual_scale.clone()
        ref_y_scale = compute_actual_scale(ref_y, tri_y.dtype, tri_y_scale.numel() > 1)
        assert torch.all((ref_y_scale - tri_y_scale).abs() < 1e-10), \
               f"ref_y_scale: {ref_y_scale}, tri_y_scale: {tri_y_scale.item()}"


# Test that we don't use unsupported block sizes.
@pytest.mark.parametrize("m", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("n", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("k", [8, 16, 32, 64, 128])
def test_small_batch_matmul(m, n, k):
    if is_hip():
        pytest.skip("Not fully tested on AMD")

    if m * n * k > 16384:
        pytest.skip()

    BATCH_SIZE = 10000

    def _make_tensor(shape, dtype, trans):
        if trans:
            shape = (shape[0], shape[2], shape[1])
        t = alloc_rand(shape, "cuda", dtype)
        return t.transpose(1, 2) if trans else t

    for x_transpose, w_transpose, bias, dtype in itertools.product(
        (False, True),
        (False, True),
        (False, True),
        (torch.float16, torch.bfloat16, torch.float8_e5m2),
    ):
        if (
            torch.cuda.get_device_capability()[0] < 10
            and dtype is torch.float8_e5m2
            and (not w_transpose)
        ):
            continue  # Not supported

        x = _make_tensor((BATCH_SIZE, m, k), dtype, x_transpose)
        w = _make_tensor((BATCH_SIZE, k, n), dtype, w_transpose)
        bias = _make_tensor((BATCH_SIZE, n), torch.float32, False) if bias else None
        tri_y = matmul_ogs(x, w, bias)

        # ref_y = matmul_ogs_torch(x.float(), w.float(), bias)

        # This is faster than matmul_ogs_torch.
        ref_y = torch.bmm(x.float(), w.float())
        if bias is not None:
            ref_y += bias[:, None, :]

        assert_close(
            ref_y,
            tri_y,
            maxtol=4e-1 if dtype is torch.float8_e5m2 else None,
            rmstol=4e-2 if dtype is torch.float8_e5m2 else None,
        )


def test_set_idle_sms():
    if not is_cuda():
        pytest.skip("Only supported on CUDA")
    from triton_kernels.matmul_ogs_details.opt_flags import make_opt_flags
    num_idle_sms = 24
    matmul_ogs_set_idle_sms(num_idle_sms)
    flags = make_opt_flags(torch.float32, torch.float32, torch.float32, PrecisionConfig(), \
                           1, 1024, 1024, 1024, None, True, False, 1, False, False, None)
    assert flags.idle_sms == num_idle_sms


@pytest.mark.parametrize("m, n, k, mode", [
    (1200, 704, 608, "ragged"),
    (800, 800, 400, "batched"),
])
@pytest.mark.parametrize("split_k", [1, 2])
@pytest.mark.parametrize("do_gather, do_scatter, fused_scatter", [
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (True, True, False),
    (True, True, True),
])
@pytest.mark.parametrize("is_persistent, epilogue_subtile", [
    (False, None),
    (True, 1),
    (True, 4),
])
@pytest.mark.parametrize("swiglu_alpha, swiglu_limit", [
    (1.1, 1.4),
    (1.0, 1.2),
    (0.7, 1.0),
])
def test_fused_act(m, n, k, mode, split_k, do_gather, do_scatter, fused_scatter, is_persistent, epilogue_subtile,
                   swiglu_alpha, swiglu_limit, device, opt_flags_scope):
    if fused_scatter and split_k > 1:
        pytest.skip("fused scatter scratchpad not supported with split_k")
    torch.manual_seed(0)
    constraints = {
        "is_persistent": is_persistent,
        "epilogue_subtile": epilogue_subtile,
        "split_k": split_k,
        "fused_scatter": fused_scatter,
    }
    n_expts_tot, n_expts_act = 1, 1
    opt_flags.update_opt_flags_constraints(constraints)

    weight_dtype, act_dtype = torch.float16, torch.float16
    if mode == "ragged":
        m, rdata, gindx, sindx = init_routing_data(m, n_expts_tot, n_expts_act, do_gather, do_scatter,
                                                   device=device)
    else:
        rdata = gindx = sindx = None

    precision_opt = init_precision(act_dtype, str(act_dtype).startswith("torch.float8"), weight_dtype, False, mode, n_expts_tot, device=device)
    x, w, bias, _, _ = init_compute_data(m, n, k, rdata, gindx, sindx, n_expts_tot, n_expts_act, mode,
                                         act_dtype, weight_dtype, False, requires_grad=False, device=device)

    if mode == "batched":
        rdata, gindx, sindx = None, None, None

    try:
        a = swiglu(matmul_ogs(x, w, bias, rdata, gindx, sindx, precision_opt), swiglu_alpha,
                   precision_config=SwiGLUPrecisionConfig(swiglu_limit))
        b = matmul_ogs(
            x, w, bias, rdata, gindx, sindx, precision_opt,
            fused_activation=FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")),
                                             (swiglu_alpha, swiglu_limit), 2))
    except opt_flags.InapplicableConstraint:
        pytest.skip("inapplicable constraint")

    assert_close(a, b)


@pytest.mark.parametrize("m, n, k", [
    (320, 2**19, 0),
    (4096, 4096, 0),
])
@pytest.mark.parametrize("view_x_as_zero_cols", [False, True])
def test_zero_reduction_dim(m, n, k, view_x_as_zero_cols):
    torch.manual_seed(0)

    if view_x_as_zero_cols:
        x = torch.randn(m, m, device="cuda", dtype=torch.bfloat16)
        x = x[:0, :].transpose(-1, -2)
    else:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.float32)

    try:
        tri_y = matmul_ogs(x, w, bias)
    except opt_flags.InapplicableConstraint:
        pytest.skip("inapplicable constraint")
    ref_y = matmul_ogs_torch(x, w, bias, round_x=lambda x, idx: x, round_y=lambda y: y)

    assert_close(ref_y, tri_y)
