from dataclasses import dataclass, fields
import pytest
import torch
from typing import Union
import triton
# routing utilities
from triton_kernels.routing import routing
# matmul utilities
import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig, MicroscalingCtx, FusedActivation, FnSpecs
from triton_kernels.matmul_ogs import can_use_persistent_tma
from triton_kernels.matmul_ogs import matmul_ogs_set_idle_sms, matmul_ogs, matmul_ogs_torch
from triton_kernels.swiglu import swiglu, swiglu_fn, PrecisionConfig as SwiGLUPrecisionConfig
# numerics utilities
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.numerics_details.mxfp import SwizzlingType, downcast_to_mxfp, upcast_from_mxfp
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


def init_routing_data(m, n_expts_tot, n_expts_act, n_expt_shards, do_gather, do_scatter, device="cuda"):
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device, requires_grad=True)
    routing_data, gather_idx, scatter_idx = routing(logits, n_expts_act, simulated_ep=n_expt_shards)
    routing_data.gate_scal = None
    gather_idx = gather_idx if do_gather else None
    scatter_idx = scatter_idx if do_scatter else None
    return m, routing_data, gather_idx, scatter_idx


def init_compute_data(m, n, k, gindx, sindx, n_expts_tot, n_expts_act, n_expt_shards, mode, act_dtype, weight_dtype,
                      has_y_gammas, requires_grad=True, device="cuda"):
    torch.manual_seed(0)
    assert mode in {'batched', 'ragged'}
    in_m = m * (n_expts_act if gindx is None else 1)
    shape_x = (n_expts_tot, in_m, k) if mode == 'batched' else (in_m, k)
    x = alloc_rand(shape_x, device=device, dtype=act_dtype, requires_grad=requires_grad)
    w = alloc_rand((n_expts_tot // n_expt_shards, k, n), device=device, dtype=weight_dtype, requires_grad=requires_grad)
    bias = alloc_rand((n_expts_tot // n_expt_shards, n), device=device, dtype=torch.float32,
                      requires_grad=requires_grad)
    gs0 = 2**torch.randint(-5, 0, (m * n_expts_act, ), device=device, dtype=torch.float32, requires_grad=requires_grad)
    gs1 = 2**torch.randint(-5, 0, (m * n_expts_act, ), device=device, dtype=torch.float32, requires_grad=requires_grad)
    gs0 = gs0.detach().requires_grad_(requires_grad)
    gs1 = gs1.detach().requires_grad_(requires_grad)
    if mode == 'batched' or (not has_y_gammas) or (has_y_gammas and (gindx is not None) and act_dtype.itemsize >= 2):
        gs0 = None
        gs1 = None
    return x, w, bias, gs0, gs1


# ---------------
# numerics stuff
# ---------------


def init_precision(out_dtype, weight_dtype, is_mixed_input, n_expts_tot=1, mx_ctx=MicroscalingCtx(), device="cuda"):
    act_use_flexpoint = out_dtype.itemsize == 1
    weight_use_flexpoint = weight_dtype.itemsize == 1 and not is_mixed_input
    # flexpoint
    make_tensor = lambda val0, val1: torch.tensor([val0, val1] * (n_expts_tot // 2) +
                                                  ([val0]
                                                   if n_expts_tot % 2 else []), dtype=torch.float32, device=device)
    make_scalar = lambda val: torch.tensor([val], dtype=torch.float32, device=device)
    in_flex_data = lambda scale, use_flex: InFlexData(dtype=out_dtype, scale=make_scalar(scale)
                                                      ) if use_flex else InFlexData()
    in_flex_edata = lambda scale0, scale1, use_flex: InFlexData(dtype=weight_dtype, scale=make_tensor(scale0, scale1)
                                                                ) if use_flex else InFlexData()
    out_flex_data = lambda scale, use_flex: OutFlexData(dtype=out_dtype, expected_scale=make_scalar(
        scale), actual_scale=make_scalar(0), checksum_scale=make_scalar(0)) if use_flex else OutFlexData()
    flex_ctx = FlexCtx(
        lhs_data=in_flex_data(1.25, act_use_flexpoint),
        rhs_data=in_flex_edata(1.50, 1.25, weight_use_flexpoint),
        out_data=out_flex_data(4.00, act_use_flexpoint),
    )
    return PrecisionConfig(flex_ctx=flex_ctx, acc_scale=2.0 if act_use_flexpoint or weight_use_flexpoint else 1.0,
                           mx_ctx=mx_ctx, out_dtype=out_dtype)


def apply_precision(x_tri, w_tri, bias_tri, gs0_tri, gs1_tri, precision_config):
    flex_ctx = precision_config.flex_ctx

    def apply(x, scale):
        if scale is None:
            return x.clone().detach().requires_grad_(True)
        elif scale.numel() == 1:
            return (x.float() * scale).detach().requires_grad_(True)
        else:
            assert x.ndim == 3
            assert scale.numel() == x.shape[0]
            return (x.float() * scale[:, None, None]).detach().requires_grad_(True)

    return (
        apply(x_tri, flex_ctx.lhs_data.scale),
        apply(w_tri, flex_ctx.rhs_data.scale),
        apply(bias_tri, None),
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
    n_expt_shards: int = 1
    split_k: int = 1
    hbm_swizzling: bool = False
    epilogue_subtile: Union[int, None] = None


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case)) for case in [
            # Non-mx types:
            Case(16, 256, 256, "ragged", "float16", "float16", 128, 4),
            Case(16, 256, 256, "ragged", "float16", "float16", 128, 4, n_expt_shards=2),
            Case(16, 256, 256, "ragged", "float16", "float16", 128, 4, n_expt_shards=4),
            Case(16, 256, 256, "ragged", "float16", "float16", 4, 1, n_expt_shards=2),
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
            Case(600, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 4, 2, n_expt_shards=2),
            Case(600, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 4, 1, n_expt_shards=2),
            Case(600, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 4, 2, split_k=2),
            Case(1000, 400, 400, "ragged", "float16", "float16", 3, 1),
            Case(1000, 700, 700, "ragged", "float16", "float16", 8, 2),
            Case(1000, 700, 700, "ragged", "float16", "float16", 8, 2, split_k=9),
            # mx types:
            Case(16, 256, 256, "ragged", "bfloat16", "mxfloat4_e2m1", 128, 4),
            Case(16, 256, 256, "ragged", "bfloat16", "mxfloat4_e2m1", 128, 4, hbm_swizzling=True),
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
            Case(1000, 704, 832, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 2, split_k=9),
            Case(1000, 704, 832, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 2, split_k=9, hbm_swizzling=True),
            Case(1000, 704, 832, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 2),
            Case(1000, 704, 832, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 2, hbm_swizzling=True),
            Case(300, 400, 400, "ragged", "float8_e5m2", "mxfloat8_e4m3fn", 8, 4),
            Case(300, 400, 400, "ragged", "float8_e5m2", "mxfloat8_e4m3fn", 8, 4, hbm_swizzling=True),
            Case(300, 400, 832, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 4),
            Case(300, 400, 832, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 4, hbm_swizzling=True),
            Case(300, 400, 400, "batched", "float8_e5m2", "mxfloat8_e4m3fn", 32, 4),
            Case(300, 400, 400, "batched", "float8_e5m2", "mxfloat8_e4m3fn", 32, 4, hbm_swizzling=True),
            Case(256, 256, 256, "ragged", "float8_e5m2", "mxfloat4_e2m1", 128, 4, hbm_swizzling=True),
            Case(256, 256, 256, "ragged", "float8_e5m2", "mxfloat4_e2m1", 128, 4, hbm_swizzling=False),
            # AMD
            Case(300, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz"),
            Case(1000, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz", 3, 1),
            Case(600, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz", 4, 2),
            Case(600, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz", 4, 2, n_expt_shards=2),
            Case(600, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz", 4, 2, split_k=2),
            Case(300, 400, 400, "ragged", "float8_e4m3fn", "float8_e4m3fn"),
            Case(1000, 400, 400, "ragged", "float8_e4m3fn", "float8_e4m3fn", 3, 1),
            Case(600, 400, 400, "ragged", "float8_e4m3fn", "float8_e4m3fn", 4, 2),
            Case(600, 400, 400, "ragged", "float8_e4m3fn", "float8_e4m3fn", 4, 2, n_expt_shards=2),
        ]
    ],
)
@pytest.mark.parametrize("block_m", [16, 128])
@pytest.mark.parametrize("do_gather, do_scatter, fused_scatter", [
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (True, True, False),
    (True, True, True),
])
@pytest.mark.parametrize("has_y_gammas", [False, True])
@pytest.mark.parametrize("is_persistent", [False, True])
def test_op(m, n, k, split_k, do_gather, do_scatter, fused_scatter, has_y_gammas, is_persistent, n_expts_tot,
            n_expts_act, n_expt_shards, mode, act_dtype_str, weight_dtype_str, block_m, hbm_swizzling, epilogue_subtile,
            device, opt_flags_scope):
    # TODO: remove when Triton FP8 supports proper RTNE
    if is_cuda():
        if "float8" in weight_dtype_str and torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("Float8 not tested on A100")
        if "float16" in act_dtype_str and "mx" in weight_dtype_str and torch.cuda.get_device_capability()[0] >= 10:
            pytest.skip("float16 x mx not supported with cuda capability >= 10")
        if "float8" in act_dtype_str and "mx" in weight_dtype_str and torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("float8 x mx not supported with cuda capability < 10")
        if n == 2880 and k == 2880 and torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("Not enough memory on A100")

    elif is_hip():
        if "float8" in act_dtype_str and "mx" in weight_dtype_str and not is_hip_cdna4():
            pytest.skip("float8 x mx only supported on CDNA4")
        if "float8" in act_dtype_str and "mxfloat8" in weight_dtype_str:
            pytest.skip("NYI: float8 x mxfloat8 not tested on AMD GPU")
        if is_persistent:
            pytest.skip("NYI: Persistent kernel not supported on AMD GPU")
        if split_k > 1:
            pytest.skip("splitK hasn't been fully tested on AMD GPU.")

    if "float8_e4m3fnuz" in (weight_dtype_str, act_dtype_str) and not is_hip_cdna3():
        pytest.skip("float8_e4m3fnuz only tested on AMD CDNA3 Platform")

    if fused_scatter and split_k > 1:
        pytest.skip("fused scatter scratchpad not supported with split_k")
    if hbm_swizzling:
        if is_hip():
            pytest.skip("NYI. HBM swizzling just implemented for CUDA.")
        if torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("NYI. Ampere swizzling.")
        if torch.cuda.get_device_capability()[0] < 10:
            if "mxfloat4" not in weight_dtype_str:
                pytest.skip("NYI. Hopper swizzling just implemented for mxfp4.")
            if k % 64 != 0 or n % 64 != 0:
                # Automatic padding not implemented for Hopper swizzle
                pytest.skip("Hopper swizzling acts on a 64x64 tile (4x1 mma tiles).")

    # launch metadata for batched / mx types may not work yet.
    test_launch_metadata = (mode == "ragged") and ("mx" not in weight_dtype_str)

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
    opt_flags.update_opt_flags_constraints(constraints)

    is_mixed_input = act_dtype_str != weight_dtype_str
    if weight_dtype_str.startswith("mx"):
        weight_dtype_str = weight_dtype_str[2:]

    test_bwd = False
    weight_dtype = dtype_str_to_torch(weight_dtype_str)
    act_dtype = dtype_str_to_torch(act_dtype_str)
    act_is_float8 = act_dtype.itemsize == 1
    precision_opt = init_precision(act_dtype, weight_dtype, is_mixed_input, n_expts_tot // n_expt_shards, device=device)
    # precision_opt.x_pad_trans_requires_flexpoint = False
    if mode == "ragged":
        m, rdata, gindx, sindx = init_routing_data(m, n_expts_tot, n_expts_act, n_expt_shards, do_gather, do_scatter,
                                                   device=device)
    else:
        rdata = gindx = sindx = None
    x_tri, w_tri, bias_tri, gs0_tri, gs1_tri = init_compute_data(m, n, k, gindx, sindx, n_expts_tot, n_expts_act,
                                                                 n_expt_shards, mode, act_dtype,  #
                                                                 torch.bfloat16 if is_mixed_input else weight_dtype,
                                                                 has_y_gammas, requires_grad=test_bwd, device=device)
    x_ref, w_ref, bias_ref, gs0_ref, gs1_ref = apply_precision(x_tri, w_tri, bias_tri, gs0_tri, gs1_tri, precision_opt)

    if is_mixed_input:
        if hbm_swizzling:
            swizzle_axis = 2
            if torch.cuda.get_device_capability()[0] < 10:
                swizzle_value = SwizzlingType.HOPPER
                swizzle_scale = SwizzlingType.HOPPER
            else:
                swizzle_value = None
                swizzle_scale = SwizzlingType.BLACKWELL
        else:
            swizzle_axis = None
            swizzle_value = None
            swizzle_scale = None
        w_tri, mx_scales_tri, weight_scale_shape = downcast_to_mxfp(w_tri, weight_dtype, axis=1,
                                                                    swizzle_axis=swizzle_axis,
                                                                    swizzle_value=swizzle_value,
                                                                    swizzle_scale=swizzle_scale)
        w_ref = upcast_from_mxfp(w_tri, mx_scales_tri, torch.bfloat16, axis=1, swizzle_axis=swizzle_axis,
                                 swizzle_value=swizzle_value, swizzle_scale=swizzle_scale)

        precision_opt.mx_ctx = MicroscalingCtx(weight_scale=mx_scales_tri, swizzle_value=swizzle_value,
                                               swizzle_scale=swizzle_scale,
                                               actual_weight_scale_shape=weight_scale_shape)

    if is_persistent and not can_use_persistent_tma(x_tri, w_tri, gindx, precision_opt):
        pytest.skip("persistent TMAs not supported for this test")

    if w_tri.shape[0] == 1:
        # Test the case when weight has dim 2, i.e., shape (K, N).
        w_tri = w_tri.squeeze(0).detach().requires_grad_(test_bwd)
        w_ref = w_ref.squeeze(0).detach().requires_grad_(test_bwd)

    if test_launch_metadata:

        def _clobber(t, used_mask):
            # Fill the unread part of the tensor with garbage, to be sure that
            # we don't actually read from the part.
            if len(used_mask) == 1:
                return
            elif t.element_size() == 1:
                t.view(torch.int8)[~used_mask] = 127
            else:
                t[~used_mask] = torch.inf

        if rdata is not None:
            n_tokens = rdata.expt_hist.sum().item()
            used_expts = (rdata.expt_hist > 0)
            _clobber(w_tri, used_expts)
            n_w_bytes = used_expts.sum().item() * n * k * w_tri.element_size()
        else:
            n_tokens = m
            n_w_bytes = w_tri.numel() * w_tri.element_size()

        if gindx is not None:
            used_x_rows = (gindx.dst_indx.view(-1, n_expts_act) != -1).any(dim=1)
            _clobber(x_tri, used_x_rows)
            n_x_bytes = used_x_rows.sum().item() * k * x_tri.element_size()
        elif rdata is not None:
            n_x_bytes = n_tokens * k * x_tri.element_size()
        else:
            n_x_bytes = x_tri.numel() * x_tri.element_size()

        nbytes = None

        def _hook(launch_metadata):
            nonlocal nbytes
            metadata = launch_metadata.get()
            if "matmul_ogs" in metadata["name"]:
                nbytes = metadata["bytes"]

        triton.knobs.runtime.launch_enter_hook = _hook

    if mode == "batched":
        rdata, gindx, sindx = None, None, None
    flex = precision_opt.flex_ctx

    # triton
    tri_y = matmul_ogs(x_tri, w_tri, bias_tri, rdata, gindx, sindx, precision_opt, gammas=gs1_ref)
    # If split_k > 1, then the intermediate tensor is fp32.
    sep_gather = mode == "ragged" and do_gather and n_expts_act > 1 and split_k == 1
    sep_scatter = mode == "ragged" and do_scatter and n_expts_act > 1 and split_k == 1
    y_scale = flex.out_data.expected_scale if act_is_float8 else 1

    if test_launch_metadata:
        if gindx is not None:
            n_y_bytes = (gindx.src_indx != -1).sum().item() * n * tri_y.element_size()
        elif rdata is not None:
            n_y_bytes = n_tokens * n * tri_y.element_size()
        else:
            n_y_bytes = tri_y.numel() * tri_y.element_size()
        assert nbytes == n_x_bytes + n_y_bytes + n_w_bytes
        triton.knobs.runtime.launch_enter_hook = None

    def round_x(x, idx):
        return x.to(act_dtype).to(torch.float32) if sep_gather else x

    round_y = lambda y: (y / y_scale).to(act_dtype).to(torch.float32) * y_scale if sep_scatter else y
    ref_y = matmul_ogs_torch(x_ref, w_ref, bias_ref,  #
                             rdata, gindx, sindx, round_x=round_x, round_y=round_y, gammas=gs1_ref)
    scale = lambda val, scal: val if scal is None else val / scal
    if n_expt_shards > 1:
        if do_scatter:
            indx = sindx.dst_indx[sindx.dst_indx != -1]
            ref_y = ref_y[indx // n_expts_act, :]
            if act_is_float8:
                tri_y = tri_y.view(torch.int8)
            tri_y = tri_y[indx // n_expts_act, :]
            if act_is_float8:
                tri_y = tri_y.view(act_dtype)
        else:
            n_rows = rdata.expt_hist.sum()
            assert n_rows > 0
            ref_y = ref_y[:n_rows]
            tri_y = tri_y[:n_rows]
    assert_close(scale(ref_y, flex.out_data.expected_scale), tri_y)

    if act_is_float8:
        tri_y_scale = flex.out_data.actual_scale.clone()
        ref_y_scale = compute_actual_scale(ref_y, tri_y.dtype)
        assert (ref_y_scale -
                tri_y_scale).abs() < 1e-10, f"ref_y_scale: {ref_y_scale}, tri_y_scale: {tri_y_scale.item()}"


def test_set_idle_sms():
    if not is_cuda():
        pytest.skip("Only supported on CUDA")
    from triton_kernels.matmul_ogs_details.opt_flags import make_opt_flags
    num_idle_sms = 24
    matmul_ogs_set_idle_sms(num_idle_sms)
    flags = make_opt_flags(torch.float32, torch.float32, torch.float32, PrecisionConfig(), \
                           1024, 1024, 1024, None, True, False, 1)
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
        "fused_scatter": fused_scatter,
        "split_k": split_k,
    }
    n_expts_tot, n_expts_act, n_expt_shards = 1, 1, 1
    opt_flags.update_opt_flags_constraints(constraints)

    weight_dtype, act_dtype = torch.float16, torch.float16
    if mode == "ragged":
        m, rdata, gindx, sindx = init_routing_data(m, n_expts_tot, n_expts_act, n_expt_shards, do_gather, do_scatter,
                                                   device=device)
    else:
        rdata = gindx = sindx = None

    precision_opt = init_precision(act_dtype, weight_dtype, False, n_expts_tot // n_expt_shards, device=device)
    x, w, bias, _, _ = init_compute_data(m, n, k, gindx, sindx, n_expts_tot, n_expts_act, n_expt_shards, mode,
                                         act_dtype, weight_dtype, False, requires_grad=False, device=device)

    if is_persistent and not can_use_persistent_tma(x.view(1, x.shape[-2], x.shape[-1]),
                                                    w.view(1, w.shape[-2], w.shape[-1]), gindx, precision_opt):
        pytest.skip("persistent TMAs not supported for this test")

    if mode == "batched":
        rdata, gindx, sindx = None, None, None
    a = swiglu(matmul_ogs(x, w, bias, rdata, gindx, sindx, precision_opt), swiglu_alpha,
               precision_config=SwiGLUPrecisionConfig(swiglu_limit))
    b = matmul_ogs(
        x, w, bias, rdata, gindx, sindx, precision_opt,
        fused_activation=FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), (swiglu_alpha, swiglu_limit),
                                         2))
    assert_close(a, b)
