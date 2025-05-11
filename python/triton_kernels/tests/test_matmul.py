from dataclasses import dataclass, fields
import pytest
import torch
from typing import Union
# benchmarking utilities
# routing utilities
from triton_kernels.routing import routing
# matmul utilities
import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig, MicroscalingCtx
from triton_kernels.matmul_ogs import can_use_persistent_tma
from triton_kernels.matmul_ogs import matmul_ogs, matmul_ogs_torch
# numerics utilities
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp, upcast_from_mxfp
# testing utilities
from triton_kernels.testing import assert_close, compute_actual_scale
# target-specific utilities
from triton_kernels.target_info import is_hip

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


def init_precision(out_dtype, act_use_flexpoint, weight_use_flexpoint, n_expts_tot=1, mx_ctx=MicroscalingCtx(),
                   device="cuda"):
    # flexpoint
    make_tensor = lambda val0, val1: torch.tensor([val0, val1] * (n_expts_tot // 2) +
                                                  ([val0]
                                                   if n_expts_tot % 2 else []), dtype=torch.float32, device=device)
    make_scalar = lambda val: torch.tensor([val], dtype=torch.float32, device=device)
    in_flex_data = lambda scale, use_flex: InFlexData(dtype=torch.float8_e5m2, scale=make_scalar(scale)
                                                      ) if use_flex else InFlexData()
    in_flex_edata = lambda scale0, scale1, use_flex: InFlexData(dtype=torch.float8_e5m2, scale=make_tensor(
        scale0, scale1)) if use_flex else InFlexData()
    out_flex_data = lambda scale, use_flex: OutFlexData(dtype=torch.float8_e5m2, expected_scale=make_scalar(
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
    swizzle_mx_scale: bool = False
    epilogue_subtile: Union[bool, None] = None


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
            Case(600, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 4, 2, epilogue_subtile=False),
            Case(600, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 4, 2, epilogue_subtile=True),
            Case(600, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 4, 2),
            Case(600, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 4, 2, n_expt_shards=2),
            Case(600, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 4, 1, n_expt_shards=2),
            Case(600, 400, 400, "ragged", "float8_e5m2", "float8_e5m2", 4, 2, split_k=2),
            Case(1000, 400, 400, "ragged", "float16", "float16", 3, 1),
            Case(1000, 700, 700, "ragged", "float16", "float16", 8, 2),
            Case(1000, 700, 700, "ragged", "float16", "float16", 8, 2, split_k=9),
            # mx types:
            Case(16, 256, 256, "ragged", "bfloat16", "mxfloat4_e2m1", 128, 4),
            Case(1000, 700, 700, "batched", "bfloat16", "mxfloat4_e2m1", 8, 2),
            Case(1000, 700, 700, "ragged", "bfloat16", "mxfloat4_e2m1", 8, 2, split_k=9),
            Case(300, 400, 400, "ragged", "bfloat16", "mxfloat8_e4m3fn", 8, 4),
            Case(300, 400, 400, "ragged", "bfloat16", "mxfloat8_e4m3fn", 8, 4, swizzle_mx_scale=True),
            Case(300, 400, 400, "batched", "bfloat16", "mxfloat8_e5m2", 32, 4),
            Case(1000, 700, 2, "batched", "bfloat16", "mxfloat4_e2m1", 8, 2),
            Case(16, 256, 256, "ragged", "float8_e5m2", "mxfloat4_e2m1", 128, 4, swizzle_mx_scale=True),
            Case(1000, 704, 800, "batched", "float8_e5m2", "mxfloat4_e2m1", 3, 1, swizzle_mx_scale=True),
            Case(1000, 704, 800, "batched", "float8_e5m2", "mxfloat4_e2m1", 3, 1, swizzle_mx_scale=False),
            Case(1000, 704, 800, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 2, split_k=9, swizzle_mx_scale=False),
            Case(1000, 704, 800, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 2, split_k=9, swizzle_mx_scale=True),
            Case(1000, 704, 800, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 2, swizzle_mx_scale=False),
            Case(1000, 704, 800, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 2, swizzle_mx_scale=True),
            Case(300, 400, 400, "ragged", "float8_e5m2", "mxfloat8_e4m3fn", 8, 4, swizzle_mx_scale=False),
            Case(300, 400, 400, "ragged", "float8_e5m2", "mxfloat8_e4m3fn", 8, 4, swizzle_mx_scale=True),
            Case(300, 400, 800, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 4, swizzle_mx_scale=False),
            Case(300, 400, 800, "ragged", "float8_e5m2", "mxfloat4_e2m1", 8, 4, swizzle_mx_scale=True),
            Case(300, 400, 400, "batched", "float8_e5m2", "mxfloat8_e4m3fn", 32, 4, swizzle_mx_scale=False),
            Case(300, 400, 400, "batched", "float8_e5m2", "mxfloat8_e4m3fn", 32, 4, swizzle_mx_scale=True),
            # AMD
            Case(300, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz"),
            Case(1000, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz", 3, 1),
            Case(600, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz", 4, 2),
            Case(600, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz", 4, 2, n_expt_shards=2),
            Case(600, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz", 4, 2),
            Case(600, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz", 4, 2, split_k=2),
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
            n_expts_act, n_expt_shards, mode, act_dtype_str, weight_dtype_str, block_m, swizzle_mx_scale,
            epilogue_subtile, device):
    # TODO: remove when Triton FP8 supports proper RTNE
    if "float8" in weight_dtype_str and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Float8 not tested on A100")
    if "float8_e4m3fnuz" in weight_dtype_str and not is_hip():
        pytest.skip("float8_e4m3fnuz only tested on HIP platforms")
    if "mx" in weight_dtype_str and is_hip():
        pytest.skip("mxfloat* only tested on CUDA platforms")
    if "float16" in act_dtype_str and "mx" in weight_dtype_str and torch.cuda.get_device_capability()[0] >= 10:
        pytest.skip("float16 x mx not supported with cuda capability >= 10")
    if "float8" in act_dtype_str and "mx" in weight_dtype_str and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("float8 x mx not supported with cuda capability < 10")
    if fused_scatter and split_k > 1:
        pytest.skip("fused scatter scratchpad not supported with split_k")

    torch.manual_seed(0)

    block_k = None
    if is_persistent and weight_dtype_str.startswith("mx") and not torch.cuda.get_device_capability()[0] >= 10:
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
    weight_is_float8 = weight_dtype.itemsize == 1
    precision_opt = init_precision(act_dtype, act_is_float8, weight_is_float8 and not is_mixed_input,
                                   n_expts_tot // n_expt_shards, device=device)
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
        swizzle_axis = 2 if swizzle_mx_scale else None
        w_tri, mx_scales_tri, weight_scale_shape = downcast_to_mxfp(w_tri, weight_dtype, axis=1,
                                                                    swizzle_axis=swizzle_axis)
        w_ref = upcast_from_mxfp(w_tri, mx_scales_tri, torch.bfloat16, axis=1, swizzle_axis=swizzle_axis)

        precision_opt.mx_ctx = MicroscalingCtx(weight_scale=mx_scales_tri, swizzle_mx=swizzle_mx_scale,
                                               actual_weight_scale_shape=weight_scale_shape)

    if is_persistent and not can_use_persistent_tma(x_tri, w_tri, gindx, precision_opt):
        pytest.skip("persistent TMAs not supported for this test")

    if w_tri.shape[0] == 1:
        # Test the case when weight has dim 2, i.e., shape (K, N).
        w_tri = w_tri.squeeze(0).detach().requires_grad_()
        w_ref = w_ref.squeeze(0).detach().requires_grad_()

    if mode == "batched":
        rdata, gindx, sindx = None, None, None
    flex = precision_opt.flex_ctx
    # triton
    tri_y = matmul_ogs(x_tri, w_tri, bias_tri, rdata, gindx, sindx, precision_opt, gammas=gs1_ref)
    # If split_k > 1, then the intermediate tensor is fp32.
    sep_gather = mode == "ragged" and do_gather and n_expts_act > 1 and split_k == 1
    sep_scatter = mode == "ragged" and do_scatter and n_expts_act > 1 and split_k == 1
    y_scale = flex.out_data.expected_scale if act_is_float8 else 1

    def round_x(x, idx):
        return x.to(act_dtype).to(torch.float32) if sep_gather else x

    round_y = lambda y: (y / y_scale).to(act_dtype).to(torch.float32) * y_scale if sep_scatter else y
    ref_y = matmul_ogs_torch(x_ref, w_ref, bias_ref,  #
                             rdata, gindx, sindx, round_x=round_x, round_y=round_y, gammas=gs1_ref)
    scale = lambda val, scal: val if scal is None else val / scal
    if n_expt_shards > 1:
        if not do_scatter:
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
