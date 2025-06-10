import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.matmul_ogs import MicroscalingCtx, matmul_ogs, PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
from triton_kernels.numerics import InFlexData
import triton_kernels.distributed as triton_dist
import triton_kernels.swiglu

import pytest


def dummy_all_gather(out, x):
    out[0].copy_(x)
    out[1].copy_(x)


def test_all_gather_non_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    x = torch.randn(4, 5)
    result = triton_dist.all_gather(x, dim=0)
    torch.testing.assert_close(result, x)


@pytest.mark.parametrize("dim", [0, 1])
def test_all_gather_distributed(monkeypatch, dim):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_gather", dummy_all_gather)

    x = torch.randn(4, 4)
    result = triton_dist.all_gather(x, dim=dim)
    expected = torch.cat([x, x], dim=dim)
    torch.testing.assert_close(result, expected)


def test_reduce_scatter_non_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    x = torch.randn(4, 6)
    result = triton_dist.reduce_scatter(x, token_mask=None, dim=0)
    torch.testing.assert_close(result, x)


def test_reduce_scatter_distributed_no_token_mask(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "reduce_scatter", lambda out, x_list: out.copy_(x_list[0]))

    x = torch.randn(4, 6)
    expected = x.chunk(2, dim=0)[0]

    result = triton_dist.reduce_scatter(x, token_mask=None, dim=0)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "dim, x_shape, token_mask",
    [
        (0, (2, 4), torch.tensor([True, False, True, False], dtype=torch.bool)),
        (1, (3, 2), torch.tensor([True, False, False, True], dtype=torch.bool)),
    ],
)
def test_reduce_scatter_distributed_with_token_mask(monkeypatch, dim, x_shape, token_mask):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "reduce_scatter", lambda out, x_list: out.copy_(x_list[0]))

    x = torch.randn(x_shape)
    shape = list(x.shape)
    shape[dim] = token_mask.shape[0]
    x_new = x.new_zeros(shape)
    if dim == 0:
        x_new[token_mask] = x
    else:
        x_new[:, token_mask] = x
    expected = x_new.chunk(2, dim=dim)[0]

    result = triton_dist.reduce_scatter(x, token_mask=token_mask, dim=dim)
    torch.testing.assert_close(result, expected)


def test_routing_distributed_EP(monkeypatch):
    # Test distributed routing with EP=1 (token_mask should be None)
    monkeypatch.setenv("WORLD_SIZE", "2")
    # Set environment for local rank and distributed group
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "all_gather", dummy_all_gather)

    logits = torch.tensor([[0.1, 0.2, 0.4, 0.3], [0.5, 0.4, 0.3, 0.1]], device="cuda", dtype=torch.float16)
    n_expts_act = 2
    EP = 2
    expt_indx = torch.tensor([[0, 1], [0, 1]], device="cuda").reshape(-1)
    topk_indx = torch.argsort(expt_indx, stable=True)
    gate_indx = torch.argsort(topk_indx, stable=True)
    rdata, gather_indx, scatter_indx, token_mask = triton_dist.routing(logits, n_expts_act, EP=EP)
    assert torch.equal(gather_indx.src_indx, topk_indx.int())
    assert torch.equal(gather_indx.dst_indx, gate_indx.int())
    assert torch.equal(scatter_indx.src_indx, gate_indx.int())
    assert torch.equal(scatter_indx.dst_indx, topk_indx.int())
    assert torch.equal(token_mask, torch.tensor([False, True, False, True], dtype=torch.bool, device="cuda"))


def quantize(w, dtype, **opt):
    if dtype == "bf16":
        wq = w.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(), MicroscalingCtx()
    else:
        assert dtype == "mx4", f"{dtype=}"
        swizzle_mx_scale = opt["swizzle_mx_scale"]
        swizzle_axis = 2 if swizzle_mx_scale else None
        w = w.to(torch.bfloat16)
        w, mx_scales, weight_scale_shape = downcast_to_mxfp(w, torch.uint8, axis=1, swizzle_axis=swizzle_axis)
        return (
            w,
            InFlexData(),
            MicroscalingCtx(weight_scale=mx_scales, swizzle_mx=swizzle_mx_scale,
                            actual_weight_scale_shape=weight_scale_shape),
        )


def gather_ep(rank, world_size, param, TP, EP):
    gathered = None
    group = dist.new_group(list(range(0, world_size, TP)))
    dist.barrier()
    if rank % TP == 0:
        gathered = []
        if rank == 0:
            gathered = [torch.zeros_like(param) for _ in range(EP)]
        dist.gather(param, gathered, dst=0, group=group)
        if rank == 0:
            gathered = torch.cat(gathered, dim=0)
    return gathered


def gather_full(rank, world_size, param, TP, EP, concat_dim_inside, concat_dim_outside):
    gathered = []
    if rank == 0:
        gathered = [torch.zeros_like(param) for _ in range(world_size)]
    dist.gather(param, gathered, dst=0)
    if rank == 0:
        rows = [torch.cat(gathered[e * TP:(e + 1) * TP], dim=concat_dim_inside) for e in range(EP)]
        return torch.cat(rows, dim=concat_dim_outside)
    return None


def distributed_run(rank, world_size, batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP):
    # We compare the distributed and single-GPU versions of the model to verify correctness.

    # init
    dev = f"cuda:{rank}"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=torch.device(dev))

    # weights & biases
    wg = torch.randn((dim1, n_expts_tot), device=dev)
    dist.broadcast(wg, src=0)

    bg = torch.randn((n_expts_tot, ), device=dev)
    dist.broadcast(bg, src=0)

    b2 = torch.randn((n_expts_tot // EP, dim1), device=dev)
    ep_indx = rank // TP
    groups = [dist.new_group(list(range(ep * TP, (ep + 1) * TP))) for ep in range(EP)]
    dist.barrier()
    group = groups[ep_indx]
    dist.broadcast(b2, src=ep_indx * TP, group=group)

    w1 = torch.randn((n_expts_tot // EP, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot // EP, dim2 // TP // 2, dim1), device=dev)
    b1 = torch.randn((n_expts_tot // EP, dim2 // TP), device=dev)

    w1_full = gather_full(rank, world_size, w1, TP, EP, concat_dim_inside=2, concat_dim_outside=0)
    w2_full = gather_full(rank, world_size, w2, TP, EP, concat_dim_inside=1, concat_dim_outside=0)
    b1_full = gather_full(rank, world_size, b1, TP, EP, concat_dim_inside=1, concat_dim_outside=0)
    b2_full = gather_ep(rank, world_size, b2, TP, EP)

    # quantization
    wg, wg_flex, wg_mx = quantize(wg, "bf16")
    w1, w1_flex, w1_mx = quantize(w1, w_dtype)
    w2, w2_flex, w2_mx = quantize(w2, w_dtype)
    if rank == 0:
        w1_full, w1_flex_f, w1_mx_f = quantize(w1_full, w_dtype)
        w2_full, w2_flex_f, w2_mx_f = quantize(w2_full, w_dtype)
    else:
        w1_full = w2_full = w1_flex_f = w2_flex_f = w1_mx_f = w2_mx_f = None

    # precision configs
    pcg = PrecisionConfig(mx_ctx=wg_mx, flex_ctx=FlexCtx(rhs_data=wg_flex))
    act = FusedActivation(FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")), (1.0, 1.0), 2)
    pc1 = PrecisionConfig(mx_ctx=w1_mx, flex_ctx=FlexCtx(rhs_data=w1_flex))
    pc2 = PrecisionConfig(mx_ctx=w2_mx, flex_ctx=FlexCtx(rhs_data=w2_flex))
    if rank == 0:
        pc1_f = PrecisionConfig(mx_ctx=w1_mx_f, flex_ctx=FlexCtx(rhs_data=w1_flex_f))
        pc2_f = PrecisionConfig(mx_ctx=w2_mx_f, flex_ctx=FlexCtx(rhs_data=w2_flex_f))
    else:
        pc1_f = pc2_f = None

    # inputs
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
    xd = torch.randn((batch // world_size, dim1), device=dev).to(dtype_map[x_dtype])
    x0 = triton_dist.all_gather(xd, dim=0)

    # single-GPU pass
    def single(x):
        xg = x.to(wg.dtype if n_expts_tot > 1 else x.dtype)
        if n_expts_tot > 1:
            logits = matmul_ogs(xg, wg, bg, precision_config=pcg)
            rdata, gi, si = triton_kernels.routing.routing(logits, n_expts_act)
        else:
            rdata = gi = si = None
        x = matmul_ogs(x, w1_full, b1_full, rdata, gather_indx=gi, precision_config=pc1_f, fused_activation=act)
        return matmul_ogs(x, w2_full, b2_full, rdata, scatter_indx=si, precision_config=pc2_f)

    # distributed pass
    def distributed(x):
        xg = x.to(wg.dtype if n_expts_tot > 1 else x.dtype)
        x = triton_dist.all_gather(x, dim=0)
        if n_expts_tot > 1:
            logits = matmul_ogs(xg, wg, bg, precision_config=pcg)
            rdata, gi, si, tm = triton_dist.routing(logits, n_expts_act, EP=EP, TP=TP)
        else:
            rdata = gi = si = tm = None
        if tm is not None:
            x = x[tm]
        x = matmul_ogs(x, w1, b1, rdata, gather_indx=gi, precision_config=pc1, fused_activation=act)
        x = matmul_ogs(x, w2, b2 if rank % TP == 0 else None, rdata, scatter_indx=si, precision_config=pc2)
        x = triton_dist.reduce_scatter(x, token_mask=tm, dim=0)
        # gather the result from all GPUs, just for verification
        return triton_dist.all_gather(x, dim=0)

    distributed_result = distributed(xd)
    if rank == 0:
        single_result = single(x0)
        torch.testing.assert_close(distributed_result, single_result, rtol=1e-2, atol=1.0)

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP",
    [
        (1024, 512, 512, 128, 2, "bf16", "bf16", 1, 1),
        (1024, 512, 512, 128, 2, "bf16", "bf16", 4, 1),
        (1024, 512, 512, 128, 2, "bf16", "bf16", 1, 4),
        (1024, 512, 512, 128, 2, "bf16", "bf16", 2, 2),
    ],
)
def test_mlp_mp(batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, monkeypatch):
    parallelism = TP * EP
    if torch.cuda.device_count() < parallelism:
        pytest.skip(f"Test requires at least {parallelism} GPUs.")

    monkeypatch.setenv("WORLD_SIZE", f"{parallelism}")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "12355")
    mp.spawn(
        distributed_run,
        args=(parallelism, batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP),
        nprocs=parallelism,
        join=True,
    )
