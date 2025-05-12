import torch
import triton_bench.distributed as triton_dist

import torch.distributed as dist
import triton_bench
import torch
import torch.multiprocessing as mp
import triton_bench.swiglu
from triton_bench.numerics_details.mxfp import downcast_to_mxfp
from triton_bench.matmul_ogs import MicroscalingCtx, matmul_ogs, PrecisionConfig, FlexCtx
from triton_bench.numerics import InFlexData
import triton_bench.distributed as triton_dist

import pytest


def test_all_gather_non_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    x = torch.randn(4, 5)
    result = triton_dist.all_gather(x, dim=0)
    torch.testing.assert_close(result, x)


def test_all_gather_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def dummy_all_gather_into_tensor(out, x):
        gathered = torch.cat([x, x], dim=0)
        out.copy_(gathered)

    monkeypatch.setattr(dist, "all_gather_into_tensor", dummy_all_gather_into_tensor)

    x = torch.randn(3, 4)
    result = triton_dist.all_gather(x, dim=0)
    expected = torch.cat([x, x], dim=0)
    torch.testing.assert_close(result, expected)


def test_all_gather_distributed_dim1(monkeypatch):
    # WORLD_SIZE=3, gather along dim=1 (columns)
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 3)

    def dummy_all_gather_into_tensor_dim1(out, x):
        # simulate gathering 3 replicas along dim=1
        out.copy_(torch.cat([x, x, x], dim=1))

    monkeypatch.setattr(dist, "all_gather_into_tensor", dummy_all_gather_into_tensor_dim1)

    x = torch.randn(2, 2)
    result = triton_dist.all_gather(x, dim=1)
    expected = torch.cat([x, x, x], dim=1)
    torch.testing.assert_close(result, expected)


def test_reduce_scatter_non_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    x = torch.randn(4, 6)
    result = triton_dist.reduce_scatter(x, token_mask=None, dim=0)
    torch.testing.assert_close(result, x)


def dummy_reduce_scatter(out, x_list):
    out.copy_(x_list[0])


def test_reduce_scatter_distributed_no_token_mask(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "reduce_scatter", dummy_reduce_scatter)

    x = torch.randn(4, 6)
    expected = x.chunk(2, dim=0)[0]

    result = triton_dist.reduce_scatter(x, token_mask=None, dim=0)
    torch.testing.assert_close(result, expected)


def test_reduce_scatter_distributed_with_token_mask(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "reduce_scatter", dummy_reduce_scatter)

    x = torch.randn(2, 4)
    token_mask = torch.tensor([True, False, True, False], dtype=torch.bool)
    shape = list(x.shape)
    # Replace first dimension with token_mask's corresponding dimension.
    shape[0] = token_mask.shape[0]
    x_new = x.new_zeros(shape)
    x_new[token_mask] = x
    # Split along dim=0 (world_size=2) and take the first chunk.
    expected = x_new.chunk(2, dim=0)[0]

    result = triton_dist.reduce_scatter(x, token_mask=token_mask, dim=0)
    torch.testing.assert_close(result, expected)


def test_reduce_scatter_distributed_with_token_mask_dim1(monkeypatch):
    # WORLD_SIZE=2, token_mask on dim=1 (columns)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "reduce_scatter", dummy_reduce_scatter)

    x = torch.randn(3, 2)
    token_mask = torch.tensor([True, False, False, True], dtype=torch.bool)
    shape = [3, 4]
    x_new = x.new_zeros(shape)
    x_new[:, token_mask] = x
    expected = x_new.chunk(2, dim=1)[0]
    result = triton_dist.reduce_scatter(x, token_mask=token_mask, dim=1)
    torch.testing.assert_close(result, expected)


def test_routing_non_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(triton_bench.routing, "routing", lambda logits, n_act, expt_indx=None, EP=1: "dummy_routing")
    result, extra = triton_dist.routing(torch.randn(5, 4), 2)
    assert result == "dummy_routing"
    assert extra is None


def test_routing_distributed_EP(monkeypatch):

    def dummy_all_gather_into_tensor(out, x):
        gathered = torch.cat([x, x], dim=0)
        out.copy_(gathered)

    # Test distributed routing with EP=1 (token_mask should be None)
    monkeypatch.setenv("WORLD_SIZE", "2")
    # Set environment for local rank and distributed group
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "all_gather_into_tensor", dummy_all_gather_into_tensor)

    logits = torch.tensor([[0.1, 0.2, 0.4, 0.3], [0.5, 0.4, 0.3, 0.1]], device="cuda")
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


def quantize(w, dtype, dev, **opt):
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
    if rank % TP == 0:
        gathered = []
        if rank == 0:
            gathered = [torch.zeros_like(param) for _ in range(EP)]
        group = dist.new_group(list(range(0, world_size, TP)))
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
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    dev = f"cuda:{rank}"

    # weights & biases
    wg = torch.randn((dim1, n_expts_tot), device=dev)
    dist.broadcast(wg, src=0)

    if rank == 0:
        bg = torch.randn((n_expts_tot,), device=dev)
    else:
        bg = torch.empty((n_expts_tot,), device=dev)
    dist.broadcast(bg, src=0)

    b2 = torch.randn((n_expts_tot // EP, dim1), device=dev)
    ep_indx = rank // TP
    group = dist.new_group(list(range(ep_indx * TP, (ep_indx + 1) * TP)))
    dist.broadcast(b2, src=ep_indx * TP, group=group)

    w1 = torch.randn((n_expts_tot // EP, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot // EP, dim2 // TP // 2, dim1), device=dev)
    b1 = torch.randn((n_expts_tot // EP, dim2 // TP), device=dev)

    w1_full = gather_full(rank, world_size, w1, TP, EP, concat_dim_inside=2, concat_dim_outside=0)
    w2_full = gather_full(rank, world_size, w2, TP, EP, concat_dim_inside=1, concat_dim_outside=0)
    b1_full = gather_full(rank, world_size, b1, TP, EP, concat_dim_inside=1, concat_dim_outside=0)
    b2_full = gather_ep(rank, world_size, b2, TP, EP)


    # quantization
    swizzle_opt = {"mx4": {"swizzle_mx_scale": True}}
    opt = swizzle_opt.get(w_dtype, {})
    wg, wg_flex, wg_mx = quantize(wg, "bf16", dev)
    w1, w1_flex, w1_mx = quantize(w1, w_dtype, dev, **opt)
    w2, w2_flex, w2_mx = quantize(w2, w_dtype, dev, **opt)
    if rank == 0:
        w1_full, w1_flex_f, w1_mx_f = quantize(w1_full, w_dtype, dev, **opt)
        w2_full, w2_flex_f, w2_mx_f = quantize(w2_full, w_dtype, dev, **opt)
    else:
        w1_full = w2_full = w1_flex_f = w2_flex_f = w1_mx_f = w2_mx_f = None

    # precision configs
    pcg = PrecisionConfig(mx_ctx=wg_mx, flex_ctx=FlexCtx(rhs_data=wg_flex))
    pcs = triton_bench.swiglu.PrecisionConfig(limit=1.0)
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
            rdata, gi, si = triton_bench.routing.routing(logits, n_expts_act)
        else:
            rdata = gi = si = None
        x = matmul_ogs(x, w1_full, b1_full, rdata, gather_indx=gi, precision_config=pc1_f)
        x = triton_bench.swiglu.swiglu(x, 1.0, pcs, routing_data=rdata)
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
        x = matmul_ogs(x, w1, b1, rdata, gather_indx=gi, precision_config=pc1)
        x = triton_bench.swiglu.swiglu(x, 1.0, pcs, routing_data=rdata)
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
