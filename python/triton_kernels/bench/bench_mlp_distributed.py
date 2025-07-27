from pathlib import Path
from copy import deepcopy
import triton.profiler as proton
from triton.profiler import viewer
import torch
import argparse
import triton_kernels
import triton_kernels.swiglu
from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
from triton_kernels.target_info import is_hip, get_cdna_version
from triton_kernels.tensor_details import layout
import warnings

import distributed as triton_dist
from bench_utils import PerfData, quantize_weight, roofline_mlp


def bench_mlp(batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, name):
    assert n_expts_tot % EP == 0
    assert dim2 % TP == 0
    rank, world_size = triton_dist.setup()
    dev = f"cuda:{rank}"
    DP = world_size

    assert n_expts_tot % EP == 0, f"{n_expts_tot=}, {EP=}, n_expts_tot must be divisible by EP"
    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"

    # input
    # weights
    wg = triton_dist.broadcast(torch.randn((dim1, n_expts_tot), device=dev))
    w1 = torch.randn((n_expts_tot // EP, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot // EP, dim2 // TP // 2, dim1), device=dev)

    # biases
    bg = triton_dist.broadcast(torch.randn((n_expts_tot, ), device=dev))
    b1 = torch.randn((n_expts_tot // EP, dim2 // TP), device=dev)
    b2 = torch.randn((n_expts_tot // EP, dim1), device=dev)
    ep_indx = (rank // TP) % EP
    groups = [list(range(ep * TP, (ep + 1) * TP)) for ep in range(EP)]
    b2 = triton_dist.broadcast(b2, src=ep_indx * TP, groups=groups, group_idx=ep_indx)

    # -- numerics --
    opt1 = dict()
    opt2 = dict()
    if w_dtype == "mx4" and not is_hip():
        num_warps = 4 if batch <= 512 else 8
        value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
        scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
            mx_axis=1, num_warps=num_warps)
        opt1 = {
            "value_layout": value_layout,
            "value_layout_opts": value_layout_opts,
            "scale_layout": scale_layout,
            "scale_layout_opts": scale_layout_opts,
        }
        opt2 = deepcopy(opt1)
    wg, wg_flex, wg_scale = quantize_weight(wg, "bf16")
    w1, w1_flex, w1_scale = quantize_weight(w1, w_dtype, **opt1)
    w2, w2_flex, w2_scale = quantize_weight(w2, w_dtype, **opt2)
    pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=wg_flex), weight_scale=wg_scale)
    act = FusedActivation(FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")), (1.0, 1.0), 2)
    pc1 = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w1_flex), weight_scale=w1_scale)
    pc2 = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w2_flex), weight_scale=w2_scale)

    # -- benchmark --
    fpath = Path(f"logs/{name}/{rank}/{x_dtype}-{w_dtype}-TP{TP}-EP{EP}/profiles/batch-{batch}.hatchet")
    fpath.parent.mkdir(parents=True, exist_ok=True)
    x_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[x_dtype]
    # special treatment of fp8_e4m3 on AMD CDNA3 because it uses fp8_e4m3fnuz
    if x_dtype == torch.float8_e4m3fn and get_cdna_version() == 3:
        x_dtype = torch.float8_e4m3fnuz

    input_x = torch.randn((batch // DP, dim1), device=dev)
    # run layer
    proton.start(str(fpath.with_suffix("")), hook="triton")
    input_x = input_x.to(x_dtype)
    xg = input_x.to(wg.dtype if n_expts_tot > 1 else input_x.dtype)
    for i in range(100):
        if n_expts_tot > 1:  # sparse
            logits = matmul_ogs(xg, wg, bg, precision_config=pcg)
            x, rdata, gather_indx, scatter_indx, metadata = triton_dist.routing(input_x, logits, n_expts_act, EP=EP,
                                                                                TP=TP)
        else:  # dense
            x = triton_dist.all_gather(input_x, dim=0)
            rdata, gather_indx, scatter_indx, metadata = None, None, None, None
        if x.nelement() > 0:
            x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1, fused_activation=act)
            x = matmul_ogs(x, w2, b2 if rank % TP == 0 else None, rdata, scatter_indx=scatter_indx,
                           precision_config=pc2)
        x = triton_dist.reduce_scatter(x, metadata=metadata, dim=0)
    proton.finalize()

    # -- analyze --
    gf, _, _, info = viewer.read(fpath)
    # Now the dataframe only contains leave nodes (i.e., kernels) that perform matmuls
    matmuls = gf.filter("MATCH ('*', c) WHERE c.'name' =~ '.*matmul.*' AND c IS LEAF").dataframe
    bytes = matmuls["bytes"].sum()
    flops = sum(matmuls[[c for c in ["flops8", "flops16"] if c in matmuls.columns]].sum())
    time = matmuls["time (ns)"].sum()
    device_type = matmuls["device_type"].iloc[0]
    device_id = matmuls["device_id"].iloc[0]
    device_info = info[device_type][device_id]
    return PerfData(
        time=time,
        flops=flops,
        bytes=bytes,
        bitwidth=x.dtype.itemsize * 8,
        device_type=device_type,
        device_info=device_info,
    )


if __name__ == "__main__":
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_cdna_version() == 4
    batch_ranges_dense = [(1024, 32768, 1024)]
    batch_ranges_moe = [(128, 512, 32), (512, 32000, 128)]
    dense_dtypes = ["fp8", "fp8"]
    quantized_dtypes = ["fp8", "mx4"] if has_native_mx4 else ["bf16", "mx4"]
    rank, world_size = triton_dist.setup()
    if world_size <= 1:
        warnings.warn("Running in non-distributed mode, which may not be optimal for performance measurements.")
    # Running all workloads at once may cause OOM on some GPUs such as H100 80GB.
    # Thus we request users to run each workload separately.
    # For example, all eligible combinations of options are listed below when four GPUs are used:
    # torchrun --nproc-per-node=4 ./bench_mlp_distributed.py --tp 2 --ep 2 --name llama4-maverick
    # torchrun --nproc-per-node=4 ./bench_mlp_distributed.py --tp 1 --ep 4 --name llama4-maverick
    # torchrun --nproc-per-node=4 ./bench_mlp_distributed.py --tp 4 --ep 1 --name llama4-maverick
    # torchrun --nproc-per-node=4 ./bench_mlp_distributed.py --tp 4 --ep 1 --name dense
    # torchrun --nproc-per-node=4 ./bench_mlp_distributed.py --tp 2 --ep 2 --name llama4-maverick --quantized
    # torchrun --nproc-per-node=4 ./bench_mlp_distributed.py --tp 1 --ep 4 --name llama4-maverick --quantized
    # torchrun --nproc-per-node=4 ./bench_mlp_distributed.py --tp 4 --ep 1 --name llama4-maverick --quantized
    # torchrun --nproc-per-node=4 ./bench_mlp_distributed.py --tp 4 --ep 1 --name dense --quantized
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--tp", type=int, default=1)
    argparse.add_argument("--ep", type=int, default=1)
    argparse.add_argument("--name", type=str, choices=["dense", "llama4-maverick"])
    argparse.add_argument("--quantized", type=bool, action="store_true", default=False)
    args = argparse.parse_args()
    dtypes = dense_dtypes if args.quantized else quantized_dtypes
    if args.name == "dense":
        assert args.ep == 1, "EP must be 1 for dense"
        roofline_mlp(batch_ranges_dense, 8192, 8192, 1, 1, *dtypes, bench_mlp, TP=args.tp, EP=args.ep, name="dense",
                     rank=rank)
    else:
        roofline_mlp(
            batch_ranges_moe,
            5120,
            8192,
            128,
            4,
            *dtypes,
            bench_mlp,
            TP=args.tp,
            EP=args.ep,
            name="llama4-maverick",
            rank=rank,
        )
    triton_dist.cleanup()
