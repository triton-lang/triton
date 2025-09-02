from itertools import chain
from pathlib import Path
from copy import deepcopy
import triton.profiler as proton
import torch
import argparse
import triton_kernels
import triton_kernels.swiglu
from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
from triton_kernels.target_info import get_cdna_version
import distributed as triton_dist
from triton_kernels.tensor_details import layout
from bench_utils import quantize_weight
import tempfile
import roofline


def bench_mlp(batch_per_expt, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP):
    assert n_expts_tot % EP == 0
    assert dim2 % TP == 0
    rank, world_size = triton_dist.setup()
    dev = f"cuda:{rank}"
    DP = world_size
    batch = batch_per_expt * n_expts_tot // n_expts_act

    assert n_expts_tot % EP == 0, f"{n_expts_tot=}, {EP=}, n_expts_tot must be divisible by EP"
    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"

    # -- init data --
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
    if w_dtype == "mx4":
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
    x_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[x_dtype]
    # special treatment of fp8_e4m3 on AMD CDNA3 because it uses fp8_e4m3fnuz
    if x_dtype == torch.float8_e4m3fn and get_cdna_version() == 3:
        x_dtype = torch.float8_e4m3fnuz

    input_x = torch.randn((batch // DP, dim1), device=dev)
    # run layer
    fpath = Path(tempfile.mktemp())
    proton.start(str(fpath), hook="triton")
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
    return roofline.parse_profile(fpath.with_suffix(".hatchet"), useful_op_regex=".*matmul.*")


def roofline_mlp(batch_sizes, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, \
                  name="", verbose=True):
    out_path = Path(f"logs/{name}/{x_dtype}x-{w_dtype}w-TP{TP}-EP{EP}/")
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = roofline.compute_roofline(dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP,  # fixed args
                                         bench_fn=bench_mlp,  # function to benchmark
                                         intensity_proxy_name="batch_per_expt",  # intensity proxy name
                                         intensity_proxy_values=batch_sizes,  # intensity proxy values to sweep
                                         verbose=verbose,  # options
                                         out_path=out_path.with_suffix(".csv"))  # output path
    png_path = roofline.plot_roofline(series=[csv_path],  # roofline data to plot
                                      flops_dtype=x_dtype,  # dtype to use for FLOPS roof
                                      xlabel="batch_per_expt", title=out_path,  # plot option
                                      out_path=out_path.with_suffix(".png"),  # output path
                                      max_tbps="memset", max_tflops="cublas")  # hardware limits

    return png_path


if __name__ == "__main__":
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_cdna_version() == 4
    batch_sizes_dense = [*range(128, 8192, 128)]
    batch_ranges_moe = [(2**(2 + k), 2**(3 + k), min(2**k, 32)) for k in range(8)]
    batch_sizes_moe = list(chain(*[range(*r) for r in batch_ranges_moe]))
    dense_dtypes = ["fp8", "fp8"]
    quantized_dtypes = ["fp8", "mx4"] if has_native_mx4 else ["bf16", "mx4"]
    rank, world_size = triton_dist.setup()
    if world_size > 1:
        # Running all workloads at once may cause OOM on some GPUs such as H100 80GB.
        # Thus we request users to run each workload separately.
        # For example, all eligible combinations of options are listed below when four GPUs are used:
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 2 --ep 2 --name gpt-oss-x2
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 1 --ep 4 --name gpt-oss-x2
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 4 --ep 1 --name gpt-oss-x2
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 4 --ep 1 --name dense
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 2 --ep 2 --name gpt-oss-x2 --quantized
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 1 --ep 4 --name gpt-oss-x2 --quantized
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 4 --ep 1 --name gpt-oss-x2 --quantized
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 4 --ep 1 --name dense --quantized
        parser = argparse.ArgumentParser()
        parser.add_argument("--tp", type=int, default=1)
        parser.add_argument("--ep", type=int, default=1)
        parser.add_argument("--name", type=str, choices=["dense", "gpt-oss-x2"])
        parser.add_argument("--quantized", action="store_true", default=False)
        args = parser.parse_args()
        dtypes = quantized_dtypes if args.quantized else dense_dtypes
        if args.name == "dense":
            assert args.ep == 1, "EP must be 1 for dense"
            roofline_mlp(batch_sizes_dense, 8192, 8192, 1, 1, dtypes[0], dtypes[1], TP=args.tp, EP=args.ep,
                         name="dense")
        else:
            roofline_mlp(batch_sizes_moe, 5760, 5760, 128, 4, dtypes[0], dtypes[1], TP=args.tp, EP=args.ep,
                         name="gpt-oss-x2")
        triton_dist.cleanup()
    else:
        roofline_mlp(batch_sizes_dense, 8192, 8192, 1, 1, quantized_dtypes[0], quantized_dtypes[1], TP=1, EP=1,
                     name="dense")
        roofline_mlp(batch_sizes_moe, 5760, 5760, 128, 4, dense_dtypes[0], dense_dtypes[1], TP=1, EP=1,
                     name="gpt-oss-x2")
        roofline_mlp(batch_sizes_moe, 5760, 5760, 128, 4, quantized_dtypes[0], quantized_dtypes[1], TP=1, EP=1,
                     name="gpt-oss-x2")
        roofline_mlp(batch_sizes_moe, 5760, 5760, 128, 4, quantized_dtypes[0], quantized_dtypes[1], TP=2, EP=1,
                     name="gpt-oss-x2")
        roofline_mlp(batch_sizes_moe, 5760, 5760, 128, 4, quantized_dtypes[0], quantized_dtypes[1], TP=4, EP=1,
                     name="gpt-oss-x2")
        roofline_mlp(batch_sizes_moe, 5760, 5760, 128, 4, quantized_dtypes[0], quantized_dtypes[1], TP=8, EP=1,
                     name="gpt-oss-x2")
