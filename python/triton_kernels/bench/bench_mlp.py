from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
import triton.profiler as proton
from triton.profiler import viewer
import torch
import argparse
import triton_kernels
import triton_kernels.swiglu
from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
from triton_kernels.target_info import is_hip, get_cdna_version
from dataclasses import dataclass
import distributed as triton_dist
from triton_kernels.tensor_details import layout
from bench_utils import quantize_weight

if torch.cuda.is_available() and not is_hip():
    from triton._C.libtriton import nvidia

    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


@dataclass
class PerfData:
    time: float
    flops: float
    bytes: float
    bitwidth: int
    device_type: str
    device_info: dict

    @property
    def tflops(self):
        return self.flops / self.time * 1e-3

    @property
    def tbps(self):
        return self.bytes / self.time * 1e-3

    @property
    def opint(self):
        # operational intensity
        assert self.bytes > 0
        return self.flops / self.bytes

    @property
    def max_tbps(self):
        return (proton.specs.max_bps(
            self.device_type,
            self.device_info["arch"],
            self.device_info["bus_width"],
            self.device_info["memory_clock_rate"],
        ) * 1e-12)

    @property
    def max_tflops(self):
        return (proton.specs.max_flops(
            self.device_type,
            self.device_info["arch"],
            self.bitwidth,
            self.device_info["num_sms"],
            self.device_info["clock_rate"],
        ) * 1e-12)

    @property
    def util(self) -> float:
        assert self.bitwidth in (8, 16)
        min_t_flop = self.flops / self.max_tflops * 1e-3
        min_t_bw = self.bytes / self.max_tbps * 1e-3
        return max(min_t_flop, min_t_bw) / self.time


def get_bench_path(name, rank, x_dtype, w_dtype, TP, EP):
    return Path(f"logs/{name}/{rank}/{x_dtype}-{w_dtype}-TP{TP}-EP{EP}/")


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
    fpath = get_bench_path(name, rank, x_dtype, w_dtype, TP, EP) / f"profiles/batch-{batch}.hatchet"
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
    # Compute total time (incl. "not useful" work)
    time = gf.filter("MATCH ('*', c) WHERE c IS LEAF").dataframe["time (ns)"].sum()
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


def roofline_mlp(batch_ranges, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP=1, EP=1, name="",
                 verbose=True):
    from itertools import chain
    from bisect import bisect_left

    batches = list(chain(*[range(*r) for r in batch_ranges]))
    # collect performance data
    perfs = []
    bench_case = f"{name} ({x_dtype}x{w_dtype}, TP={TP}, EP={EP})"
    print(f"Benchmarking {bench_case}...")
    print("===============================================================")
    for batch in batches:
        perfs += [bench_mlp(batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, name)]
        if verbose:
            print(f"Batch: {batch}; Util: {perfs[-1].util}; TFLOPS: {perfs[-1].tflops}; TBPS: {perfs[-1].tbps}")
    print("===============================================================")
    # machine limits
    max_tbps = perfs[0].max_tbps
    max_tflops = perfs[0].max_tflops
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    ax.set_xlabel("batch size (toks/expt)")
    ax.set_ylabel("performance  [TFLOP/s]")
    ax.set_title(f"{bench_case} roofline")
    # add a tiny margin so points are not flush with the frame
    xs = [batch * n_expts_act / n_expts_tot for batch in batches]
    perf = [p.tflops for p in perfs]
    xmin, xmax = min(xs), max(xs)
    dx = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(100, max_tflops + 500)
    # plot roofline
    opints = [p.opint for p in perfs]
    knee = bisect_left(opints, max_tflops / max_tbps)
    if knee > 0:  # has a bandwidth-bound knee
        x_bw = [xs[0], xs[knee - 1]]
        y_bw = [opints[0] * max_tbps, max_tflops]
    else:  # no knee found, compute-bound only
        x_bw = y_bw = []
    x_comp = xs[knee:]
    y_comp = [max_tflops] * len(x_comp)
    ax.plot(x_bw, y_bw, "--", label=f"BW-bound  ({max_tbps:.1f} TB/s)", color="blue")
    ax.plot(x_comp, y_comp, "--", label=f"Compute-bound  ({max_tflops:.0f} TFLOP/s)", color="orange")
    # plot data
    ax.scatter(xs, perf, marker="+")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, which="both", ls=":", lw=0.5)
    fig.tight_layout()
    rank, _ = triton_dist.setup()
    fpath = get_bench_path(name, rank, x_dtype, w_dtype, TP, EP) / "roofline.png"
    plt.savefig(fpath)


if __name__ == "__main__":
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_cdna_version() == 4
    batch_ranges_dense = [(1024, 32768, 1024)]
    batch_ranges_moe = [(128, 512, 32), (512, 32000, 128)]
    dense_dtypes = ["fp8", "fp8"]
    quantized_dtypes = ["fp8", "mx4"] if has_native_mx4 else ["bf16", "mx4"]
    rank, world_size = triton_dist.setup()
    if world_size > 1:
        # Running all workloads at once may cause OOM on some GPUs such as H100 80GB.
        # Thus we request users to run each workload separately.
        # For example, all eligible combinations of options are listed below when four GPUs are used:
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 2 --ep 2 --name llama4-maverick
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 1 --ep 4 --name llama4-maverick
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 4 --ep 1 --name llama4-maverick
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 4 --ep 1 --name dense
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 2 --ep 2 --name llama4-maverick --quantized
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 1 --ep 4 --name llama4-maverick --quantized
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 4 --ep 1 --name llama4-maverick --quantized
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 4 --ep 1 --name dense --quantized
        argparse = argparse.ArgumentParser()
        argparse.add_argument("--tp", type=int, default=1)
        argparse.add_argument("--ep", type=int, default=1)
        argparse.add_argument("--name", type=str, choices=["dense", "llama4-maverick"])
        argparse.add_argument("--quantized", action="store_true", default=False)
        args = argparse.parse_args()
        dtypes = dense_dtypes if args.quantized else quantized_dtypes
        if args.name == "dense":
            assert args.ep == 1, "EP must be 1 for dense"
            roofline_mlp(batch_ranges_dense, 8192, 8192, 1, 1, *dtypes, TP=args.tp, EP=args.ep, name="dense")
        else:
            roofline_mlp(batch_ranges_moe, 5120, 8192, 128, 4, *dtypes, TP=args.tp, EP=args.ep, name="llama4-maverick")
        triton_dist.cleanup()
    else:
        roofline_mlp(batch_ranges_dense, 8192, 8192, 1, 1, *dense_dtypes, TP=1, EP=1, name="dense")
        roofline_mlp(batch_ranges_dense, 8192, 8192, 1, 1, *quantized_dtypes, TP=1, EP=1, name="dense")
        roofline_mlp(batch_ranges_moe, 5120, 8192, 128, 4, *dense_dtypes, TP=1, EP=1, name="llama4-maverick")
        roofline_mlp(batch_ranges_moe, 5120, 8192, 128, 4, *quantized_dtypes, TP=1, EP=1, name="llama4-maverick")
