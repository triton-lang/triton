from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
import json
import triton.profiler as proton
import torch
import argparse
import triton_kernels
import triton_kernels.swiglu
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp, SwizzlingType
from triton_kernels.matmul_ogs import MicroscalingCtx, matmul_ogs, PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
from triton_kernels.numerics import InFlexData
from triton_kernels.target_info import is_hip, get_cdna_version
from dataclasses import dataclass

import triton_kernels.distributed as triton_dist

if torch.cuda.is_available() and not is_hip():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def _query_gpu_specs():
    import subprocess
    if is_hip():
        cmd = ["rocm-smi", "--showproductname", "-d=0", "--csv"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        model = output.splitlines()[1].split(",")[2]
        if model in ["0x74a9", "0x74a1"]:
            name = "AMD Instinct MI300X"
        elif model == "0x74a5":
            name = "AMD Instinct MI325X"
        else:
            name = "AMD"
    else:
        cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i=0"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        name = output.splitlines()[0]

    gpu_specs = {
        "NVIDIA H100 80GB HBM3": {"MAX_TFLOPS8": 1979, "MAX_TFLOPS16": 989, "MAX_TBPS": 3.35},
        "NVIDIA GB200": {"MAX_TFLOPS8": 4500, "MAX_TFLOPS16": 2250, "MAX_TBPS": 8.0},
        "AMD Instinct MI300X": {"MAX_TFLOPS8": 2615, "MAX_TFLOPS16": 1307, "MAX_TBPS": 5.3},
        "AMD Instinct MI325X": {"MAX_TFLOPS8": 2615, "MAX_TFLOPS16": 1307, "MAX_TBPS": 6.0},
    }
    return gpu_specs.get(name)


SPECS = _query_gpu_specs()


def quantize(w, dtype, dev, **opt):
    if dtype == "bf16":
        wq = w.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(), MicroscalingCtx()
    elif dtype == "fp8":
        fp8e4_dtype = torch.float8_e4m3fn if get_cdna_version() != 3 \
            else torch.float8_e4m3fnuz
        wq = w.to(fp8e4_dtype).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(dtype=wq.dtype, scale=w.abs().max().unsqueeze(0)), \
                   MicroscalingCtx()
    else:
        assert dtype == "mx4", f"{dtype=}"
        swizzle_mx_scale = opt["swizzle_mx_scale"]
        swizzle_mx_value = opt["swizzle_mx_value"]
        swizzle_axis = 2 if swizzle_mx_scale else None
        w = w.to(torch.bfloat16)
        w, mx_scales, weight_scale_shape = downcast_to_mxfp(w, torch.uint8, axis=1, swizzle_axis=swizzle_axis,
                                                            swizzle_scale=swizzle_mx_scale,
                                                            swizzle_value=swizzle_mx_value)
        return w, InFlexData(), MicroscalingCtx(weight_scale=mx_scales, swizzle_scale=swizzle_mx_scale,
                                                swizzle_value=swizzle_mx_value,
                                                actual_weight_scale_shape=weight_scale_shape)


@dataclass
class PerfData:
    time: float
    flops: float
    bytes: float
    bitwidth: int

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
    def util(self) -> float:
        if SPECS is None:
            return 0.0
        assert self.bitwidth in (8, 16)

        peak_flops = SPECS["MAX_TFLOPS8"] if self.bitwidth == 8 else SPECS["MAX_TFLOPS16"]
        min_t_flop = self.flops / peak_flops * 1e-3  # ns → µs
        min_t_bw = self.bytes / SPECS["MAX_TBPS"] * 1e-3
        return max(min_t_flop, min_t_bw) / self.time


def bench_mlp(batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, name):
    assert n_expts_tot % EP == 0
    assert dim2 % TP == 0
    local_rank, world_size = triton_dist.setup()
    dev = f"cuda:{local_rank}"
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
    optg = dict()
    opt1 = dict()
    opt2 = dict()
    if w_dtype == "mx4" and not is_hip():
        if torch.cuda.get_device_capability()[0] < 9:
            # NYI for Ampere
            swizzle_mx_value = None
            swizzle_mx_scale = None
        elif torch.cuda.get_device_capability()[0] < 10:
            swizzle_mx_value = SwizzlingType.HOPPER
            swizzle_mx_scale = SwizzlingType.HOPPER
        else:
            swizzle_mx_value = None
            swizzle_mx_scale = SwizzlingType.BLACKWELL
        opt1 = {"swizzle_mx_value": swizzle_mx_value, "swizzle_mx_scale": swizzle_mx_scale}
        opt2 = deepcopy(opt1)
    wg, wg_flex, wg_mx = quantize(wg, "bf16", dev, **optg)
    w1, w1_flex, w1_mx = quantize(w1, w_dtype, dev, **opt1)
    w2, w2_flex, w2_mx = quantize(w2, w_dtype, dev, **opt2)
    pcg = PrecisionConfig(mx_ctx=wg_mx, flex_ctx=FlexCtx(rhs_data=wg_flex))
    act = FusedActivation(FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")), (1.0, 1.0), 2)
    pc1 = PrecisionConfig(mx_ctx=w1_mx, flex_ctx=FlexCtx(rhs_data=w1_flex))
    pc2 = PrecisionConfig(mx_ctx=w2_mx, flex_ctx=FlexCtx(rhs_data=w2_flex))

    # -- benchmark --
    fpath = Path(f"logs/{name}/{rank}/{x_dtype}-{w_dtype}-TP{TP}-EP{EP}/profiles/batch-{batch}.hatchet")
    fpath.parent.mkdir(parents=True, exist_ok=True)
    x_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[x_dtype]
    # special treatment of fp8_e4m3 on AMD CDNA3 because it uses fp8_e4m3fnuz
    if x_dtype == torch.float8_e4m3fn and get_cdna_version() == 3:
        x_dtype = torch.float8_e4m3fnuz

    input_x = torch.randn((batch // DP, dim1), device=dev)
    # run layer
    proton.start(str(fpath.with_suffix('')), hook="triton")
    input_x = input_x.to(x_dtype)
    for i in range(100):
        xg = input_x.to(wg.dtype if n_expts_tot > 1 else input_x.dtype)
        x = triton_dist.all_gather(input_x, dim=0)
        if n_expts_tot > 1:
            logits = matmul_ogs(xg, wg, bg, precision_config=pcg)
            rdata, gather_indx, scatter_indx, token_mask = triton_dist.routing(logits, n_expts_act, EP=EP, TP=TP)
        else:
            rdata, gather_indx, scatter_indx, token_mask = None, None, None, None
        if token_mask is not None:
            x = x[token_mask]
        if x.nelement() > 0:
            x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1, fused_activation=act)
            x = matmul_ogs(x, w2, b2, rdata, scatter_indx=scatter_indx, precision_config=pc2)
        x = triton_dist.reduce_scatter(x, token_mask=token_mask, dim=0)
    proton.finalize()

    # -- cleanup --
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()

    # -- analyze --
    with open(f"{fpath}") as fd:
        data = json.load(fd)
        # TODO: this will be broken if kernels use scopes themselves
        # compute useful (a.k.a. matmul) bytes and flops
        matmuls = [
            x for x in data[0]["children"] if "_matmul" in x["frame"]["name"] and "metadata" not in x["frame"]["name"]
        ]
        bytes = sum([x["metrics"]["bytes"] for x in matmuls])
        flops = {w: sum([x["metrics"].get(f"flops{w}", 0) for x in matmuls]) for w in [8, 16]}
        flops = sum([flops[w] for w in [8, 16]])
        # compute total time (incl. "not useful" work)
        # TODO: proton should really be recording that in the json instead of
        # relying on the user to aggregate
        time = sum(x["metrics"].get("time (ns)", 0) for x in data[0]["children"])
    return PerfData(time, flops, bytes, x_dtype.itemsize * 8)


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
    ax.set_ylim(100, SPECS["MAX_TFLOPS8"] + 500)
    # plot roofline
    max_tbps = SPECS["MAX_TBPS"]
    max_tflops = SPECS["MAX_TFLOPS8"]
    opints = [p.opint for p in perfs]
    knee = bisect_left(opints, max_tflops / max_tbps) - 1
    x_bw, x_comp = xs[:knee], xs[knee:]
    x_bw = [x_bw[0], x_comp[0]]
    y_bw = [opints[0] * max_tbps, max_tflops]
    y_comp = [max_tflops] * len(x_comp)
    ax.plot(x_bw, y_bw, "--", label=f"BW-bound  ({max_tbps:.0f} TB/s)")
    ax.plot(x_comp, y_comp, "--", label=f"Compute-bound  ({max_tflops:.0f} TFLOP/s)")
    # plot data
    ax.scatter(xs, perf, marker="+")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, which="both", ls=":", lw=0.5)
    fig.tight_layout()
    rank, _ = triton_dist.setup()
    fpath = Path(f"logs/{name}/{rank}/{x_dtype}-{w_dtype}-TP{TP}-EP{EP}/roofline.png")
    plt.savefig(fpath)


if __name__ == "__main__":
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_cdna_version() == 4
    rank, world_size = triton_dist.setup()
    if SPECS is None:
        print("Current GPU has no specs provided, utilization is N/A")
    dense_dtypes = ["fp8", "fp8"]
    quantized_dtypes = ["fp8", "mx4"] if has_native_mx4 else ["bf16", "mx4"]
    if world_size > 1:
        # Running all workloads at once may cause OOM on some GPUs such as H100 80GB.
        # Thus we request users to run each workload separately.
        # For example, to run the dense workload with 4 TP and 2 EP, use:
        # e.g., torchrun --nproc-per-node=4 ./bench_mlp.py --tp 2 --ep 2 --name llama4-maverick
        batch_ranges_dense = [(1024, 32768, 1024)]
        batch_ranges_moe = [(128, 512, 32), (512, 32000, 128)]
        argparse = argparse.ArgumentParser()
        argparse.add_argument("--tp", type=int, default=1)
        argparse.add_argument("--ep", type=int, default=1)
        argparse.add_argument("--name", type=str, choices=["dense", "llama4-maverick"])
        argparse.add_argument("--quantized", type=bool, default=False)
        args = argparse.parse_args()
        dtypes = dense_dtypes if args.quantized else quantized_dtypes
        if args.name == "dense":
            assert args.ep == 1, "EP must be 1 for dense"
            roofline_mlp(batch_ranges_dense, 8192, 8192, 1, 1, *dense_dtypes, TP=args.tp, EP=args.ep,
                         name="dense")
        else:
            roofline_mlp(batch_ranges_moe, 5120, 8192, 128, 4, *dense_dtypes, TP=args.tp, EP=args.ep,
                         name="llama4-maverick")
        triton_dist.cleanup() 
    else:
        batch_ranges_dense = [(1024, 32768, 1024)]
        batch_ranges_moe = [(128, 512, 32), (512, 32000, 128)]
        roofline_mlp(batch_ranges_dense, 8192, 8192, 1, 1, *dense_dtypes, TP=1, EP=1, name="dense")
        roofline_mlp(batch_ranges_dense, 8192, 8192, 1, 1, *quantized_dtypes, TP=1, EP=1, name="dense")
        roofline_mlp(batch_ranges_moe, 5120, 8192, 128, 4, *dense_dtypes, TP=1, EP=1, name="llama4-maverick")
        roofline_mlp(batch_ranges_moe, 5120, 8192, 128, 4, *quantized_dtypes, TP=1, EP=1, name="llama4-maverick")
