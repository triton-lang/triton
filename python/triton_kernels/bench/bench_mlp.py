from pathlib import Path
import matplotlib.pyplot as plt
import json
import triton.profiler as proton
import torch
import triton_kernels.swiglu
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.matmul_ogs import MicroscalingCtx, matmul_ogs, PrecisionConfig, FlexCtx
from triton_kernels.numerics import InFlexData
from triton_kernels.routing import routing
from triton_kernels.target_info import is_hip, get_cdna_version
from dataclasses import dataclass

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
        "HGX GB200": {"MAX_TFLOPS8": 4500, "MAX_TFLOPS16": 2250, "MAX_TBPS": 8.0},
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
        swizzle_axis = 2 if swizzle_mx_scale else None
        w = w.to(torch.bfloat16)
        w, mx_scales, weight_scale_shape = downcast_to_mxfp(w, torch.uint8, axis=1, swizzle_axis=swizzle_axis)
        return w, InFlexData(), MicroscalingCtx(weight_scale=mx_scales, swizzle_mx=swizzle_mx_scale,
                                                actual_weight_scale_shape=weight_scale_shape)


@dataclass
class PerfData:
    time: float
    flops: float
    bytes: float

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

        peak_flops = max(SPECS["MAX_TFLOPS8"], SPECS.get("MAX_TFLOPS16", 0))
        min_t_flop = self.flops / peak_flops * 1e-3  # ns → µs
        min_t_bw = self.bytes / SPECS["MAX_TBPS"] * 1e-3
        return max(min_t_flop, min_t_bw) / self.time


def bench_mlp(batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, name):
    assert n_expts_tot % EP == 0
    assert dim2 % TP == 0
    dev = "cuda"

    # input
    # weights
    wg = torch.randn((dim1, n_expts_tot), device=dev)
    w1 = torch.randn((n_expts_tot // EP, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot // EP, dim2 // TP // 2, dim1), device=dev)
    # biases
    bg = torch.randn((n_expts_tot, ), device=dev)
    b1 = torch.randn((n_expts_tot // EP, dim2 // TP), device=dev)
    b2 = torch.randn((n_expts_tot // EP, dim1), device=dev)

    # -- numerics --
    optg = dict()
    opt1 = {"swizzle_mx_scale": True} if w_dtype == "mx4" else dict()
    opt2 = {"swizzle_mx_scale": True} if w_dtype == "mx4" else dict()
    wg, wg_flex, wg_mx = quantize(wg, "bf16", dev, **optg)
    w1, w1_flex, w1_mx = quantize(w1, w_dtype, dev, **opt1)
    w2, w2_flex, w2_mx = quantize(w2, w_dtype, dev, **opt2)
    pcg = PrecisionConfig(mx_ctx=wg_mx, flex_ctx=FlexCtx(rhs_data=wg_flex))
    pcs = triton_kernels.swiglu.PrecisionConfig(limit=1.0)
    pc1 = PrecisionConfig(mx_ctx=w1_mx, flex_ctx=FlexCtx(rhs_data=w1_flex))
    pc2 = PrecisionConfig(mx_ctx=w2_mx, flex_ctx=FlexCtx(rhs_data=w2_flex))

    # -- benchmark --
    fpath = Path(f"logs/{name}/{x_dtype}-{w_dtype}-TP{TP}-EP{EP}/profiles/batch-{batch}.hatchet")
    fpath.parent.mkdir(parents=True, exist_ok=True)
    x_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[x_dtype]
    # special treatment of fp8_e4m3 on AMD CDNA3 because it uses fp8_e4m3fnuz
    if x_dtype == torch.float8_e4m3fn and get_cdna_version() == 3:
        x_dtype = torch.float8_e4m3fnuz

    x = torch.randn((batch, dim1), device=dev)
    xg = x.to(wg.dtype if n_expts_tot > 1 else x_dtype)
    x = x.to(x_dtype)
    # run layer
    proton.start(str(fpath.with_suffix('')), hook="triton")
    for i in range(100):
        if n_expts_tot > 1:
            logits = matmul_ogs(xg, wg, bg, precision_config=pcg)
            rdata, gather_indx, scatter_indx = routing(logits, n_expts_act, simulated_ep=EP)
        else:
            rdata, gather_indx, scatter_indx = None, None, None
        x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1)
        x = triton_kernels.swiglu.swiglu(x, 1.0, pcs, routing_data=rdata)
        x = matmul_ogs(x, w2, b2, rdata, scatter_indx=scatter_indx, precision_config=pc2)
    proton.finalize()

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
    return PerfData(time, flops, bytes)


def roofline_mlp(batch_ranges, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP=1, EP=1, name="",
                 verbose=True):
    from itertools import chain
    from bisect import bisect_left
    batches = list(chain(*[range(*r) for r in batch_ranges]))
    # collect performance data
    perfs = []
    print(f"Benchmarking {name} ({x_dtype}x{w_dtype}, TP={TP}, EP={EP})...")
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
    ax.set_title("roofline")
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
    y_bw = [op * max_tbps for op in opints[:knee]]
    y_comp = [max_tflops] * len(x_comp)
    ax.plot(x_bw, y_bw, "--", label=f"BW-bound  ({max_tbps:.0f} TB/s)")
    ax.plot(x_comp, y_comp, "--", label=f"Compute-bound  ({max_tflops:.0f} TFLOP/s)")
    # plot data
    ax.scatter(xs, perf, marker="+")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, which="both", ls=":", lw=0.5)
    fig.tight_layout()
    fpath = Path(f"logs/{name}/{x_dtype}-{w_dtype}-TP{TP}-EP{EP}/roofline.png")
    plt.savefig(fpath)


if __name__ == "__main__":
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_cdna_version() == 4
    if SPECS is None:
        print("Current GPU has no specs provided, utilization is N/A")
    batch_ranges = [(1024, 32768, 1024)]
    dense_dtypes = ["fp8", "fp8"]
    quantized_dtypes = ["fp8", "mx4"] if has_native_mx4 else ["bf16", "mx4"]
    roofline_mlp(batch_ranges, 8192, 8192, 1, 1, *dense_dtypes, TP=1, EP=1, name="dense")
    roofline_mlp(batch_ranges, 8192, 8192, 1, 1, *quantized_dtypes, TP=1, EP=1, name="dense")
    roofline_mlp(batch_ranges, 5120, 8192, 128, 4, *dense_dtypes, TP=1, EP=1, name="llama4-maverick")
    roofline_mlp(batch_ranges, 5120, 8192, 128, 4, *quantized_dtypes, TP=1, EP=1, name="llama4-maverick")
