from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
import triton.profiler as proton
from triton.profiler import viewer
import torch
import triton_kernels
import triton_kernels.swiglu
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp, SwizzlingType
from triton_kernels.matmul_ogs import MicroscalingCtx, matmul_ogs, PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
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


def quantize(w, dtype, dev, **opt):
    if dtype == "bf16":
        wq = w.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(), MicroscalingCtx()
    elif dtype == "fp8":
        fp8e4_dtype = torch.float8_e4m3fn if get_cdna_version() != 3 \
            else torch.float8_e4m3fnuz
        wq = w.to(fp8e4_dtype)
        return wq, InFlexData(dtype=wq.dtype, scale=w.abs().max().unsqueeze(0)), \
                   MicroscalingCtx()
    else:
        assert dtype == "mx4", f"{dtype=}"
        swizzle_mx_scale = opt.get("swizzle_mx_scale", None)
        swizzle_mx_value = opt.get("swizzle_mx_value", None)
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
        return proton.specs.max_bps(self.device_type, self.device_info["arch"], self.device_info["bus_width"],
                                    self.device_info["memory_clock_rate"]) * 1e-12

    @property
    def max_tflops(self):
        return proton.specs.max_flops(self.device_type, self.device_info["arch"], self.bitwidth,
                                      self.device_info["num_sms"], self.device_info["clock_rate"]) * 1e-12

    @property
    def util(self) -> float:
        assert self.bitwidth in (8, 16)
        min_t_flop = self.flops / self.max_tflops * 1e-3
        min_t_bw = self.bytes / self.max_tbps * 1e-3
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
    proton.start(str(fpath.with_suffix('')), hook="launch")
    for i in range(100):
        if n_expts_tot > 1:
            logits = matmul_ogs(xg, wg, bg, precision_config=pcg)
            rdata, gather_indx, scatter_indx = routing(logits, n_expts_act, simulated_ep=EP)
        else:
            rdata, gather_indx, scatter_indx = None, None, None
        x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1, fused_activation=act)
        x = matmul_ogs(x, w2, b2, rdata, scatter_indx=scatter_indx, precision_config=pc2)
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
    return PerfData(time=time, flops=flops, bytes=bytes, bitwidth=x.dtype.itemsize * 8, device_type=device_type,
                    device_info=device_info)


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
    knee = bisect_left(opints, max_tflops / max_tbps) - 1
    x_bw, x_comp = xs[:knee], xs[knee:]
    x_bw = [x_bw[0], x_comp[0]]
    y_bw = [opints[0] * max_tbps, max_tflops]
    y_comp = [max_tflops] * len(x_comp)
    ax.plot(x_bw, y_bw, "--", label=f"BW-bound  ({max_tbps:.1f} TB/s)")
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
    batch_ranges_dense = [(1024, 32768, 1024)]
    batch_ranges_moe = [(128, 512, 32), (512, 32000, 128)]
    dense_dtypes = ["fp8", "fp8"]
    quantized_dtypes = ["fp8", "mx4"] if has_native_mx4 else ["bf16", "mx4"]
    roofline_mlp(batch_ranges_dense, 8192, 8192, 1, 1, *dense_dtypes, TP=1, EP=1, name="dense")
    roofline_mlp(batch_ranges_dense, 8192, 8192, 1, 1, *quantized_dtypes, TP=1, EP=1, name="dense")
    roofline_mlp(batch_ranges_moe, 5120, 8192, 128, 4, *dense_dtypes, TP=1, EP=1, name="llama4-maverick")
    roofline_mlp(batch_ranges_moe, 5120, 8192, 128, 4, *quantized_dtypes, TP=1, EP=1, name="llama4-maverick")
