from pathlib import Path
from dataclasses import dataclass
from triton_kernels.numerics import InFlexData
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.tensor import convert_layout
from triton_kernels.tensor import wrap_torch_tensor, FP4
from triton_kernels.target_info import is_cuda, get_cdna_version, cuda_capability_geq
import matplotlib.pyplot as plt
import triton.profiler as proton
import torch


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


def quantize_weight(w, dtype, **opt):
    if dtype == "bf16":
        wq = w.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(), None
    elif dtype == "fp8":
        fp8e4_dtype = torch.float8_e4m3fn if get_cdna_version() != 3 else torch.float8_e4m3fnuz
        wq = w.to(fp8e4_dtype)
        if is_cuda() and not cuda_capability_geq(10, 0):
            wq = wq.transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(dtype=wq.dtype, scale=w.abs().max().unsqueeze(0)), None
    else:
        assert dtype == "mx4", f"{dtype=}"
        w, w_scale = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=1)
        if opt:
            w = convert_layout(wrap_torch_tensor(w, dtype=FP4), opt["value_layout"], **opt["value_layout_opts"])
            w_scale = convert_layout(wrap_torch_tensor(w_scale), opt["scale_layout"], **opt["scale_layout_opts"])
        return w, InFlexData(), w_scale


def roofline_mlp(batch_ranges, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, bench_mlp, TP=1, EP=1, name="",
                 rank=None, verbose=True):
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
    rank_part = f"/{rank}" if rank is not None else ""
    fpath = Path(f"logs/{name}{rank_part}/{x_dtype}-{w_dtype}-TP{TP}-EP{EP}/roofline.png")
    plt.savefig(fpath)
