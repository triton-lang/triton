import ctypes
import matplotlib.pyplot as plt
import triton
from triton._C.libtriton import nvidia
import torch
import csv
from dataclasses import dataclass
import inspect


@dataclass
class PerfRecord:
    time_ns: float
    flops: float
    bytes: float


def parse_profile(profile_path, useful_op_regex):
    """
    construct a PerfRecord from a (proton) profile path and a regex for useful operations
    """
    from triton.profiler import viewer
    gf, _, _, _ = viewer.read(profile_path)
    # aggregate "useful" flops + bytes
    useful = gf.filter(f"MATCH ('*', c) WHERE c.'name' =~ '{useful_op_regex}' AND c IS LEAF").dataframe
    bytes = int(useful["bytes"].sum())
    flops = int(sum(useful[[c for c in ["flops8", "flops16"] if c in useful.columns]].sum()))
    # take all ops (incl. "not useful" ones) when computing total time
    allops = gf.filter("MATCH ('*', c) WHERE c IS LEAF").dataframe
    time_ns = allops["time (ns)"].sum()
    return PerfRecord(time_ns=time_ns, flops=flops, bytes=bytes)


# -- compute roofline --


def write_csv(xs, perfs, fpath):
    csv_path = fpath.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "flops", "bytes", "time_ns"])
        for x, p in zip(xs, perfs):
            writer.writerow([x, p.flops, p.bytes, p.time_ns])
    return csv_path

def compute_roofline(*args, \
                  bench_fn, intensity_proxy_name, intensity_proxy_values, out_path, verbose, \
                  **kwargs):
    # validate input args
    if not isinstance(intensity_proxy_name, str):
        raise TypeError("intensity_proxy must be a string naming a parameter in target_fn")
    # determine position of intensity_proxy in target_fn signature
    sig = inspect.signature(bench_fn)
    params = list(sig.parameters.values())
    if intensity_proxy_name not in sig.parameters:
        raise ValueError(f"Parameter '{intensity_proxy_name}' not found in {bench_fn.__name__} signature")
    pos_index = [p.name for p in params].index(intensity_proxy_name)

    # wrapper to inject intensity proxy into target_fn and call it
    def inject_proxy_and_call(val, args, kwargs):
        args_list = list(args)
        args_list.insert(pos_index, val)
        return bench_fn(*args_list, **kwargs)

    # collect performance data
    perfs = []
    if verbose:
        print("=========================================")
        print(f"{out_path   }...")
        print("=========================================")
    for val in intensity_proxy_values:
        perf = inject_proxy_and_call(val, args, kwargs)
        perfs.append(perf)
        if verbose:
            tflops = perfs[-1].flops / perfs[-1].time_ns * 1e-3
            tbps = perfs[-1].bytes / perfs[-1].time_ns * 1e-3
            ms = perfs[-1].time_ns / 1e6
            print(f"{intensity_proxy_name}: {val:5d} | MS: {ms:.2f} | TFLOPS: {tflops:#.4g} | TBPS: {tbps:.2f}")
    # write to csv
    return write_csv(intensity_proxy_values, perfs, out_path)


# -- plot roofline --


def get_memset_tbps():
    # Measure device memory set bandwidth using CUDA driver API (cuMemsetD8Async)
    if torch.version.cuda is None:
        raise RuntimeError("get_memset_tbps is only supported on CUDA")
    # load cuda
    cuda = ctypes.CDLL("libcuda.so")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    if cuda.cuInit(0) != 0:
        raise RuntimeError("cuInit failed")
    # initialize cuMemsetD8Async
    cuda.cuMemsetD8Async.argtypes = [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t, ctypes.c_void_p]
    cuda.cuMemsetD8Async.restype = ctypes.c_int
    # benchmark `cuMemsetD8Async`
    n_bytes = 1 << 32
    buf = torch.empty(n_bytes, device="cuda", dtype=torch.uint8)
    dptr = ctypes.c_uint64(buf.data_ptr())
    fn = lambda: cuda.cuMemsetD8Async(dptr, ctypes.c_ubyte(0), ctypes.c_size_t(n_bytes), ctypes.c_void_p(0))
    time_ms = triton.testing.do_bench(fn, rep=1000)
    tbps = (n_bytes / (time_ms * 1e-3)) * 1e-12
    return tbps


def get_cublas_tflops(dtype):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[dtype]
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
    device = "cuda"
    M, N, K = 8192, 8192, 8192
    a = torch.randn(M, K, device=device, dtype=torch.float32).to(dtype)
    b = torch.randn(K, N, device=device, dtype=torch.float32).to(dtype).T
    c = torch.empty((M, N), device=device, dtype=dtype)
    time_ms = triton.testing.do_bench(lambda: cublas.matmul(a, b, c), rep=1000)
    return 2 * M * N * K / time_ms * 1e-9


# Load CSV series: expect columns x, flops, bytes, time_ns (or time)
def load_perf_csv(path):
    xs, flops, bytes_, times = [], [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # Support both time_ns and time as column names
        has_time_ns = "time_ns" in reader.fieldnames
        has_time = "time" in reader.fieldnames
        if not (has_time_ns or has_time):
            raise ValueError(f"CSV {path} missing time_ns/time column")
        for row in reader:
            xs.append(int(row["x"]))
            flops.append(int(row["flops"]))
            bytes_.append(int(row["bytes"]))
            tval = row["time_ns"] if has_time_ns else row["time"]
            times.append(int(float(tval)))
    return xs, flops, bytes_, times


def validate_perfs(perfs):
    xs_ref, flops_ref, bytes_ref, _ = perfs[0]
    for _, (xs, flops, bytes, _) in enumerate(perfs[1:], start=1):
        for i in range(len(xs)):
            if xs[i] != xs_ref[i]:
                raise ValueError(f"x mismatch between series[0] and series[{i}]")


def plot_roofline(series, flops_dtype, out_path, max_tbps, max_tflops, title="", xlabel="", labels=None,
                  zoom_range=None, zoom_loc="upper right", zoom_size=("45%", "45%"), points_of_interest=None):
    from bisect import bisect_left
    from pathlib import Path
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import ConnectionPatch
    from matplotlib.ticker import MaxNLocator

    perfs = [load_perf_csv(p) for p in series]
    validate_perfs(perfs)
    xs, flops_ref, bytes_ref, _ = perfs[0]
    n = len(xs)

    if not isinstance(max_tbps, int):
        assert max_tbps == "memset"
        max_tbps = get_memset_tbps()
    if not isinstance(max_tflops, int):
        assert max_tflops == "cublas"
        max_tflops = get_cublas_tflops(flops_dtype)

    grey = "#7f7f7f"
    opints = [f / b for f, b in zip(flops_ref, bytes_ref)]  # arithmetic intensity per sample
    kappa = max_tflops / max_tbps  # intensity at the knee

    # --- knee interpolation ---
    knee_idx = bisect_left(opints, kappa)
    if knee_idx <= 0:
        x_knee, knee_in_domain = xs[0], True
    elif knee_idx >= n:
        x_knee, knee_in_domain = xs[-1], False
    else:
        i0, i1 = knee_idx - 1, knee_idx
        t = (kappa - opints[i0]) / (opints[i1] - opints[i0])
        x_knee = xs[i0] + t * (xs[i1] - xs[i0])
        knee_in_domain = True

    # --- piecewise roofline segments (for plotting the grey guideline) ---
    if knee_idx >= n:
        bw_x, bw_y = xs[:], [op * max_tbps for op in opints]
        comp_x, comp_y = [], []
    elif knee_idx <= 0:
        bw_x, bw_y = [], []
        comp_x, comp_y = xs[:], [max_tflops] * n
    else:
        bw_x = xs[:knee_idx] + [x_knee]
        bw_y = [op * max_tbps for op in opints[:knee_idx]] + [max_tflops]
        comp_x = [x_knee] + xs[knee_idx:]
        comp_y = [max_tflops] * (1 + (n - knee_idx))

    y_roof_sampled = [min(op * max_tbps, max_tflops) for op in opints]

    # --- helpers ---
    def idx_range(xlist, lo, hi):
        i0 = bisect_left(xlist, lo)
        i1 = bisect_left(xlist, hi, lo=i0)
        return i0, max(i1, i0 + 1) if i0 < len(xlist) else (i0, i0)

    def visible_y(x0, x1):
        i0, i1 = idx_range(xs, x0, x1)
        vals = [min(op * max_tbps, max_tflops) for op in opints[i0:i1]]
        if knee_in_domain and (x0 <= x_knee <= x1):
            vals.append(max_tflops)
        if not vals:
            return (0.0, 1.0)
        lo, hi = min(vals), max(vals)
        return lo, hi

    def interp(yxs, yys, x):
        """Linear interpolation on (xs, ys), clamped at the ends."""
        j = bisect_left(yxs, x)
        if j <= 0:
            return yys[0]
        if j >= len(yxs):
            return yys[-1]
        x0, x1 = yxs[j - 1], yxs[j]
        y0, y1 = yys[j - 1], yys[j]
        t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
        return y0 + t * (y1 - y0)

    # Prepare series curves
    series_perf, series_labels = [], []
    for idx, (pth, (_, f, b, t)) in enumerate(zip(series, perfs)):
        perf = [ff / tt * 1e-3 if tt > 0 else 0.0 for ff, tt in zip(f, t)]
        series_perf.append(perf)
        series_labels.append(labels[idx] if labels and idx < len(labels) else Path(pth).stem)

    def plot_on(ax, extent=None, add_legend=False, annotate=False):
        # Grey roofline
        if bw_x:
            ax.plot(bw_x, bw_y, ls="--", color=grey, label=f"BW-bound - {max_tbps:.1f} TB/s [memset]")
        if comp_x:
            ax.plot(comp_x, comp_y, ls=":", color=grey, label=f"Compute-bound - {max_tflops:.0f} TFLOP/s [cuBLAS]")

        # Series
        for lab, perf in zip(series_labels, series_perf):
            ax.plot(xs, perf, label=lab if add_legend else None, lw=1.8 if add_legend else 1.4, zorder=2)

        # Layout
        xmin, xmax = xs[0], xs[-1]
        if extent:
            x0, x1 = max(extent[0], xmin), min(extent[1], xmax)
            ax.set_xlim(x0, x1)
            lo, hi = visible_y(x0, x1)
            pad = 0.08 * (hi - lo) if hi > lo else max(0.1, 0.1 * hi)
            ax.set_ylim(max(0.0, lo - pad), hi + pad)
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
        else:
            dx = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
            ax.set_xlim(xmin - dx, xmax + dx)
            ax.set_ylim(min(y_roof_sampled) * 0.8 if y_roof_sampled else 0.0, max_tflops * 1.05)

        # Points of interest
        if annotate and points_of_interest:
            for x_pt, label in points_of_interest.items():
                if (extent is not None) and not (ax.get_xlim()[0] <= x_pt <= ax.get_xlim()[1]):
                    continue
                y_pt = interp(xs, series_perf[0], x_pt)
                ax.plot([x_pt], [y_pt], marker="o", ms=4, mfc="white", mec="black", zorder=3)
                ax.annotate(f"{label}\n({y_pt:.2f})", xy=(x_pt, y_pt), xytext=(5, 8), textcoords="offset points",
                            fontsize=7 if extent else 8, ha="left", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="0.6",
                                      alpha=0.8), arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))

    def draw_connectors(ax_main, ax_inset, x0, x1):
        y0_main = ax_main.get_ylim()[0]
        y0_in = ax_inset.get_ylim()[0]
        style = dict(ls=(0, (2, 2)), color="lightcoral", lw=0.4)
        for xx in (x0, x1):
            ax_main.add_artist(
                ConnectionPatch(xyA=(xx, y0_main), coordsA=ax_main.transData, xyB=(xx, y0_in),
                                coordsB=ax_inset.transData, **style))

    # --- draw ---
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("performance  [TFLOP/s]")
    ax.set_title(title)
    plot_on(ax, extent=None, add_legend=True, annotate=True)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, which="both", ls=":", lw=0.5)

    if zoom_range is not None:
        x0, x1 = sorted(zoom_range)
        assert x1 > x0
        axins = inset_axes(ax, width=zoom_size[0], height=zoom_size[1], loc=zoom_loc)
        plot_on(axins, extent=(x0, x1), add_legend=False, annotate=True)
        axins.grid(True, which="both", ls=":", lw=0.4)
        draw_connectors(ax, axins, x0, x1)

    plt.savefig(out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot roofline(s) from perf CSV series")
    parser.add_argument("--series", type=str, nargs="+", required=True,
                        help="list of .csv files; columns must be `x`, `flops`, `bytes`, `time_ns`")
    parser.add_argument("--dtype", type=str, required=True, choices=["fp16", "bf16", "fp8"],
                        help="data type used for compute-bound roof")
    parser.add_argument("--out_path", type=str, required=True, help="path to write the output image")
    parser.add_argument("--title", type=str, default="", help="plot title")
    parser.add_argument("--xlabel", type=str, default="", help="x-axis label")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="optional list of names for each series, in order; must match number of --series")
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.series):
        parser.error("--labels must have the same number of entries as --series")
    plot_roofline(args.series, args.dtype, args.out_path, title=args.title, xlabel=args.xlabel, labels=args.labels)
