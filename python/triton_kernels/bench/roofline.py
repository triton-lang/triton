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
            print(f"{intensity_proxy_name}: {val:5d} | TFLOPS: {tflops:#.4g} | TBPS: {tbps:.2f}")
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


def plot_roofline(series, flops_dtype, out_path, max_tbps, max_tflops, title="", xlabel="", labels=None):
    from bisect import bisect_left
    from pathlib import Path
    perfs = [load_perf_csv(p) for p in series]
    validate_perfs(perfs)
    xs, flops_ref, bytes_ref, _ = perfs[0]
    if not isinstance(max_tbps, int):
        assert max_tbps == "memset"
        max_tbps = get_memset_tbps()
    if not isinstance(max_tflops, int):
        assert max_tflops == "cublas"
        max_tflops = get_cublas_tflops(flops_dtype)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("performance  [TFLOP/s]")
    ax.set_title(title)
    xmin, xmax = min(xs), max(xs)
    dx = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(100, max_tflops + 500)

    # roofline from operational intensity (identical across series)
    opints = [f / b for f, b in zip(flops_ref, bytes_ref)]
    knee = bisect_left(opints, max_tflops / max_tbps)
    if knee > 0:
        x_bw = [xs[0], xs[knee - 1]]
        y_bw = [opints[0] * max_tbps, max_tflops]
    else:
        x_bw = y_bw = []
    x_comp = xs[max(knee - 1, 0):]
    y_comp = [max_tflops] * len(x_comp)
    grey = "#7f7f7f"
    ax.plot(x_bw, y_bw, linestyle="--", color=grey, label=f"BW-bound - {max_tbps:.1f} TB/s [memset]", zorder=1)
    ax.plot(x_comp, y_comp, linestyle=":", color=grey, label=f"Compute-bound  - {max_tflops:.0f} TFLOP/s [cuBLAS]",
            zorder=1)

    # Plot each series as a lineplot of TFLOP/s
    for idx, (pth, (_, f, b, t)) in enumerate(zip(series, perfs)):
        perf_tflops = [ff / tt * 1e-3 if tt > 0 else 0.0 for ff, tt in zip(f, t)]
        label = (labels[idx] if labels and idx < len(labels) else Path(pth).stem)
        ax.plot(xs, perf_tflops, label=label, linewidth=1.8, zorder=2)

    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, which="both", ls=":", lw=0.5)
    fig.tight_layout()
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
