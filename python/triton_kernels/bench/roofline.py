import ctypes
import matplotlib.pyplot as plt
import triton
from triton._C.libtriton import nvidia
import torch


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


def roofline(xs, perfs, dtype, fpath, title="", xlabel="", device_id=None):
    from bisect import bisect_left
    # roofline limits
    max_tbps = get_memset_tbps()
    max_tflops = get_cublas_tflops(dtype)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("performance  [TFLOP/s]")
    ax.set_title(title)
    # add a tiny margin so points are not flush with the frame
    perf = [p.tflops for p in perfs]
    xmin, xmax = min(xs), max(xs)
    dx = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(100, max_tflops + 500)
    # plot roofline
    opints = [p.flops / p.bytes for p in perfs]
    knee = bisect_left(opints, max_tflops / max_tbps)
    if knee > 0:  # has a bandwidth-bound knee
        x_bw = [xs[0], xs[knee - 1]]
        y_bw = [opints[0] * max_tbps, max_tflops]
    else:  # no knee found, compute-bound only
        x_bw = y_bw = []
    x_comp = xs[max(knee - 1, 0):]
    y_comp = [max_tflops] * len(x_comp)
    ax.plot(x_bw, y_bw, "--", label=f"BW-bound  (memset @ {max_tbps:.1f} TB/s)", color="blue")
    ax.plot(x_comp, y_comp, "--", label=f"Compute-bound  (cuBLAS @ {max_tflops:.0f} TFLOP/s)", color="orange")
    # plot data
    ax.scatter(xs, perf, marker="+")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, which="both", ls=":", lw=0.5)
    fig.tight_layout()
    plt.savefig(fpath)
