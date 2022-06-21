import torch

import triton


def rounded_linspace(low, high, steps, div):
    ret = torch.linspace(low, high, steps)
    ret = torch.div(ret.int() + div - 1, div, rounding_mode='trunc') * div
    ret = torch.unique(ret)
    return list(map(int, ret))


# Square benchmarks
nt = {False: "n", True: "t"}
square_confs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=rounded_linspace(512, 8192, 32, 128),
        line_arg="provider",
        line_vals=["cublas", "triton", "cutlass"],
        line_names=["cuBLAS", "Triton", "CUTLASS"],
        ylabel="TFLOPS",
        plot_name=f"matmul-square-{nt[AT]}{nt[BT]}",
        args={"AT": AT, "BT": BT, "dtype": torch.float16},
    ) for AT in [False] for BT in [False]
]

# Transformer training benchmarks
transformer_confs = [
    triton.testing.Benchmark(
        x_names=[x],
        x_vals=rounded_linspace(NK // 16, NK, 32, 128),
        line_arg="provider",
        line_vals=["cublas", "triton", "cutlass"],
        line_names=["cuBLAS", "Triton", "CUTLASS"],
        ylabel="TFLOPS",
        plot_name=f"matmul-M{M}-{'NK'.replace(x, '')}{NK}",
        args={"M": M, 'NK'.replace(x, ''): NK, "AT": False, "BT": False, "dtype": torch.float16}
    ) for NK in [12288]
    for i, x in enumerate(["N", "K"])
    for M in [2048]
]


@triton.testing.perf_report(square_confs)
def bench_op(M, N, K, AT, BT, dtype, provider, warmup=25, rep=75):
    a = torch.rand((K, M) if AT else (M, K), device="cuda", dtype=dtype)
    b = torch.rand((N, K) if BT else (K, N), device="cuda", dtype=dtype)
    if AT:
        a = a.t()
    if BT:
        b = b.t()
    tflops = lambda ms: 2. * M * N * K / ms * 1e-9
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=warmup, rep=rep)
        return tflops(ms), tflops(max_ms), tflops(min_ms)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton.ops.matmul(a, b), warmup=warmup, rep=rep)
        return tflops(ms), tflops(max_ms), tflops(min_ms)
    if provider == "cutlass":
        cutlass_matmul = triton.testing.cutlass_matmul
        try:
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: cutlass_matmul(a, b), warmup=warmup, rep=rep)
            return tflops(ms), tflops(max_ms), tflops(min_ms)
        except Exception:
            return None
    return None
