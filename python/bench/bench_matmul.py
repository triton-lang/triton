import triton
import torch
import os


def rounded_linspace(low, high, steps, div):
    ret = torch.linspace(low, high, steps)
    ret = (ret.int() + div - 1) // div * div
    ret = torch.unique(ret)
    return list(map(int, ret))


# Square benchmarks
nt = {False: "n", True: "t"}
square_confs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=rounded_linspace(512, 8192, 32, 128),
        y_name="provider",
        y_vals=["torch", "triton", "cutlass"],
        y_lines=["Torch", "Triton", "CUTLASS"],
        ylabel="TFLOPS",
        loglog=False,
        plot_name=f"matmul-square-{nt[AT]}{nt[BT]}",
        args={
            "AT": AT,
            "BT": BT,
            "dtype": torch.float16
        },
    ) for AT in [False] for BT in [False]
]

# Transformer training benchmarks
transformer_confs = [
    triton.testing.Benchmark(
        x_names=[x],
        x_vals = rounded_linspace(NK//16, NK, 32, 128),
        y_name="provider",
        y_vals=["torch", "triton", "cutlass"],
        y_lines=["Torch", "Triton", "CUTLASS"],
        ylabel="TFLOPS",
        loglog=False,
        plot_name=f"matmul-M{M}-{'NK'.replace(x, '')}{NK}",
        args= {"M": M, 'NK'.replace(x,''): NK, "AT": False, "BT": False, "dtype": torch.float16}
    ) for NK in [8192]\
      for i, x in enumerate(["N", "K"])\
      for M in [2048]
]


@triton.testing.perf_report(transformer_confs)
def bench_op(M, N, K, AT, BT, dtype, provider, warmup=10, rep=50):
    a = torch.rand((K, M) if AT else (M, K), device="cuda", dtype=dtype)
    b = torch.rand((N, K) if BT else (K, N), device="cuda", dtype=dtype)
    if AT: a = a.t()
    if BT: b = b.t()
    num_flops = 2 * M * N * K
    tflops = lambda ms: 2. * M * N * K / ms * 1e-9
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=warmup, rep=rep)
        return tflops(ms), tflops(max_ms), tflops(min_ms)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton.ops.matmul(a, b), warmup=warmup, rep=rep)
        return tflops(ms), tflops(max_ms), tflops(min_ms)
    if provider == "cutlass" and "CUTLASS_PROFILER" in os.environ:
        import subprocess
        import tempfile
        import pandas as pd
        # run program specified by CUTLASS_PROFILER env variable
        layout_a = "column" if AT else "row"
        layout_b = "column" if BT else "row"
        # create temporary file name
        fd, fname = tempfile.mkstemp()
        # run program and gets its output
        cmd = [
            os.environ["CUTLASS_PROFILER"],
            f"--m={M}",
            f"--n={N}",
            f"--k={K}",
            f"--A=f16:{layout_a}",
            f"--B=f16:{layout_b}",
            "--C=f16:column",
            "--accum=f32",
            "--operation=gemm",
            "--verification-enabled=false",
            f"--warmup-iterations={warmup}",
            f"--profiling-iterations={rep}",
            f"--output={fname}",
            "--dist=uniform,min:0,max:1,scale:-1",
            "--verbose=false",
        ]
        # run cmd
        subprocess.run(cmd, stdout=subprocess.PIPE)
        # read CSV output
        df_c = pd.read_csv(f"{fname}.gemm.csv")
        tflops = max(df_c["GFLOPs"]) / 1e3
        return tflops
    return None
