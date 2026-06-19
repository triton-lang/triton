"""
Multi-CTA Matrix Multiplication
===============================

This tutorial shows how to launch a standard Triton matmul with multiple CTAs
per program. On Blackwell, Triton can automatically lower eligible
``tl.dot`` operations to MMAv5 ``cta_group::2`` instructions when the RHS
operand can be shared across the two CTAs.

You will learn:

* How to request a two-CTA launch from the Triton frontend.
* How to inspect TTGIR/PTX to confirm that MMAv5 two-CTA MMA was selected.
* How to compare a Triton multi-CTA matmul with cuBLAS.
"""

# %%
# Setup
# -----

import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


def is_cuda():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "cuda"


if torch.cuda.is_available():
    from triton._C.libtriton import nvidia

    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None

TORCH_HAS_FP8 = hasattr(torch, "float8_e4m3fn")

# %%
# Kernel
# ------
#
# This is intentionally close to the basic matmul tutorial. The only launch-time
# difference is that the wrapper below passes ``num_ctas=2``. The compiler is
# responsible for deciding whether the dot can use MMAv5 two-CTA mode.


def proton_autotune_do_bench(kernel_call, quantiles):
    return triton.testing.do_bench_proton(kernel_call, warmup=1, rep=5, quantiles=quantiles)


def matmul_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_M"]
    block_n = nargs["BLOCK_N"]
    block_k = nargs["BLOCK_K"]
    nargs["a_desc"].block_shape = [block_m, block_k]
    nargs["b_desc"].block_shape = [block_k, block_n]
    nargs["c_desc"].block_shape = [block_m, block_n]


def make_matmul_configs(configs, num_ctas):
    return [
        triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_SIZE_M": 8,
                "NUM_STAGES": num_stages,
            },
            num_stages=num_stages,
            num_warps=num_warps,
            num_ctas=num_ctas,
            pre_hook=matmul_set_block_size_hook,
        )
        for block_m, block_n, block_k, num_stages, num_warps in configs
    ]


MATMUL_CONFIGS = make_matmul_configs(
    [
        (128, 128, 64, 2, 4),
        (128, 128, 64, 3, 4),
        (128, 128, 64, 4, 4),
        (128, 256, 64, 3, 4),
        (128, 256, 64, 4, 4),
        (256, 128, 32, 3, 4),
        (256, 128, 32, 4, 4),
        (256, 128, 64, 3, 4),
        (256, 128, 64, 4, 4),
        (256, 128, 64, 4, 8),
    ],
    num_ctas=1,
)

TWO_CTA_CONFIGS = make_matmul_configs(
    [
        (256, 128, 64, 2, 4),
        (256, 128, 64, 3, 4),
        (256, 128, 64, 4, 4),
        (256, 128, 64, 5, 4),
        (256, 128, 64, 3, 8),
        (256, 128, 64, 4, 8),
        (512, 64, 64, 2, 4),
        (512, 64, 64, 3, 4),
        (512, 64, 64, 4, 4),
    ],
    num_ctas=2,
)

WS_CONFIGS = make_matmul_configs(
    [
        (128, 128, 64, 2, 4),
        (128, 128, 64, 3, 4),
        (128, 128, 64, 4, 4),
        (128, 128, 128, 2, 4),
        (128, 128, 128, 3, 4),
        (128, 128, 128, 4, 4),
        (128, 256, 64, 2, 4),
        (128, 256, 64, 3, 4),
        (128, 256, 64, 4, 4),
    ],
    num_ctas=2,
)


@triton.jit
def matmul_kernel(
        a_desc, b_desc, c_desc,  #
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr, NUM_STAGES: tl.constexpr, FP8_INPUTS: tl.constexpr,
        WARP_SPECIALIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        off_k = k * BLOCK_K
        a = a_desc.load([off_m, off_k])
        b = b_desc.load([off_k, off_n])
        acc = tl.dot(a, b, acc)

    c_desc.store([off_m, off_n], acc.to(tl.float16))


AUTOTUNE_KEY = ["M", "N", "K", "FP8_INPUTS", "WARP_SPECIALIZE"]

matmul_kernel_1cta = triton.autotune(
    configs=MATMUL_CONFIGS, key=AUTOTUNE_KEY, do_bench=proton_autotune_do_bench)(matmul_kernel)
matmul_kernel_2cta = triton.autotune(
    configs=TWO_CTA_CONFIGS, key=AUTOTUNE_KEY, do_bench=proton_autotune_do_bench)(matmul_kernel)
matmul_kernel_2cta_ws = triton.autotune(
    configs=WS_CONFIGS, key=AUTOTUNE_KEY, do_bench=proton_autotune_do_bench)(matmul_kernel)


def matmul(a, b, *, num_ctas, warp_specialize=False, out=None):
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.dtype == b.dtype, "matrix A and B must have the same dtype"
    supported_dtypes = [torch.float16]
    if TORCH_HAS_FP8:
        supported_dtypes.append(torch.float8_e4m3fn)
    assert a.dtype in supported_dtypes, "this tutorial uses fp16 or fp8 inputs"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16) if out is None else out
    a_desc = TensorDescriptor.from_tensor(a, [1, 1])
    b_desc = TensorDescriptor.from_tensor(b, [1, 1])
    c_desc = TensorDescriptor.from_tensor(c, [1, 1])
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
    if warp_specialize:
        kernel = matmul_kernel_2cta_ws
    elif num_ctas == 2:
        kernel = matmul_kernel_2cta
    else:
        kernel = matmul_kernel_1cta
    return kernel[grid](a_desc, b_desc, c_desc, M, N, K, FP8_INPUTS=(TORCH_HAS_FP8 and a.dtype == torch.float8_e4m3fn),
                        WARP_SPECIALIZE=warp_specialize)


def tflops(ms, M, N, K):
    return 2.0 * M * N * K * 1e-12 / (ms * 1e-3)


def bench(fn):
    return triton.testing.do_bench_proton(fn, warmup=1, rep=5)


BENCHMARK_SHAPES = list(range(1024, 4097, 256))


def fmt_config(config):
    return (f"{config.kwargs['BLOCK_M']}x{config.kwargs['BLOCK_N']}x{config.kwargs['BLOCK_K']}"
            f"s{config.num_stages}w{config.num_warps}")


# %%
# Correctness and IR Inspection
# -----------------------------


def validate_and_inspect():
    if not is_blackwell():
        raise RuntimeError("This tutorial requires an NVIDIA Blackwell GPU.")

    M, N, K = 1024, 1024, 1024
    torch.manual_seed(0)
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    out = torch.empty((M, N), device=DEVICE, dtype=torch.float16)

    ref = torch.matmul(a, b)
    compiled_2cta = matmul(a, b, num_ctas=2, out=out)
    torch.testing.assert_close(ref.to(torch.float32), out.to(torch.float32), atol=0.06, rtol=0.06)

    compiled_ws = matmul(a, b, num_ctas=2, warp_specialize=True, out=out)
    torch.testing.assert_close(ref.to(torch.float32), out.to(torch.float32), atol=0.06, rtol=0.06)

    ttgir = compiled_2cta.asm["ttgir"]
    ptx = compiled_2cta.asm["ptx"]
    ws_ttgir = compiled_ws.asm["ttgir"]
    ws_ptx = compiled_ws.asm["ptx"]
    print(f"TTGIR contains two_ctas: {'two_ctas' in ttgir}", flush=True)
    print(f"WS TTGIR contains two_ctas: {'two_ctas' in ws_ttgir}", flush=True)
    print(f"WS TTGIR contains ttg.warp_specialize: {'ttg.warp_specialize' in ws_ttgir}", flush=True)
    print(f"PTX contains cta_group::2: {'cta_group::2' in ptx}", flush=True)
    print(f"WS PTX contains cta_group::2: {'cta_group::2' in ws_ptx}", flush=True)


# %%
# Benchmark
# ---------
#
# ``torch.matmul`` and the explicit CublasLt helper both use cuBLAS on CUDA.


def benchmark(shapes=BENCHMARK_SHAPES, precision="fp16"):
    if not is_blackwell():
        raise RuntimeError("This tutorial requires an NVIDIA Blackwell GPU.")
    if precision not in {"fp16", "fp8", "all"}:
        raise ValueError("precision must be one of: 'fp16', 'fp8', or 'all'")

    print("Benchmarking Triton multi-CTA matmul", flush=True)
    print("====================================", flush=True)
    if precision in {"fp16", "all"}:
        benchmark_precision(shapes, precision="fp16")
    if precision in {"fp8", "all"}:
        if not TORCH_HAS_FP8 or not is_cuda():
            raise RuntimeError("fp8 benchmarking requires CUDA and torch.float8_e4m3fn")
        benchmark_precision(shapes, precision="fp8")


def benchmark_precision(shapes, precision):
    fp8_inputs = precision == "fp8"
    print(f"\n{precision.upper()} square shapes", flush=True)
    print("    M=N=K       1CTA       2CTA     cuBLAS      best shapes", flush=True)

    for size in shapes:
        M = N = K = int(size)
        device = str(DEVICE)
        a = torch.empty((M, K), device=device, dtype=torch.float16).normal_()
        b = torch.empty((K, N), device=device, dtype=torch.float16).normal_()
        if fp8_inputs:
            a = a.to(torch.float8_e4m3fn)
            b = b.to(torch.float8_e4m3fn)
        c_triton = torch.empty((M, N), device=str(DEVICE), dtype=torch.float16)

        matmul(a, b, num_ctas=1, out=c_triton)
        ms_1cta = bench(lambda: matmul(a, b, num_ctas=1, out=c_triton))
        cfg_1cta = matmul_kernel_1cta.best_config

        compiled_2cta = matmul(a, b, num_ctas=2, out=c_triton)
        if "two_ctas" not in compiled_2cta.asm["ttgir"] or "cta_group::2" not in compiled_2cta.asm["ptx"]:
            raise RuntimeError("2CTA autotune selected a kernel without MMAv5 cta_group::2")
        ms_2cta = bench(lambda: matmul(a, b, num_ctas=2, out=c_triton))
        cfg_2cta = matmul_kernel_2cta.best_config

        prefix = f"{size:>9} {tflops(ms_1cta, M, N, K):>10.2f} {tflops(ms_2cta, M, N, K):>10.2f}"
        shape_text = f"{fmt_config(cfg_1cta)} / {fmt_config(cfg_2cta)}"
        b_trans = b.T.contiguous()
        if fp8_inputs:
            c_ref = torch.empty((M, N), device=str(DEVICE), dtype=torch.float8_e4m3fn)
        else:
            c_ref = torch.empty_like(c_triton)
        if cublas is not None:
            ms_cublas = bench(lambda: cublas.matmul(a, b_trans, c_ref))
        else:
            ms_cublas = bench(lambda: torch.matmul(a, b, out=c_ref))
        print(f"{prefix} {tflops(ms_cublas, M, N, K):>10.2f}      {shape_text}", flush=True)


if __name__ == "__main__":
    validate_and_inspect()
    benchmark()
