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


def is_fp8_dtype(dtype):
    return TORCH_HAS_FP8 and dtype == torch.float8_e4m3fn

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


@triton.autotune(configs=MATMUL_CONFIGS, key=["M", "N", "K", "FP8_INPUTS"], do_bench=proton_autotune_do_bench)
@triton.jit
def matmul_kernel_1cta(
        a_desc, b_desc, c_desc,  #
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr, NUM_STAGES: tl.constexpr, FP8_INPUTS: tl.constexpr):
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
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=False):
        off_k = k * BLOCK_K
        a = a_desc.load([off_m, off_k])
        b = b_desc.load([off_k, off_n])
        acc = tl.dot(a, b, acc)

    c_desc.store([off_m, off_n], acc.to(tl.float16))


@triton.autotune(configs=TWO_CTA_CONFIGS, key=["M", "N", "K", "FP8_INPUTS"], do_bench=proton_autotune_do_bench)
@triton.jit
def matmul_kernel_2cta(
        a_desc, b_desc, c_desc,  #
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr, NUM_STAGES: tl.constexpr, FP8_INPUTS: tl.constexpr):
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
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=False):
        off_k = k * BLOCK_K
        a = a_desc.load([off_m, off_k])
        b = b_desc.load([off_k, off_n])
        acc = tl.dot(a, b, acc)

    c_desc.store([off_m, off_n], acc.to(tl.float16))


@triton.autotune(configs=WS_CONFIGS, key=["M", "N", "K", "FP8_INPUTS"], do_bench=proton_autotune_do_bench)
@triton.jit
def matmul_kernel_2cta_ws(
        a_desc, b_desc, c_desc,  #
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr, NUM_STAGES: tl.constexpr, FP8_INPUTS: tl.constexpr):
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
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=True):
        off_k = k * BLOCK_K
        a = a_desc.load([off_m, off_k])
        b = b_desc.load([off_k, off_n])
        acc = tl.dot(a, b, acc)

    c_desc.store([off_m, off_n], acc.to(tl.float16))


def get_matmul_kernel(num_ctas, warp_specialize):
    if warp_specialize:
        return matmul_kernel_2cta_ws
    if num_ctas == 2:
        return matmul_kernel_2cta
    return matmul_kernel_1cta


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
    kernel = get_matmul_kernel(num_ctas, warp_specialize)
    return kernel[grid](a_desc, b_desc, c_desc, M, N, K, FP8_INPUTS=is_fp8_dtype(a.dtype))


def selected_config(num_ctas, warp_specialize=False):
    return get_matmul_kernel(num_ctas, warp_specialize).best_config


def cublas_matmul(a, b, out):
    if cublas is not None:
        # CublasLt's helper expects B in transposed contiguous form.
        return cublas.matmul(a, b.T.contiguous(), out)
    return torch.matmul(a, b, out=out)


def bench_cublas_matmul(a, b, b_trans, out):
    if cublas is not None:
        return cublas.matmul(a, b_trans, out)
    return torch.matmul(a, b, out=out)


def tflops(ms, M, N, K):
    return 2.0 * M * N * K * 1e-12 / (ms * 1e-3)


def bench(fn):
    return triton.testing.do_bench_proton(fn, warmup=1, rep=5)


BENCHMARK_SHAPES = list(range(1024, 4097, 256))


def fmt_config(config):
    return (f"{config.kwargs['BLOCK_M']}x{config.kwargs['BLOCK_N']}x{config.kwargs['BLOCK_K']}"
            f"s{config.num_stages}w{config.num_warps}")


def uses_mmav5_two_ctas(compiled):
    return "two_ctas" in compiled.asm["ttgir"] and "cta_group::2" in compiled.asm["ptx"]


def make_inputs(M, N, K, fp8_inputs):
    M, N, K = int(M), int(N), int(K)
    device = str(DEVICE)
    a = torch.empty((M, K), device=device, dtype=torch.float16).normal_()
    b = torch.empty((K, N), device=device, dtype=torch.float16).normal_()
    if fp8_inputs:
        a = a.to(torch.float8_e4m3fn)
        b = b.to(torch.float8_e4m3fn)
    return a, b


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


def benchmark(shapes=BENCHMARK_SHAPES, include_fp8=False):
    if not is_blackwell():
        raise RuntimeError("This tutorial requires an NVIDIA Blackwell GPU.")

    print("Benchmarking Triton multi-CTA matmul", flush=True)
    print("====================================", flush=True)
    benchmark_precision(shapes, fp8_inputs=False)
    if include_fp8 and TORCH_HAS_FP8 and is_cuda():
        benchmark_precision(shapes, fp8_inputs=True)


def benchmark_precision(shapes, fp8_inputs):
    precision = "fp8" if fp8_inputs else "fp16"
    print(f"\n{precision.upper()} square shapes", flush=True)
    if fp8_inputs:
        print("    M=N=K       1CTA       2CTA    2CTA+WS     cuBLAS      best shapes", flush=True)
    else:
        print("    M=N=K       1CTA       2CTA    2CTA+WS     cuBLAS      best shapes", flush=True)

    for size in shapes:
        M = N = K = int(size)
        a, b = make_inputs(M, N, K, fp8_inputs)
        c_triton = torch.empty((M, N), device=str(DEVICE), dtype=torch.float16)

        matmul(a, b, num_ctas=1, out=c_triton)
        ms_1cta = bench(lambda: matmul(a, b, num_ctas=1, out=c_triton))
        cfg_1cta = selected_config(num_ctas=1)

        compiled_2cta = matmul(a, b, num_ctas=2, out=c_triton)
        if not uses_mmav5_two_ctas(compiled_2cta):
            raise RuntimeError("2CTA autotune selected a kernel without MMAv5 cta_group::2")
        ms_2cta = bench(lambda: matmul(a, b, num_ctas=2, out=c_triton))
        cfg_2cta = selected_config(num_ctas=2)

        matmul(a, b, num_ctas=2, warp_specialize=True, out=c_triton)
        ms_2cta_ws = bench(lambda: matmul(a, b, num_ctas=2, warp_specialize=True, out=c_triton))
        cfg_2cta_ws = selected_config(num_ctas=2, warp_specialize=True)

        prefix = (f"{size:>9} {tflops(ms_1cta, M, N, K):>10.2f} "
                  f"{tflops(ms_2cta, M, N, K):>10.2f} {tflops(ms_2cta_ws, M, N, K):>10.2f}")
        shape_text = f"{fmt_config(cfg_1cta)} / {fmt_config(cfg_2cta)} / {fmt_config(cfg_2cta_ws)}"
        if fp8_inputs:
            b_trans = b.T.contiguous()
            c_ref = torch.empty((M, N), device=str(DEVICE), dtype=torch.float8_e4m3fn)
            ms_cublas = bench(lambda: bench_cublas_matmul(a, b, b_trans, c_ref))
            print(f"{prefix} {tflops(ms_cublas, M, N, K):>10.2f}      {shape_text}", flush=True)
        else:
            b_trans = b.T.contiguous()
            c_ref = torch.empty_like(c_triton)
            ms_cublas = bench(lambda: bench_cublas_matmul(a, b, b_trans, c_ref))
            print(f"{prefix} {tflops(ms_cublas, M, N, K):>10.2f}      {shape_text}", flush=True)


if __name__ == "__main__":
    validate_and_inspect()
    benchmark()
