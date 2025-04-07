"""
Programmatic Dependent Launch
=====================
This script demonstrates the use of programmatic dependent launch ontop of a non-persistent implementation of matrix multiplication using Triton.
The kernels support both FP16 and FP8 data types but the FP8 implementation is only available on CUDA devices with compute capability >= 9.0.

.. code-block:: bash
    python 11-programmatic-dependent-launch.py
"""

import torch
import triton
import triton.language as tl

from typing import Optional


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    elif "c_desc_ptr" in args:
        bytes_per_elem = 2
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


def matmul_get_configs():
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "USE_TMA" : USE_TMA}, num_stages=s, num_warps=w) \
        for BM in [128] \
        for BN in [128, 256] \
        for BK in [64, 128] \
        for USE_TMA in [True, False] \
        for s in ([3, 4]) \
        for w in [4, 8] \
    ]


@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel(a_ptr, b_ptr, c_ptr,  #
                  M, N, K,  #
                  BLOCK_SIZE_M: tl.constexpr,  #
                  BLOCK_SIZE_N: tl.constexpr,  #
                  BLOCK_SIZE_K: tl.constexpr,  #
                  USE_TMA: tl.constexpr,  #
                  USE_GDC: tl.constexpr):  #
    # This kernel can use TMA or traditional tl.load/tl.store
    a_desc = None
    b_desc = None
    c_desc = None
    a_ptrs = None
    b_ptrs = None
    c_ptrs = None

    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    pid_m = start_pid % num_pid_m
    pid_n = start_pid // num_pid_m

    if USE_TMA:
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
    else:
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
        # B is N x K
        b_ptrs = b_ptr + (offs_k[None, :] + offs_bn[:, None] * K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if USE_GDC:
        # GDC wait is used to wait for the prior kernel to complete before continuing.
        # If we utilize Programmatic Dependent Launch, we must wait on the prior kernel
        # to complete in case a or b are written to by the prior kernel.
        # This is done to prevent races.
        tl.extra.cuda.gdc_wait()

    if USE_TMA:
        # We have no guarentees about what memory may be passed for our make tensor descriptor,
        # and whether there is inter kernel dependencies. We therefore must safely allocate after
        # Waiting on the prior kernel to finish.
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[N, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )
        c_desc = tl.make_tensor_descriptor(
            c_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

    for ki in range(k_tiles):
        a = None
        b = None
        if USE_TMA:
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K
        accumulator = tl.dot(a, b.T, accumulator)

    if USE_GDC:
        # GDC launch dependents is used to launch dependent kernels.
        # Once GDC launch it is possible for the next kernel to begin if
        # there are enough resources.
        tl.extra.cuda.gdc_launch_dependents()

    c = accumulator.to(dtype)
    if USE_TMA:
        c_desc.store([offs_am, offs_bn], c)
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, launch_pdl=True):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        USE_GDC=launch_pdl,  # set constexpr in kernel to use grid dependence control
        launch_pdl=launch_pdl,  # launch kernel with PDL flag set enabled
    )
    return c


def validate(M, N, K, dtype):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.T.contiguous()

    torch_result = torch.matmul(a, b.T)
    matmul_result = matmul(a, b)

    torch_vs_matmul = "✅" if torch.allclose(torch_result.to(torch.float16), matmul_result.to(torch.float16),
                                            atol=1.0) else "❌"
    print(f"M={M}, N={N}, K={K} verification naive vs: ", end="")
    print(f"matmul: {torch_vs_matmul} ", end="")
    print()


TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["K"],
        x_vals=range(512, 4096 + 1, 512),
        x_log=False,
        line_arg="provider",
        line_vals=["pdl-fp16", "fp16"] + (["pdl-fp8", "fp8"] if TORCH_HAS_FP8 else []),
        line_names=["PDL [FP16]", "No PDL [FP16]"] + (["PDL [FP8]", "No PDL [FP8]"] if TORCH_HAS_FP8 else []),
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-")],
        ylabel="TFLOPS",
        plot_name="pdl-performance",
        args={},
    ))
def benchmark(K, provider, M, N):

    tflops = (lambda ms: (2.0 * M * N * K) * 1e-12 / (ms * 1e-3))

    quantiles = [0.5, 0.2, 0.8]
    if "fp8" in provider:
        a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(torch.float8_e5m2)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(torch.float8_e5m2)
        b = b.T.contiguous()
        if "pdl" in provider:
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                lambda: matmul(a, b),
                quantiles=quantiles,
                rep=1000,
            )
            return tflops(ms), tflops(max_ms), tflops(min_ms)
        else:
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                lambda: matmul(a, b, False),
                quantiles=quantiles,
                rep=1000,
            )
            return tflops(ms), tflops(max_ms), tflops(min_ms)
    else:
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        b = b.T.contiguous()
        if "pdl" in provider:
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                lambda: matmul(a, b),
                quantiles=quantiles,
                rep=1000,
            )
            return tflops(ms), tflops(max_ms), tflops(min_ms)
        else:
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                lambda: matmul(a, b, False),
                quantiles=quantiles,
                rep=1000,
            )
            return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    if supports_tma():
        validate(32, 32, 32, torch.float16)
        benchmark.run(print_data=True, save_path=".", M=128, N=8192)
    else:
        print("TMA not supported on this device")
