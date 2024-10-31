#!/usr/bin/env python

# -*- coding: utf-8 -*-

# Imports:
# --------

import argparse
import itertools
import os
import sys
from typing import Any, Callable, Optional

import pytest
import torch
from torch import Tensor

import triton
import triton.language as tl

# Input generation:
# -----------------


def gen_input(M: int, N: int, K: int, use_bias: bool, device: str = "cuda") -> tuple[Tensor, Tensor, Optional[Tensor]]:
    assert M > 0, "M for input generation must be positive."
    assert M <= 8, "M for input generation must be less or equal to 8."
    assert N > 0, "N for input generation must be positive."
    assert K > 0, "K for input generation must be positive."

    torch.manual_seed(42)

    a: Tensor = torch.randn((M, K), dtype=torch.float16, device=device)
    b: Tensor = torch.randn((N, K), dtype=a.dtype, device=a.device).T
    bias: Optional[Tensor] = torch.randn(M, dtype=a.dtype, device=a.device) if use_bias else None

    return a, b, bias


# PyTorch GEMM:
# -------------


def torch_matmul(a: Tensor, b: Tensor, bias: Optional[Tensor]) -> Tensor:
    c: Tensor = torch.matmul(a, b)
    if bias is not None:
        c += bias[:, None]
    return c


# Triton GEMM:
# ------------


# Autotune configurations for Triton GEMM implemented with `tl.dot`.
def get_triton_dot_autotune_configs() -> list[triton.Config]:
    block_size_n_range: list[int] = [16, 32]
    block_size_k_range: list[int] = [128, 256, 512]
    kpack_range: list[int] = [1, 2]
    num_warps_range: list[int] = [1, 2]
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k, "waves_per_eu": 0,
                "matrix_instr_nonkdim": 16, "kpack": kpack
            }, num_warps=num_warps, num_stages=2) for block_size_n, block_size_k, kpack, num_warps in itertools.product(
                block_size_n_range, block_size_k_range, kpack_range, num_warps_range)
    ]


# Autotune configurations for Triton GEMM implemented with explicit dot product.
def get_triton_multreduce_autotune_configs() -> list[triton.Config]:
    block_size_k_range: list[int] = [128, 256, 512]
    kpack_range: list[int] = [1, 2]
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": block_size_k, "waves_per_eu": 0, "kpack": kpack},
            num_warps=8, num_stages=2) for block_size_k, kpack in itertools.product(block_size_k_range, kpack_range)
    ]


def get_triton_autotune_key() -> list[str]:
    return ["M", "N", "K"]


def get_triton_heuristics() -> dict[str, Callable[[dict[str, Any]], Any]]:
    return {"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0}


# Core Triton GEMM kernel.
@triton.jit
def triton_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #
                         M: int, N: int, K: int,  #
                         stride_am: int, stride_ak: int,  #
                         stride_bk: int, stride_bn: int,  #
                         stride_cm: int, stride_cn: int,  #
                         stride_bias: int,  #
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
                         USE_BIAS: tl.constexpr, USE_DOT: tl.constexpr, EVEN_K: tl.constexpr  #
                         ):
    # Compute program ID:
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Compute A and B base pointers:
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Load BIAS:
    if USE_BIAS:
        bias_ptrs = bias_ptr + offs_am * stride_bias
        bias = tl.load(bias_ptrs, mask=offs_am < M, other=0)

    # Initialize accumulator:
    acc_dtype = tl.float32 if a_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    # GEMM loop:

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            # Unmasked load of A and B:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            # Masked load of A and B:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        # Compute dot product:
        if USE_DOT:
            accumulator += tl.dot(a, b)
        else:
            a = tl.reshape(a, (BLOCK_SIZE_M, BLOCK_SIZE_K, 1)).to(acc_dtype)
            b = tl.reshape(b, (1, BLOCK_SIZE_K, BLOCK_SIZE_N)).to(acc_dtype)
            accumulator += tl.sum(a * b, axis=1)
        # Advance A and B pointers:
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Convert accumulator back to C's type:
    c = accumulator.to(c_ptr.type.element_ty)

    # Add BIAS:
    if USE_BIAS:
        c += bias[:, None]

    # Compute C pointers and store C:
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# Triton GEMM kernel implemented with `tl.dot`.
@triton.autotune(configs=get_triton_dot_autotune_configs(), key=get_triton_autotune_key())
@triton.heuristics(get_triton_heuristics())
@triton.jit
def triton_dot_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #
                             M: int, N: int, K: int,  #
                             stride_am: int, stride_ak: int,  #
                             stride_bk: int, stride_bn: int,  #
                             stride_cm: int, stride_cn: int,  #
                             stride_bias: int,  #
                             BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
                             USE_BIAS: tl.constexpr, EVEN_K: tl.constexpr  #
                             ):
    triton_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #
                         M, N, K,  #
                         stride_am, stride_ak,  #
                         stride_bk, stride_bn,  #
                         stride_cm, stride_cn,  #
                         stride_bias,  #
                         BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
                         USE_BIAS=USE_BIAS, USE_DOT=True, EVEN_K=EVEN_K)


# Triton GEMM kernel implemented with explicit dot product.
@triton.autotune(configs=get_triton_multreduce_autotune_configs(), key=get_triton_autotune_key())
@triton.heuristics(get_triton_heuristics())
@triton.jit
def triton_multreduce_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #
                                    M: int, N: int, K: int,  #
                                    stride_am: int, stride_ak: int,  #
                                    stride_bk: int, stride_bn: int,  #
                                    stride_cm: int, stride_cn: int,  #
                                    stride_bias: int,  #
                                    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                    BLOCK_SIZE_K: tl.constexpr,  #
                                    USE_BIAS: tl.constexpr, EVEN_K: tl.constexpr  #
                                    ):
    triton_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #
                         M, N, K,  #
                         stride_am, stride_ak,  #
                         stride_bk, stride_bn,  #
                         stride_cm, stride_cn,  #
                         stride_bias,  #
                         BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
                         USE_BIAS=USE_BIAS, USE_DOT=False, EVEN_K=EVEN_K)


def triton_matmul(triton_provider: str, a: Tensor, b: Tensor, bias: Optional[Tensor]) -> Tensor:
    assert triton_provider in ["triton-dot", "triton-multreduce"]

    M: int
    N: int
    K: int
    M, K = a.shape
    _, N = b.shape

    c: Tensor = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(args: dict[str, Any]) -> tuple[int]:
        return (triton.cdiv(M, args["BLOCK_SIZE_M"]) * triton.cdiv(N, args["BLOCK_SIZE_N"]), )

    matmult_kernel = triton_dot_matmul_kernel if triton_provider == "triton-dot" else triton_multreduce_matmul_kernel

    matmult_kernel[grid](
        # Data pointers
        a,
        b,
        c,
        bias,
        # Size of matrices
        M,
        N,
        K,
        # Strides
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        bias.stride(0) if bias is not None else 0,
        # Other kernel parameters
        USE_BIAS=bias is not None,
    )

    return c


# Wrapper for calling PyTorch GEMM or Triton GEMM:
# ------------------------------------------------


def matmul(provider: str, a: Tensor, b: Tensor, bias: Optional[Tensor]) -> Tensor:
    assert provider in ["torch", "triton-dot", "triton-multreduce"]

    assert a.is_cuda, "Matrix A must be in GPU."
    assert a.is_contiguous(), "Matrix A must be contiguous."
    assert b.is_cuda, "Matrix B must be in GPU."
    assert a.device == b.device, "Matrix A and matrix B must be in the same GPU."
    assert a.dtype == b.dtype, "Matrix A and matrix B must have the same data type."
    assert a.dim() == b.dim() == 2, "Matrix A and matrix B must be two-dimensional tensors."
    assert a.shape[1] == b.shape[0], "Matrix A columns must be equal to matrix B rows."

    if bias is not None:
        assert bias.is_cuda, "Bias vector must be in GPU."
        assert bias.is_contiguous(), "Bias vector must be continuous."
        assert bias.device == a.device, "Matrix A and bias vector must be in the same GPU."
        assert bias.dtype == a.dtype, "Matrix A and bias vector must have the same data type."
        assert bias.dim() == 1, "Bias vector must be one-dimensional tensor."
        assert bias.shape == (a.shape[0], ), "Bias vector length must be equal to matrix A rows."

    if provider == "torch":
        return torch_matmul(a, b, bias)

    return triton_matmul(provider, a, b, bias)


# Run Triton GEMM:
# ----------------
# This is useful to run the kernel in isolation, in order to get performance traces for instance.


def run_triton_matmul(M: int, N: int, K: int, use_bias: bool, use_dot: bool) -> Tensor:
    a: Tensor
    b: Tensor
    bias: Optional[Tensor]
    a, b, bias = gen_input(M, N, K, use_bias)
    triton_provider: str = "triton-dot" if use_dot else "triton-multreduce"
    c: Tensor = matmul(triton_provider, a, b, bias)
    return c


# Test Triton GEMM, comparing it to PyTorch GEMM reference implementation:
# ------------------------------------------------------------------------
# It's a pytest suite, you can run it with `pytest multreduce_matmul_kernel.py`.
# You can also run a single test with
# `multreduce_matmul_kernel.py::test_matmul[M-N-K-use_bias]`.


def get_target_shapes() -> list[tuple[int, int, int]]:
    # yapf: disable
    return [
        (1, 8192, 28672),   # Llama 70B
        (1, 6144, 6144),    # Grok
        (1, 4096, 4096),    # Generic GEMM
        (2, 16384, 16384),  # Generic GEMM
        (1, 4096, 3078),    # Uneven K
        (1, 23, 31),        # Very small shape, uneven K
        (1, 23, 128),       # Very small shape, even K
    ]
    # yapf: enable


def allclose(x: Tensor, y: Tensor) -> bool:
    return torch.allclose(x, y, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("M, N, K", get_target_shapes())
def test_matmul(M: int, N: int, K: int, use_bias: bool) -> None:
    a: Tensor
    b: Tensor
    bias: Optional[Tensor]
    a, b, bias = gen_input(M, N, K, use_bias)

    c_torch: Tensor = matmul("torch", a, b, bias)
    c_triton_dot: Tensor = matmul("triton-dot", a, b, bias)
    c_triton_multreduce: Tensor = matmul("triton-multreduce", a, b, bias)

    assert allclose(c_torch, c_triton_dot), "PyTorch and Triton Dot results don't match."
    assert allclose(c_torch, c_triton_multreduce), "PyTorch and Triton Multreduce results don't match."


# Benchmark Triton GEMM, comparing it to PyTorch GEMM reference implementation:
# -----------------------------------------------------------------------------


# Convert milliseconds to GiB/s.
def ms_to_gibps(M: int, N: int, K: int, milliseconds: float) -> float:
    read_elems: int = M * K + K * N
    write_elems: int = M * N
    transf_elems: int = read_elems + write_elems
    transf_bytes: int = 2 * transf_elems  # times 2 due to fp16
    transf_gibibytes: float = 2**-30 * transf_bytes
    seconds: float = 1e-3 * milliseconds
    return round(transf_gibibytes / seconds, 2)


def run_benchmark(use_bias: bool) -> None:
    perf_unit: str = "GiB/s"
    line_vals: list[str] = ["torch", "triton-dot", "triton-multreduce"]
    line_names: list[str] = [f"{x.replace('-', ' ').title()} ({perf_unit})" for x in line_vals]

    # Triton benchmark:
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=get_target_shapes(),
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            ylabel=perf_unit,
            args={},
            plot_name=f"fp16_{os.path.splitext(os.path.basename(__file__))[0]}",
        ))
    def benchmark(M: int, N: int, K: int, provider: str) -> tuple[float, float, float]:

        def perf(milliseconds: float) -> float:
            return ms_to_gibps(M, N, K, milliseconds)

        a: Tensor
        b: Tensor
        bias: Optional[Tensor]
        a, b, bias = gen_input(M, N, K, use_bias)

        p20_ms: float
        p50_ms: float
        p80_ms: float
        p20_ms, p50_ms, p80_ms = triton.testing.do_bench(lambda: matmul(provider, a, b, bias),
                                                         quantiles=[0.2, 0.5, 0.8])

        p20_gibps: float = perf(p80_ms)
        p50_gibps: float = perf(p50_ms)
        p80_gibps: float = perf(p20_ms)

        print(", ".join([
            f"(M, N, K) = {(M, N, K)}",
            f"provider = {provider}",
            f"p20 = {p20_gibps} {perf_unit}",
            f"p50 = {p50_gibps} {perf_unit}",
            f"p80 = {p80_gibps} {perf_unit}",
        ]))

        if provider == "triton-dot":
            print(f"Triton Dot kernel best config = {triton_dot_matmul_kernel.best_config}")
        elif provider == "triton-multreduce":
            print(f"Triton Multreduce kernel best config = {triton_multreduce_matmul_kernel.best_config}")

        return p50_gibps, p20_gibps, p80_gibps

    print(f"Running benchmark (use_bias = {use_bias})...")
    benchmark.run(show_plots=False, print_data=True)
    print("Done.")


# Script entry point:
# -------------------


def positive_int(value: str) -> int:
    try:
        int_value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not an integer.")
    if int_value <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer.")
    return int_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C = A * B + BIAS matrix multiplication kernel for small matrices (M ≤ 8)",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "mode", choices=["run", "bench"], help="mode of operation:\n"
        "  run: run Triton kernel for a given (M, N, K) shape\n"
        "  bench: benchmark performance for target shapes\n")
    shape_group = parser.add_argument_group("kernel shape arguments")
    shape_group.add_argument("-M", type=positive_int, help="rows of matrix A (must be less or equal to 8)")
    shape_group.add_argument("-N", type=positive_int, help="columns of matrix A / rows of matrix B")
    shape_group.add_argument("-K", type=positive_int, help="columns of matrix B")
    shape_group.add_argument("--use-bias", default=False, action="store_true", help="use BIAS vector")
    shape_group.add_argument("--use-dot", default=False, action="store_true", help="use tl.dot for dot product")
    args = parser.parse_args()
    if args.mode == "run":
        try:
            sizes: tuple[Optional[int], ...] = tuple(size for size in (args.M, args.N, args.K))
            if any(size is None for size in sizes):
                raise ValueError(f"(M, N, K) = {sizes}, all sizes must be specified together.")
            if args.M > 8:
                raise ValueError(f"M = {args.M} is too big, this kernel was designed for M ≤ 8.")
        except ValueError as arg_error:
            print(arg_error)
            sys.exit(1)
    return args


def main() -> int:
    args: argparse.Namespace = parse_args()
    status: int = 0
    try:
        match args.mode:
            case "run":
                run_triton_matmul(args.M, args.N, args.K, args.use_bias, args.use_dot)
            case "bench":
                run_benchmark(args.use_bias)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as error:
        print(f"\nUnexpected error: {error}")
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
