"""
Persistent Matmul
=====================
This script demonstrates persistent kernel implementations of matrix multiplication using Triton.
Various matmul methods are included, such as naive, persistent, and TMA (Tensor Memory Accelerator) based approaches.
The kernels support both FP16 and FP8 data types but the FP8 implementation is only available on CUDA devices with compute capability >= 9.0.

Triton and cuBLAS implementations are benchmarked under different configurations and evaluated using the proton profiler.
Users can pass command-line arguments to specify matrix dimensions and iteration steps flexibly.

.. code-block:: bash

    # FP8
    python 09-persistent-matmul.py --prec fp8 --K_range 128 1024 --K_step 128

    # FP16
    python 09-persistent-matmul.py --prec fp16 --K_range 128 1024 --K_step 128

Note that currently this tutorial will fail on devices with a small shared memory size, such as RTX-4090.
"""

import argparse

import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
import triton.profiler as proton
import pathlib
from contextlib import contextmanager

from typing import Optional

if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


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
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel(a_ptr, b_ptr, c_ptr,  #
                  M, N, K,  #
                  stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  BLOCK_SIZE_M: tl.constexpr,  #
                  BLOCK_SIZE_N: tl.constexpr,  #
                  BLOCK_SIZE_K: tl.constexpr,  #
                  GROUP_SIZE_M: tl.constexpr,  #
                  ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    configs = {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_stages": 4,
            "num_warps": 8
        }, torch.float16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3,
            "num_warps": 8
        }
    }
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
        num_stages=configs[dtype]["num_stages"],  #
        num_warps=configs[dtype]["num_warps"],  #
    )
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent_fused(a_ptr, b_ptr, c_ptr,  #
                                   M, N, K,  #
                                   stride_am, stride_ak,  #
                                   stride_bk, stride_bn,  #
                                   stride_cm, stride_cn,  #
                                   BLOCK_SIZE_M: tl.constexpr,  #
                                   BLOCK_SIZE_N: tl.constexpr,  #
                                   BLOCK_SIZE_K: tl.constexpr,  #
                                   GROUP_SIZE_M: tl.constexpr,  #
                                   NUM_SMS: tl.constexpr,  #
                                   ):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    pid_m = 0
    pid_n = 0
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCK_SIZE_M
            start_n = pid_n * BLOCK_SIZE_N
            offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        if ki == k_tiles - 1:
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            if (c_ptr.dtype.element_ty == tl.float8e4nv):
                c = accumulator.to(tl.float8e4nv)
            else:
                c = accumulator.to(tl.float16)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_persistent_fused(a, b):
    configs = {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_stages": 4,
            "num_warps": 8
        }, torch.float16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3,
            "num_warps": 8
        }
    }
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )

    matmul_kernel_persistent_fused[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
        NUM_SMS=NUM_SMS,  #
        num_stages=configs[dtype]["num_stages"],  #
        num_warps=configs[dtype]["num_warps"],  #
    )
    #kernel = matmul_kernel_persistent_fused.warmup(
    #    a, b, c,  #
    #    M, N, K,  #
    #    a.stride(0), a.stride(1),  #
    #    b.stride(0), b.stride(1),  #
    #    c.stride(0), c.stride(1),  #
    #    BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
    #    BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
    #    BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
    #    GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
    #    NUM_SMS=NUM_SMS,  #
    #    num_stages=configs[dtype]["num_stages"],  #
    #    num_warps=configs[dtype]["num_warps"],  #
    #    grid=grid
    #)
    #print(kernel.asm["ttgir"])
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(a_ptr, b_ptr, c_ptr,  #
                             M, N, K,  #
                             stride_am, stride_ak,  #
                             stride_bk, stride_bn,  #
                             stride_cm, stride_cn,  #
                             BLOCK_SIZE_M: tl.constexpr,  #
                             BLOCK_SIZE_N: tl.constexpr,  #
                             BLOCK_SIZE_K: tl.constexpr,  #
                             GROUP_SIZE_M: tl.constexpr,  #
                             NUM_SMS: tl.constexpr,  #
                             ):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if (c_ptr.dtype.element_ty == tl.float8e4nv):
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


matmul_kernel_persistent_ttgir = """
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_0 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c132_i32 = arith.constant 132 : i32
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %6 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = arith.addi %arg3, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.addi %arg4, %c255_i32 : i32
    %10 = arith.divsi %9, %c256_i32 : i32
    %11 = arith.addi %arg5, %c63_i32 : i32
    %12 = arith.divsi %11, %c64_i32 : i32
    %13 = arith.muli %8, %10 : i32
    %14 = arith.muli %10, %c8_i32 : i32
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %17 = arith.subi %13, %0 : i32
    %18 = arith.ceildivsi %17, %c132_i32 : i32
    %19 = arith.maxsi %12, %c1_i32 : i32
    %20 = arith.muli %18, %19 : i32
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %22 = ttg.local_alloc  : () -> !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable>
    %23 = arith.cmpi sgt, %20, %c0_i32 : i32
    %24 = arith.divsi %0, %14 : i32
    %25 = arith.muli %24, %c8_i32 : i32
    %26 = arith.subi %8, %25 : i32
    %27 = arith.minsi %26, %c8_i32 : i32
    %28 = arith.remsi %0, %27 : i32
    %29 = arith.addi %25, %28 : i32
    %30 = arith.remsi %0, %14 : i32
    %31 = arith.divsi %30, %27 : i32
    %32 = arith.muli %29, %c128_i32 : i32
    %33 = arith.muli %31, %c256_i32 : i32
    %34 = tt.splat %32 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %35 = arith.addi %34, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %36 = tt.splat %33 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %37 = arith.addi %36, %4 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %38 = arith.cmpi slt, %35, %5 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %39 = arith.select %38, %35, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %40 = arith.cmpi slt, %37, %6 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %41 = arith.select %40, %37, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %42 = tt.expand_dims %39 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %43 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %44 = arith.muli %42, %43 : tensor<128x1xi32, #blocked1>
    %45 = tt.expand_dims %15 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %46 = tt.broadcast %44 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %47 = tt.broadcast %45 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %48 = arith.addi %46, %47 : tensor<128x64xi32, #blocked1>
    %49 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %50 = tt.addptr %49, %48 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %51 = tt.expand_dims %16 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %52 = tt.expand_dims %41 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %53 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked>
    %54 = arith.muli %52, %53 : tensor<1x256xi32, #blocked>
    %55 = tt.broadcast %51 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %56 = tt.broadcast %54 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %57 = arith.addi %55, %56 : tensor<64x256xi32, #blocked>
    %58 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked>
    %59 = tt.addptr %58, %57 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %60 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1>
    %61 = arith.cmpi slt, %45, %60 : tensor<1x64xi32, #blocked1>
    %62 = tt.broadcast %61 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %63 = ttg.memdesc_subview %21[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %64 = tt.splat %23 : i1 -> tensor<128x64xi1, #blocked1>
    %65 = arith.andi %64, %62 : tensor<128x64xi1, #blocked1>
    %66 = ttg.async_copy_global_to_local %50, %63 mask %65 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %67 = ttg.async_commit_group %66
    %68 = tt.splat %arg5 : i32 -> tensor<64x1xi32, #blocked>
    %69 = arith.cmpi slt, %51, %68 : tensor<64x1xi32, #blocked>
    %70 = tt.broadcast %69 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %71 = ttg.memdesc_subview %22[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    %72 = tt.splat %23 : i1 -> tensor<64x256xi1, #blocked>
    %73 = arith.andi %72, %70 : tensor<64x256xi1, #blocked>
    %74 = ttg.async_copy_global_to_local %59, %71 mask %73 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
    %75 = ttg.async_commit_group %74
    %76 = arith.cmpi sgt, %20, %c1_i32 : i32
    %77 = arith.remsi %c1_i32, %19 : i32
    %78 = arith.cmpi eq, %77, %c0_i32 : i32
    %79 = arith.cmpi ne, %77, %c0_i32 : i32
    %80 = arith.extui %79 : i1 to i32
    %81:5 = scf.if %78 -> (i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32) {
      %121 = arith.addi %0, %c132_i32 : i32
      %122 = arith.divsi %121, %14 : i32
      %123 = arith.muli %122, %c8_i32 : i32
      %124 = arith.subi %8, %123 : i32
      %125 = arith.minsi %124, %c8_i32 : i32
      %126 = arith.remsi %121, %125 : i32
      %127 = arith.addi %123, %126 : i32
      %128 = arith.remsi %121, %14 : i32
      %129 = arith.divsi %128, %125 : i32
      %130 = arith.muli %127, %c128_i32 : i32
      %131 = arith.muli %129, %c256_i32 : i32
      %132 = tt.splat %130 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %133 = arith.addi %132, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %134 = tt.splat %131 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %135 = arith.addi %134, %4 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %136 = arith.cmpi slt, %133, %5 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %137 = arith.select %136, %133, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %138 = arith.cmpi slt, %135, %6 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %139 = arith.select %138, %135, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      scf.yield %130, %131, %137, %139, %121 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32
    } else {
      scf.yield %32, %33, %39, %41, %0 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32
    }
    %82 = arith.muli %80, %c64_i32 : i32
    %83 = tt.splat %82 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %84 = tt.splat %82 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %85 = arith.addi %83, %15 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %86 = arith.addi %84, %16 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %87 = tt.expand_dims %81#2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %88 = arith.muli %87, %43 : tensor<128x1xi32, #blocked1>
    %89 = tt.expand_dims %85 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %90 = tt.broadcast %88 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %91 = tt.broadcast %89 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %92 = arith.addi %90, %91 : tensor<128x64xi32, #blocked1>
    %93 = tt.addptr %49, %92 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %94 = tt.expand_dims %86 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %95 = tt.expand_dims %81#3 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %96 = arith.muli %95, %53 : tensor<1x256xi32, #blocked>
    %97 = tt.broadcast %94 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %98 = tt.broadcast %96 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %99 = arith.addi %97, %98 : tensor<64x256xi32, #blocked>
    %100 = tt.addptr %58, %99 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %101 = arith.subi %arg5, %82 : i32
    %102 = tt.splat %101 : i32 -> tensor<1x64xi32, #blocked1>
    %103 = arith.cmpi slt, %45, %102 : tensor<1x64xi32, #blocked1>
    %104 = tt.broadcast %103 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %105 = ttg.memdesc_subview %21[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %106 = tt.splat %76 : i1 -> tensor<128x64xi1, #blocked1>
    %107 = arith.andi %106, %104 : tensor<128x64xi1, #blocked1>
    %108 = ttg.async_copy_global_to_local %93, %105 mask %107 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %109 = ttg.async_commit_group %108
    %110 = tt.splat %101 : i32 -> tensor<64x1xi32, #blocked>
    %111 = arith.cmpi slt, %51, %110 : tensor<64x1xi32, #blocked>
    %112 = tt.broadcast %111 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %113 = ttg.memdesc_subview %22[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    %114 = tt.splat %76 : i1 -> tensor<64x256xi1, #blocked>
    %115 = arith.andi %114, %112 : tensor<64x256xi1, #blocked>
    %116 = ttg.async_copy_global_to_local %100, %113 mask %115 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
    %117 = ttg.async_commit_group %116
    %lol = arith.subi %12, %c1_i32 : i32
    %118:16 = scf.for %arg9 = %c0_i32 to %20 step %c1_i32 iter_args(
      %arg10 = %81#4, %arg11 = %cst_3, %arg12 = %81#0, %arg13 = %81#1,
      %arg14 = %81#2, %arg15 = %81#3, %arg16 = %c1_i32, %arg17 = %c-1_i32,
      %arg18 = %80, %arg19 = %c0_i32, %arg21 = %75, %arg22 = %117, %arg23 = %32, %arg24 = %81#0, %arg25 = %33, %arg26 = %81#1) -> (i32, tensor<128x256xf32, #mma>, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, i32, i32, i32, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32)  : i32 {
      %121 = arith.subi %20, %c2_i32 : i32
      %122 = arith.cmpi slt, %arg9, %121 : i32
      %rollover = arith.cmpi eq, %arg18, %lol : i32
      %123 = arith.addi %arg18, %c1_i32 : i32
      %126 = arith.select %rollover, %c0_i32, %123 : i32
      %125 = arith.cmpi eq, %126, %c0_i32 : i32
      %127:5 = scf.if %125 -> (i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32) {
        %178 = arith.addi %arg10, %c132_i32 : i32
        %179 = arith.divsi %178, %14 : i32
        %180 = arith.muli %179, %c8_i32 : i32
        %181 = arith.subi %8, %180 : i32
        %182 = arith.minsi %181, %c8_i32 : i32
        %183 = arith.remsi %178, %182 : i32
        %184 = arith.addi %180, %183 : i32
        %185 = arith.remsi %178, %14 : i32
        %186 = arith.divsi %185, %182 : i32
        %187 = arith.muli %184, %c128_i32 : i32
        %188 = arith.muli %186, %c256_i32 : i32
        %189 = tt.splat %187 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %190 = arith.addi %189, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %191 = tt.splat %188 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %192 = arith.addi %191, %4 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %193 = arith.cmpi slt, %190, %5 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %194 = arith.select %193, %190, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %195 = arith.cmpi slt, %192, %6 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %196 = arith.select %195, %192, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        scf.yield %187, %188, %194, %196, %178 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32
      } else {
        scf.yield %arg12, %arg13, %arg14, %arg15, %arg10 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32
      }
      %128 = arith.addi %arg17, %c1_i32 : i32
      %129 = arith.cmpi slt, %128, %c3_i32 : i32
      %130 = arith.select %129, %128, %c0_i32 : i32
      %131 = arith.cmpi ne, %arg19, %c0_i32 : i32
      %132 = ttg.memdesc_subview %21[%130, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %133 = ttg.async_wait %arg21 {num = 2 : i32}
      %134 = ttg.memdesc_subview %22[%130, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %135 = ttng.warp_group_dot %132, %134, %arg11, %131 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> * !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> tensor<128x256xf32, #mma>
      %136:3 = ttng.warp_group_dot_wait %135, %132, %134 {pendings = 1 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %137 = arith.addi %arg16, %c1_i32 : i32
      %138 = arith.cmpi slt, %137, %c3_i32 : i32
      %139 = arith.select %138, %137, %c0_i32 : i32
      %140 = arith.muli %126, %c64_i32 : i32
      %141 = tt.splat %140 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %142 = tt.splat %140 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %143 = arith.addi %141, %15 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %144 = arith.addi %142, %16 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %145 = tt.expand_dims %127#2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %146 = arith.muli %145, %43 : tensor<128x1xi32, #blocked1>
      %147 = tt.expand_dims %143 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
      %148 = tt.broadcast %146 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %149 = tt.broadcast %147 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %150 = arith.addi %148, %149 : tensor<128x64xi32, #blocked1>
      %151 = tt.addptr %49, %150 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %152 = tt.expand_dims %144 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %153 = tt.expand_dims %127#3 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
      %154 = arith.muli %153, %53 : tensor<1x256xi32, #blocked>
      %155 = tt.broadcast %152 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %156 = tt.broadcast %154 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %157 = arith.addi %155, %156 : tensor<64x256xi32, #blocked>
      %158 = tt.addptr %58, %157 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
      %159 = arith.subi %arg5, %140 : i32
      %160 = tt.splat %159 : i32 -> tensor<1x64xi32, #blocked1>
      %161 = arith.cmpi slt, %45, %160 : tensor<1x64xi32, #blocked1>
      %162 = tt.broadcast %161 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
      %163 = ttg.memdesc_subview %21[%139, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %164 = tt.splat %122 : i1 -> tensor<128x64xi1, #blocked1>
      %165 = arith.andi %164, %162 : tensor<128x64xi1, #blocked1>
      %166 = ttg.async_copy_global_to_local %151, %163 mask %165 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
      %167 = ttg.async_commit_group %166
      %168 = tt.splat %159 : i32 -> tensor<64x1xi32, #blocked>
      %169 = arith.cmpi slt, %51, %168 : tensor<64x1xi32, #blocked>
      %170 = tt.broadcast %169 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
      %171 = ttg.memdesc_subview %22[%139, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %172 = tt.splat %122 : i1 -> tensor<64x256xi1, #blocked>
      %173 = arith.andi %172, %170 : tensor<64x256xi1, #blocked>
      %174 = ttg.async_copy_global_to_local %158, %171 mask %173 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
      %175 = ttg.async_commit_group %174
      %176 = arith.subi %19, %c1_i32 : i32
      %177 = arith.cmpi eq, %arg19, %176 : i32
      scf.if %177 {
        %178:3 = ttng.warp_group_dot_wait %136#0, %132, %134 {pendings = 0 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
        %179 = tt.splat %arg23 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %180 = arith.addi %179, %1 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %181 = tt.splat %arg25 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %182 = arith.addi %181, %2 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %183 = tt.expand_dims %180 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
        %184 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked2>
        %185 = arith.muli %184, %183 : tensor<128x1xi32, #blocked2>
        %186 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
        %187 = tt.addptr %186, %185 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
        %188 = tt.expand_dims %182 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
        %189 = tt.broadcast %187 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x256x!tt.ptr<f16>, #blocked2>
        %190 = tt.broadcast %188 : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
        %191 = tt.addptr %189, %190 : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
        %192 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked2>
        %193 = arith.cmpi slt, %183, %192 : tensor<128x1xi32, #blocked2>
        %194 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked2>
        %195 = arith.cmpi slt, %188, %194 : tensor<1x256xi32, #blocked2>
        %196 = tt.broadcast %193 : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
        %197 = tt.broadcast %195 : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
        %198 = arith.andi %196, %197 : tensor<128x256xi1, #blocked2>
        %199 = arith.truncf %178#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
        %200 = ttg.convert_layout %199 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked2>
        tt.store %191, %200, %198 : tensor<128x256x!tt.ptr<f16>, #blocked2>
      }
      scf.yield %127#4, %136#0, %127#0, %127#1,
                 %127#2, %127#3, %139, %130,
                 %126, %arg18, %arg22, %175, %arg24, %127#0, %arg26, %127#1 : i32, tensor<128x256xf32, #mma>, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, i32, i32, i32, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32
    }
    %119 = ttng.warp_group_dot_wait %118#1 {pendings = 0 : i32} : tensor<128x256xf32, #mma>
    %120 = ttg.async_wait  {num = 0 : i32}
    ttg.local_dealloc %21 : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable>
    tt.return
  }
}


"""

file = pathlib.Path("matmul_kernel_persistent.ttgir")
file.write_text(matmul_kernel_persistent_ttgir)
matmul_kernel_persistent_precompiled = triton.compile(str(file))


def matmul_persistent(a, b):
    configs = {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_stages": 4,
            "num_warps": 8
        }, torch.float16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3,
            "num_warps": 8
        }
    }
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )

    #assert a.stride(1) == 1 and b.stride(0) == 1 and c.stride(1) == 1
    #bytes_per_elem = a.element_size()
    #flops_str = f"flops{bytes_per_elem * 8}"
    #with proton.scope(f"precompiled [M={M}, N={N}, K={K}]",
    #                  {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
    #    matmul_kernel_persistent_precompiled[(grid(configs[torch.float16])[0], 1, 1)](
    #        a,
    #        b,
    #        c,  #
    #        M,
    #        N,
    #        K,  #
    #        a.stride(0),
    #        b.stride(1),  #
    #        c.stride(0),
    #    )

    matmul_kernel_persistent[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
        NUM_SMS=NUM_SMS,  #
        num_stages=configs[dtype]["num_stages"],  #
        num_warps=configs[dtype]["num_warps"],  #
    )

    #kernel = matmul_kernel_persistent.warmup(
    #    a, b, c,  #
    #    M, N, K,  #
    #    a.stride(0), a.stride(1),  #
    #    b.stride(0), b.stride(1),  #
    #    c.stride(0), c.stride(1),  #
    #    BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
    #    BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
    #    BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
    #    GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
    #    NUM_SMS=NUM_SMS,  #
    #    num_stages=configs[dtype]["num_stages"],  #
    #    num_warps=configs[dtype]["num_warps"],  #
    #    grid=grid
    #)
    #print(kernel.asm["ttgir"])
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(a_desc_ptr, b_desc_ptr, c_desc_ptr,  #
                                 M, N, K,  #
                                 BLOCK_SIZE_M: tl.constexpr,  #
                                 BLOCK_SIZE_N: tl.constexpr,  #
                                 BLOCK_SIZE_K: tl.constexpr,  #
                                 GROUP_SIZE_M: tl.constexpr,  #
                                 FP8_OUTPUT: tl.constexpr,  #
                                 NUM_SMS: tl.constexpr):  #
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K

            a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype)
            b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype)
            accumulator = tl.dot(a, b.T, accumulator)

        c = accumulator.to(dtype)
        tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])


def matmul_tma_persistent(a, b):
    # Autotuner does not work with TMA. Use manual config.
    configs = {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_stages": 4,
            "num_warps": 8
        }, torch.float16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3,
            "num_warps": 8
        }
    }

    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    desc_a = triton.tools.experimental_descriptor.create_2d_tma_descriptor(a.data_ptr(), M, K,
                                                                           configs[dtype]["BLOCK_SIZE_M"],
                                                                           configs[dtype]["BLOCK_SIZE_K"],
                                                                           a.element_size())
    desc_b = triton.tools.experimental_descriptor.create_2d_tma_descriptor(b.data_ptr(), N, K,
                                                                           configs[dtype]["BLOCK_SIZE_N"],
                                                                           configs[dtype]["BLOCK_SIZE_K"],
                                                                           b.element_size())
    desc_c = triton.tools.experimental_descriptor.create_2d_tma_descriptor(c.data_ptr(), M, N,
                                                                           configs[dtype]["BLOCK_SIZE_M"],
                                                                           configs[dtype]["BLOCK_SIZE_N"],
                                                                           c.element_size())
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_tma_persistent[grid](
        desc_a, desc_b, desc_c,  #
        M, N, K,  #
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        num_stages=configs[dtype]["num_stages"],  #
        num_warps=configs[dtype]["num_warps"],  #
    )
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_descriptor_persistent(a_ptr, b_ptr, c_ptr,  #
                                        M, N, K,  #
                                        BLOCK_SIZE_M: tl.constexpr,  #
                                        BLOCK_SIZE_N: tl.constexpr,  #
                                        BLOCK_SIZE_K: tl.constexpr,  #
                                        GROUP_SIZE_M: tl.constexpr,  #
                                        NUM_SMS: tl.constexpr):  #
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl._experimental_make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl._experimental_make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl._experimental_make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:

            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)

            c_desc.store([offs_am, offs_bn], c)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_descriptor_persistent(a, b):
    configs = {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_stages": 4,
            "num_warps": 8
        }, torch.float16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3,
            "num_warps": 8
        }
    }

    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_descriptor_persistent[grid](
        a, b, c,  #
        M, N, K,  #
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
        NUM_SMS=NUM_SMS,  #
        num_stages=configs[dtype]["num_stages"],  #
        num_warps=configs[dtype]["num_warps"],  #
    )
    return c


def cublas_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(f"cublas [M={M}, N={N}, K={K}]",
                      {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
        cublas.matmul(a, b, c)
    return c


def torch_matmul(a, b):
    M, K = a.shape
    N, K = b.shape
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(f"torch [M={M}, N={N}, K={K}]",
                      {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
        c = torch.matmul(a, b.T)
    return c


@contextmanager
def proton_context():
    proton.activate(0)
    try:
        yield
    finally:
        proton.deactivate(0)


def bench_fn(reps, warmup_reps, fn, *args):
    for _ in range(warmup_reps):
        fn(*args)
    with proton_context():
        for _ in range(reps):
            fn(*args)


def bench(K, dtype, reps=1000, warmup_reps=10000):
    M = 8192
    N = 8192
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)

    b = b.T.contiguous()

    #if cublas is not None:
    #    bench_fn(reps, warmup_reps, cublas_matmul, a, b)
    #if dtype == torch.float16:
    #    bench_fn(reps, warmup_reps, torch_matmul, a, b)
    bench_fn(reps, warmup_reps, matmul, a, b.T)
    bench_fn(reps, warmup_reps, matmul_persistent_fused, a, b.T)
    bench_fn(reps, warmup_reps, matmul_persistent, a, b.T)
    #if supports_tma():
    #    bench_fn(reps, warmup_reps, matmul_tma_persistent, a, b)
    #    bench_fn(reps, warmup_reps, matmul_descriptor_persistent, a, b)


def validate(M, N, K, dtype):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.T.contiguous()

    torch_result = torch_matmul(a, b) if dtype == torch.float16 else None
    cublas_result = cublas_matmul(a, b) if cublas is not None else None
    naive_result = matmul(a, b.T)
    persistent_result = matmul_persistent(a, b.T)
    tma_persistent_result = matmul_tma_persistent(a, b) if supports_tma() else None
    descriptor_persistent_result = matmul_descriptor_persistent(a, b) if supports_tma() else None

    if torch_result is not None:
        naive_vs_torch = "✅" if torch.allclose(naive_result.to(torch.float16), torch_result.to(torch.float16),
                                               atol=1.0) else "❌"
    if cublas_result is not None:
        naive_vs_cublas = "✅" if torch.allclose(naive_result.to(torch.float16), cublas_result.to(torch.float16),
                                                atol=1.0) else "❌"
    naive_vs_persistent = "✅" if torch.allclose(naive_result.to(torch.float16), persistent_result.to(torch.float16),
                                                atol=1.0) else "❌"
    if tma_persistent_result is not None:
        naive_vs_tma_persistent = "✅" if torch.allclose(cublas_result.to(torch.float16),
                                                        tma_persistent_result.to(torch.float16), atol=1.0) else "❌"
    if descriptor_persistent_result is not None:
        naive_vs_descriptor_persistent = "✅" if torch.allclose(cublas_result.to(
            torch.float16), descriptor_persistent_result.to(torch.float16), atol=1.0) else "❌"
    print(f"M={M}, N={N}, K={K} verification naive vs: ", end="")
    if torch_result is not None:
        print(f"torch: {naive_vs_torch} ", end="")
    if cublas_result is not None:
        print(f"cublas: {naive_vs_cublas} ", end="")
    print(f"persistent: {naive_vs_persistent} ", end="")
    if tma_persistent_result is not None:
        print(f"TMA persistent: {naive_vs_tma_persistent} ", end="")
    if descriptor_persistent_result is not None:
        print(f"Tensor descriptor persistent: {naive_vs_descriptor_persistent} ", end="")
    print()


def show_profile(precision, profile_name):
    import triton.profiler.viewer as proton_viewer
    metrics = ["time/ms"]
    if precision == 'fp8':
        metrics = ["tflop8/s"] + metrics
    elif precision == 'fp16':
        metrics = ["tflop16/s"] + metrics
    file_name = f"{profile_name}.hatchet"
    proton_viewer.parse(metrics, file_name, depth=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--prec", type=str, choices=["fp8", "fp16"], default="fp16")
    args = parser.parse_args()

    if args.prec == 'fp8' and (not hasattr(torch, "float8_e4m3fn") or not is_cuda()):
        print("This example requires CUDA with fp8 support.")
        exit(1)

    dtype = torch.float8_e4m3fn if args.prec == 'fp8' else torch.float16

    if args.K and args.K_range is None:
        args.K_range = [args.K, args.K]
        args.K_step = 1  # doesn't matter as long as it's not 0

    torch.manual_seed(0)

    validate(32, 32, 32, dtype)
    validate(8192, 8192, 512, dtype)

    proton.start("matmul", hook="triton")
    for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
        bench(K, dtype)
    proton.finalize()
    show_profile(args.prec, "matmul")
