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
  tt.func public @matmul_kernel_persistent_fused(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %false = arith.constant false
    %cst = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_0 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c132_i32 = arith.constant 132 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.addi %arg5, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.divsi %7, %c132_i32 : i32
    %9 = arith.remsi %7, %c132_i32 : i32
    %10 = arith.cmpi slt, %0, %9 : i32
    %11 = scf.if %10 -> (i32) {
      %122 = arith.addi %8, %c1_i32 : i32
      scf.yield %122 : i32
    } else {
      scf.yield %8 : i32
    }
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = arith.muli %4, %c8_i32 : i32
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %17 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %18 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %19 = arith.muli %6, %11 : i32
    %20 = arith.subi %6, %c1_i32 : i32
    %21 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %22 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %23 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked>
    %24 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked>
    %25 = tt.expand_dims %12 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %26 = tt.expand_dims %13 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %27 = ttg.local_alloc  : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %28 = ttg.local_alloc  : () -> !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable>
    %29 = arith.cmpi sgt, %19, %c0_i32 : i32
    %30 = arith.divsi %0, %14 : i32
    %31 = arith.muli %30, %c8_i32 : i32
    %32 = arith.subi %2, %31 : i32
    %33 = arith.minsi %32, %c8_i32 : i32
    %34 = arith.remsi %0, %33 : i32
    %35 = arith.addi %31, %34 : i32
    %36 = arith.remsi %0, %14 : i32
    %37 = arith.divsi %36, %33 : i32
    %38 = arith.muli %35, %c128_i32 : i32
    %39 = arith.muli %37, %c256_i32 : i32
    %40 = tt.splat %38 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %41 = arith.addi %40, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %42 = tt.splat %39 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %43 = arith.addi %42, %17 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %44 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %45 = arith.cmpi slt, %41, %44 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %46 = arith.select %45, %41, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %47 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %48 = arith.cmpi slt, %43, %47 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %49 = arith.select %48, %43, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %50 = tt.expand_dims %46 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %51 = arith.muli %50, %21 : tensor<128x1xi32, #blocked1>
    %52 = tt.broadcast %51 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %53 = tt.broadcast %25 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %54 = arith.addi %52, %53 : tensor<128x64xi32, #blocked1>
    %55 = tt.addptr %22, %54 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %56 = tt.expand_dims %49 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %57 = arith.muli %56, %23 : tensor<1x256xi32, #blocked>
    %58 = tt.broadcast %26 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %59 = tt.broadcast %57 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %60 = arith.addi %58, %59 : tensor<64x256xi32, #blocked>
    %61 = tt.addptr %24, %60 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %62 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1>
    %63 = arith.cmpi slt, %25, %62 : tensor<1x64xi32, #blocked1>
    %64 = tt.broadcast %63 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %65 = ttg.memdesc_subview %27[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %66 = tt.splat %29 : i1 -> tensor<128x64xi1, #blocked1>
    %67 = arith.andi %66, %64 : tensor<128x64xi1, #blocked1>
    %68 = ttg.async_copy_global_to_local %55, %65 mask %67 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %69 = ttg.async_commit_group %68
    %70 = tt.splat %arg5 : i32 -> tensor<64x1xi32, #blocked>
    %71 = arith.cmpi slt, %26, %70 : tensor<64x1xi32, #blocked>
    %72 = tt.broadcast %71 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %73 = ttg.memdesc_subview %28[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    %74 = tt.splat %29 : i1 -> tensor<64x256xi1, #blocked>
    %75 = arith.andi %74, %72 : tensor<64x256xi1, #blocked>
    %76 = ttg.async_copy_global_to_local %61, %73 mask %75 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
    %77 = ttg.async_commit_group %76
    %78 = arith.cmpi sgt, %19, %c1_i32 : i32
    %79 = arith.cmpi ne, %20, %c0_i32 : i32
    %80 = arith.extui %79 : i1 to i32
    %81 = arith.cmpi eq, %80, %c0_i32 : i32
    %82:5 = scf.if %81 -> (i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) {
      %122 = arith.addi %0, %c132_i32 : i32
      %123 = arith.divsi %122, %14 : i32
      %124 = arith.muli %123, %c8_i32 : i32
      %125 = arith.subi %2, %124 : i32
      %126 = arith.minsi %125, %c8_i32 : i32
      %127 = arith.remsi %122, %126 : i32
      %128 = arith.addi %124, %127 : i32
      %129 = arith.remsi %122, %14 : i32
      %130 = arith.divsi %129, %126 : i32
      %131 = arith.muli %128, %c128_i32 : i32
      %132 = arith.muli %130, %c256_i32 : i32
      %133 = tt.splat %131 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %134 = arith.addi %133, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %135 = tt.splat %132 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %136 = arith.addi %135, %17 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %137 = arith.cmpi slt, %134, %44 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %138 = arith.select %137, %134, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %139 = arith.cmpi slt, %136, %47 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %140 = arith.select %139, %136, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      scf.yield %122, %128, %130, %138, %140 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    } else {
      scf.yield %0, %35, %37, %46, %49 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    }
    %83 = arith.muli %80, %c64_i32 : i32
    %84 = tt.splat %83 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %85 = tt.splat %83 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %86 = arith.addi %84, %12 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %87 = arith.addi %85, %13 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %88 = tt.expand_dims %82#3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %89 = arith.muli %88, %21 : tensor<128x1xi32, #blocked1>
    %90 = tt.expand_dims %86 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %91 = tt.broadcast %89 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %92 = tt.broadcast %90 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %93 = arith.addi %91, %92 : tensor<128x64xi32, #blocked1>
    %94 = tt.addptr %22, %93 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %95 = tt.expand_dims %87 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %96 = tt.expand_dims %82#4 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %97 = arith.muli %96, %23 : tensor<1x256xi32, #blocked>
    %98 = tt.broadcast %95 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %99 = tt.broadcast %97 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %100 = arith.addi %98, %99 : tensor<64x256xi32, #blocked>
    %101 = tt.addptr %24, %100 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %102 = arith.subi %arg5, %83 : i32
    %103 = tt.splat %102 : i32 -> tensor<1x64xi32, #blocked1>
    %104 = arith.cmpi slt, %25, %103 : tensor<1x64xi32, #blocked1>
    %105 = tt.broadcast %104 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %106 = ttg.memdesc_subview %27[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %107 = tt.splat %78 : i1 -> tensor<128x64xi1, #blocked1>
    %108 = arith.andi %107, %105 : tensor<128x64xi1, #blocked1>
    %109 = ttg.async_copy_global_to_local %94, %106 mask %108 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %110 = ttg.async_commit_group %109
    %111 = tt.splat %102 : i32 -> tensor<64x1xi32, #blocked>
    %112 = arith.cmpi slt, %26, %111 : tensor<64x1xi32, #blocked>
    %113 = tt.broadcast %112 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %114 = ttg.memdesc_subview %28[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    %115 = tt.splat %78 : i1 -> tensor<64x256xi1, #blocked>
    %116 = arith.andi %115, %113 : tensor<64x256xi1, #blocked>
    %117 = ttg.async_copy_global_to_local %101, %114 mask %116 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
    %118 = ttg.async_commit_group %117
    %119:18 = scf.for %arg9 = %c0_i32 to %19 step %c1_i32 iter_args(%arg10 = %80, %arg11 = %82#0, %arg12 = %82#1, %arg13 = %82#2, %arg14 = %cst_3, %arg15 = %82#3, %arg16 = %82#4, %arg17 = %false, %arg18 = %c1_i32, %arg19 = %c-1_i32, %arg20 = %77, %arg21 = %118, %arg22 = %c0_i32, %arg23 = %80, %arg24 = %35, %arg25 = %82#1, %arg26 = %37, %arg27 = %82#2) -> (i32, i32, i32, i32, tensor<128x256xf32, #mma>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i1, i32, i32, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32, i32, i32)  : i32 {
      %122 = arith.subi %19, %c2_i32 : i32
      %123 = arith.cmpi slt, %arg9, %122 : i32
      %124 = arith.cmpi eq, %arg10, %20 : i32
      %125 = arith.addi %arg10, %c1_i32 : i32
      %126 = arith.select %124, %c0_i32, %125 : i32
      %127 = arith.cmpi eq, %126, %c0_i32 : i32
      %128:5 = scf.if %127 -> (i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) {
        %178 = arith.addi %arg11, %c132_i32 : i32
        %179 = arith.divsi %178, %14 : i32
        %180 = arith.muli %179, %c8_i32 : i32
        %181 = arith.subi %2, %180 : i32
        %182 = arith.minsi %181, %c8_i32 : i32
        %183 = arith.remsi %178, %182 : i32
        %184 = arith.addi %180, %183 : i32
        %185 = arith.remsi %178, %14 : i32
        %186 = arith.divsi %185, %182 : i32
        %187 = arith.muli %184, %c128_i32 : i32
        %188 = arith.muli %186, %c256_i32 : i32
        %189 = tt.splat %187 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %190 = arith.addi %189, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %191 = tt.splat %188 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %192 = arith.addi %191, %17 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %193 = arith.cmpi slt, %190, %44 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %194 = arith.select %193, %190, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %195 = arith.cmpi slt, %192, %47 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %196 = arith.select %195, %192, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        scf.yield %178, %184, %186, %194, %196 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      } else {
        scf.yield %arg11, %arg12, %arg13, %arg15, %arg16 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      }
      %129 = arith.addi %arg19, %c1_i32 : i32
      %130 = arith.cmpi slt, %129, %c3_i32 : i32
      %131 = arith.select %130, %129, %c0_i32 : i32
      %132 = ttg.memdesc_subview %27[%131, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %133 = ttg.async_wait %arg20 {num = 2 : i32}
      %134 = ttg.memdesc_subview %28[%131, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %135 = ttng.warp_group_dot %132, %134, %arg14, %arg17 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> * !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> tensor<128x256xf32, #mma>
      %136:3 = ttng.warp_group_dot_wait %135, %132, %134 {pendings = 1 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %137 = arith.addi %arg18, %c1_i32 : i32
      %138 = arith.cmpi slt, %137, %c3_i32 : i32
      %139 = arith.select %138, %137, %c0_i32 : i32
      %140 = arith.muli %126, %c64_i32 : i32
      %141 = tt.splat %140 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %142 = tt.splat %140 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %143 = arith.addi %141, %12 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %144 = arith.addi %142, %13 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %145 = tt.expand_dims %128#3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %146 = arith.muli %145, %21 : tensor<128x1xi32, #blocked1>
      %147 = tt.expand_dims %143 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
      %148 = tt.broadcast %146 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %149 = tt.broadcast %147 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %150 = arith.addi %148, %149 : tensor<128x64xi32, #blocked1>
      %151 = tt.addptr %22, %150 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %152 = tt.expand_dims %144 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %153 = tt.expand_dims %128#4 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
      %154 = arith.muli %153, %23 : tensor<1x256xi32, #blocked>
      %155 = tt.broadcast %152 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %156 = tt.broadcast %154 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %157 = arith.addi %155, %156 : tensor<64x256xi32, #blocked>
      %158 = tt.addptr %24, %157 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
      %159 = arith.subi %arg5, %140 : i32
      %160 = tt.splat %159 : i32 -> tensor<1x64xi32, #blocked1>
      %161 = arith.cmpi slt, %25, %160 : tensor<1x64xi32, #blocked1>
      %162 = tt.broadcast %161 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
      %163 = ttg.memdesc_subview %27[%139, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %164 = tt.splat %123 : i1 -> tensor<128x64xi1, #blocked1>
      %165 = arith.andi %164, %162 : tensor<128x64xi1, #blocked1>
      %166 = ttg.async_copy_global_to_local %151, %163 mask %165 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
      %167 = ttg.async_commit_group %166
      %168 = tt.splat %159 : i32 -> tensor<64x1xi32, #blocked>
      %169 = arith.cmpi slt, %26, %168 : tensor<64x1xi32, #blocked>
      %170 = tt.broadcast %169 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
      %171 = ttg.memdesc_subview %28[%139, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %172 = tt.splat %123 : i1 -> tensor<64x256xi1, #blocked>
      %173 = arith.andi %172, %170 : tensor<64x256xi1, #blocked>
      %174 = ttg.async_copy_global_to_local %158, %171 mask %173 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
      %175 = ttg.async_commit_group %174
      %176 = arith.cmpi eq, %arg22, %20 : i32
      %177 = arith.cmpi ne, %arg22, %20 : i32
      scf.if %176 {
        %178:3 = ttng.warp_group_dot_wait %136#0, %132, %134 {pendings = 0 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
        %179 = arith.muli %arg24, %c128_i32 : i32
        %180 = tt.splat %179 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %181 = arith.addi %180, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %182 = arith.muli %arg26, %c256_i32 : i32
        %183 = tt.splat %182 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %184 = arith.addi %183, %18 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %185 = tt.expand_dims %181 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
        %186 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked2>
        %187 = arith.muli %186, %185 : tensor<128x1xi32, #blocked2>
        %188 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
        %189 = tt.addptr %188, %187 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
        %190 = tt.expand_dims %184 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
        %191 = tt.broadcast %189 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x256x!tt.ptr<f16>, #blocked2>
        %192 = tt.broadcast %190 : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
        %193 = tt.addptr %191, %192 : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
        %194 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked2>
        %195 = arith.cmpi slt, %185, %194 : tensor<128x1xi32, #blocked2>
        %196 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked2>
        %197 = arith.cmpi slt, %190, %196 : tensor<1x256xi32, #blocked2>
        %198 = tt.broadcast %195 : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
        %199 = tt.broadcast %197 : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
        %200 = arith.andi %198, %199 : tensor<128x256xi1, #blocked2>
        %201 = arith.truncf %178#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
        %202 = ttg.convert_layout %201 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked2>
        tt.store %193, %202, %200 : tensor<128x256x!tt.ptr<f16>, #blocked2>
      }
      scf.yield %126, %128#0, %128#1, %128#2, %136#0, %128#3, %128#4, %177, %139, %131, %arg21, %175, %arg23, %126, %arg25, %128#1, %arg27, %128#2 : i32, i32, i32, i32, tensor<128x256xf32, #mma>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i1, i32, i32, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32, i32, i32
    }
    %120 = ttng.warp_group_dot_wait %119#4 {pendings = 0 : i32} : tensor<128x256xf32, #mma>
    %121 = ttg.async_wait  {num = 0 : i32}
    ttg.local_dealloc %27 : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %28 : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable>
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
    #bench_fn(reps, warmup_reps, matmul, a, b.T)
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
