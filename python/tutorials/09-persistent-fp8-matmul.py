import argparse
import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl
import triton.profiler as proton

from triton._C.libtriton import nvidia

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9):
    print("This tutorial fp8_matmul is only supported on CUDA with cc >= 90")
    sys.exit(0)

cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
cublas = nvidia.cublas.CublasLt(cublas_workspace)


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def _matmul_launch_metadata(grid, kernel, args):
    ret = dict()
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    ret["flops8"] = 2. * M * N * K
    ret["bytes"] = M * K + N * K
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
                  FP8_OUTPUT: tl.constexpr,  #
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

    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    offs_am = tl.where(offs_am < M - start_m, offs_am, 0)
    offs_bn = tl.where(offs_bn < N - start_n, offs_bn, 0)

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

    if FP8_OUTPUT:
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    GROUP_SIZE = 8
    num_stages = 3
    num_warps = 8

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
        BLOCK_SIZE_M=BLOCK_SIZE_M,  #
        BLOCK_SIZE_N=BLOCK_SIZE_N,  #
        BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        num_stages=num_stages,  #
        num_warps=num_warps,  #
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
                             FP8_OUTPUT: tl.constexpr,  #
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

    for i in range(0, k_tiles * tiles_per_SM):
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
            offs_am = tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tl.arange(0, BLOCK_SIZE_N)
            offs_am = tl.where(offs_am < M - start_m, offs_am, 0)
            offs_bn = tl.where(offs_bn < N - start_n, offs_bn, 0)
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
            if FP8_OUTPUT:
                c = accumulator.to(tl.float8e4nv)
            else:
                c = accumulator.to(tl.float16)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_persistent(a, b):
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    GROUP_SIZE = 8
    num_stages = 3
    num_warps = 8

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
    matmul_kernel_persistent[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=BLOCK_SIZE_M,  #
        BLOCK_SIZE_N=BLOCK_SIZE_N,  #
        BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE,  #
        NUM_SMS=NUM_SMS,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        num_stages=num_stages,  #
        num_warps=num_warps,  #
    )
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

            offs_am = tl.multiple_of(offs_am, BLOCK_SIZE_M)
            offs_bn = tl.multiple_of(offs_bn, BLOCK_SIZE_N)

        offs_k = ki * BLOCK_SIZE_K

        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype)
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if FP8_OUTPUT:
                c = accumulator.to(tl.float8e4nv)
            else:
                c = accumulator.to(tl.float16)

            tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_tma_persistent(a, b):
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    GROUP_SIZE = 8
    num_stages = 3
    num_warps = 8

    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.zeros((M, N), device=a.device, dtype=dtype)

    TMA_SIZE = 128

    desc_a = np.empty(TMA_SIZE, dtype=np.int8)
    desc_b = np.empty(TMA_SIZE, dtype=np.int8)
    desc_c = np.empty(TMA_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(a.data_ptr(), M, K, BLOCK_SIZE_M, BLOCK_SIZE_K,
                                                              a.element_size(), desc_a)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(b.data_ptr(), N, K, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                                              b.element_size(), desc_b)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(c.data_ptr(), M, N, BLOCK_SIZE_M, BLOCK_SIZE_N,
                                                              c.element_size(), desc_c)

    desc_a = torch.tensor(desc_a, device="cuda")
    desc_b = torch.tensor(desc_b, device="cuda")
    desc_c = torch.tensor(desc_c, device="cuda")

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_tma_persistent[grid](
        desc_a, desc_b, desc_c,  #
        M, N, K,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M,  #
        BLOCK_SIZE_N=BLOCK_SIZE_N,  #
        BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        num_stages=num_stages,  #
        num_warps=num_warps,  #
    )
    return c


def cublas_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    with proton.scope(f"cublas M={M}, N={N}, K={K}", {"bytes": M * K + N * K, "flops8": 2. * M * N * K}):
        if dtype == torch.float8_e4m3fn:
            cublas.fp8_matmul(a, b, c)
        else:
            c = torch.matmul(a, b.T)
    return c


def bench(K, dtype, reps=10):
    M = 8192
    N = 8192
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)

    b = b.T.contiguous()

    proton.activate(0)

    for _ in range(reps):
        cublas_matmul(a, b)
        time.sleep(0.01)
    for _ in range(reps):
        matmul(a, b.T)
        time.sleep(0.01)
    for _ in range(reps):
        matmul_persistent(a, b.T)
        time.sleep(0.01)
    for _ in range(reps):
        matmul_tma_persistent(a, b)
        time.sleep(0.01)

    proton.deactivate(0)


def validate(M, N, K, dtype):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.T.contiguous()
    if dtype == torch.float8_e4m3fn:
        cublas_result = cublas_matmul(a, b)
    else:
        cublas_result = torch.matmul(a, b.T)
    # print(f"{cublas_result=}")
    naive_result = matmul(a, b.T)
    # print(f"{naive_result=}")
    persistent_result = matmul_persistent(a, b.T)
    # print(f"{persistent_result=}")
    tma_persistent_result = matmul_tma_persistent(a, b)
    # print(f"{tma_persistent_result=}")

    naive_vs_cublas = "✅" if torch.allclose(naive_result.to(torch.float16), cublas_result.to(torch.float16),
                                            atol=1.0) else "❌"
    naive_vs_persistent = "✅" if torch.allclose(naive_result.to(torch.float16), persistent_result.to(torch.float16),
                                                atol=1.0) else "❌"
    naive_vs_tma_persistent = "✅" if torch.allclose(cublas_result.to(torch.float16),
                                                    tma_persistent_result.to(torch.float16), atol=1.0) else "❌"
    print(
        f"M={M}, N={N}, K={K} verification naive vs: cublas {naive_vs_cublas}, persistent {naive_vs_persistent}, TMA persistent {naive_vs_tma_persistent}"
    )


parser = argparse.ArgumentParser()
parser.add_argument("-K", type=int, required=False)
parser.add_argument("--K_range", type=int, nargs=2)
parser.add_argument("--K_step", type=int, default=512)
parser.add_argument("--prec", type=str, choices=['fp8', 'fp16'], default='fp8')
args = parser.parse_args()

if args.prec == 'fp8' and (not hasattr(torch, "float8_e4m3fn") or not is_cuda()):
    print("This example requires CUDA with fp8 support.")
    exit(1)

dtype = torch.float8_e4m3fn if args.prec == 'fp8' else torch.float16

if args.K:
    args.K_range = [args.K, args.K]
    args.K_step = 1  # doesn't matter as long as it's not 0

torch.manual_seed(0)

validate(32, 32, 32, dtype)
validate(8192, 8192, 512, dtype)

proton.start("matmul", hook="triton")
for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
    bench(K, dtype)
proton.finalize()
