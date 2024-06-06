import argparse
import time

import numpy as np
import torch
import triton
import triton.language as tl

from triton._C.libtriton import nvidia

cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
cublas = nvidia.cublas.CublasLt(cublas_workspace)


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


def matmul(a, b, out_type=torch.float16):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape

    assert out_type in [torch.float16, torch.float8_e4m3fn], "Unsupported output type"
    c = torch.empty((M, N), device=a.device, dtype=out_type)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        BLOCK_SIZE_M=128,  #
        BLOCK_SIZE_N=256,  #
        BLOCK_SIZE_K=128,  #
        GROUP_SIZE_M=8,  #
        FP8_OUTPUT=out_type == torch.float8_e4m3fn,
        num_stages=3,
        num_warps=8,
    )
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
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


def matmul_persistent(a, b, out_type=torch.float16):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape

    assert out_type in [torch.float16, torch.float8_e4m3fn], "Unsupported output type"
    c = torch.empty((M, N), device=a.device, dtype=out_type)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        BLOCK_SIZE_M=128,  #
        BLOCK_SIZE_N=256,  #
        BLOCK_SIZE_K=128,  #
        GROUP_SIZE_M=8,  #
        FP8_OUTPUT=out_type == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        num_stages=3,
        num_warps=8,
    )
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_fused(a_desc_ptr, b_desc_ptr, c_desc_ptr, M, N, K,  #
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

    pid_m = start_pid // num_pid_n - NUM_SMS // num_pid_n
    pid_n = (start_pid % num_pid_n - NUM_SMS % num_pid_n
             )  # after adding a n_step, pid_n should be start_pid % num_pid_n
    tile_id = start_pid - NUM_SMS
    ki = -1

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0

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

        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], tl.float8e4nv)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], tl.float8e4nv)
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if FP8_OUTPUT:
                c = accumulator.to(tl.float8e4nv)
            else:
                c = accumulator.to(tl.float16)

            tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_tma_persistent(a, b, out_type=torch.float16):
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 128
    GROUP_SIZE = 8

    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed

    M, K = a.shape
    N, K = b.shape

    assert out_type in [torch.float16, torch.float8_e4m3fn], "Unsupported output type"
    c = torch.empty((M, N), device=a.device, dtype=out_type)

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
    matmul_kernel_tma_fused[grid](
        desc_a,
        desc_b,
        desc_c,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE,
        FP8_OUTPUT=c.dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        num_stages=3,
        num_warps=8,
    )
    return c


def cublas_matmul(a, b, out_type=torch.float16):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed

    M, K = a.shape
    N, K = b.shape

    assert out_type == torch.float8_e4m3fn, "Unsupported output type"

    c = torch.empty((M, N), device=a.device, dtype=out_type)

    # with proton.scope(f"cublas M={M}, N={N}, K={K}", {"bytes": M*K + N*K, "flops8": 2.*M * N * K}):
    #cublas.fp8_matmul(a, b, c)
    cublas.fp8_matmul_raw(M, N, K, a.data_ptr(), b.data_ptr(), c.data_ptr())
    return c


def bench(K):
    M = 8192
    N = 8192
    out_type = torch.float8_e4m3fn
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)

    b = b.T.contiguous()

    # proton.activate(0)
    for _ in range(10):
        cublas_matmul(a, b, out_type=out_type)
        time.sleep(0.01)
    for _ in range(10):
        matmul(a, b.T, out_type=out_type)
        time.sleep(0.01)
    for _ in range(10):
        matmul_persistent(a, b.T, out_type=out_type)
        time.sleep(0.01)
    for _ in range(10):
        matmul_tma_persistent(a, b, out_type=out_type)
        time.sleep(0.01)
    # proton.deactivate(0)

    # print(cublas_result)
    # print(triton_result)
    # print(triton_result_pers)
    # print(triton_result_tma)

    # if torch.allclose(
    #     triton_result.to(torch.float16), triton_result_tma.to(torch.float16), atol=0.125, rtol=0
    # ):
    #     print("✅ Triton and Torch match for K=", K)
    # else:
    #     print("❌ Triton and Torch differ for K=", K)


parser = argparse.ArgumentParser()
parser.add_argument("-K", type=int, default=4096)
args = parser.parse_args()

torch.manual_seed(0)
# proton.start("matmul", hook="triton")
bench(args.K)
# proton.finalize()
