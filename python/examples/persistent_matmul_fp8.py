import argparse
import time

import numpy as np
import torch
import triton
import triton.language as tl

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 128
GROUP_SIZE = 8
num_stages = 5
num_warps = 4


@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4)
#     ],
#     key=["M", "N", "K"],
# )
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    ACTIVATION: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)

    if FP8_OUTPUT:
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, out_type=torch.float16, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
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
        BLOCK_SIZE_M=BLOCK_SIZE_M,  #
        BLOCK_SIZE_N=BLOCK_SIZE_N,  #
        BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE,  #
        ACTIVATION=activation,  #
        FP8_OUTPUT=out_type == torch.float8_e4m3fn,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    # print(f"matmul_kernel, {M=}, {N=}, {K=}, best config: {matmul_kernel.best_config}")
    return c


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4)
#     ],
#     key=["M", "N", "K"],
# )
@triton.jit
def matmul_kernel_persistent(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    ACTIVATION: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
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

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.
            accumulator = tl.dot(a, b, accumulator)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        # You can fuse arbitrary activation functions here
        # while the accumulator is still in FP32!
        if ACTIVATION == "leaky_relu":
            accumulator = leaky_relu(accumulator)
        if FP8_OUTPUT:
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def matmul_persistent(a, b, out_type=torch.float16, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape

    assert out_type in [torch.float16, torch.float8_e4m3fn], "Unsupported output type"

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
        BLOCK_SIZE_M=BLOCK_SIZE_M,  #
        BLOCK_SIZE_N=BLOCK_SIZE_N,  #
        BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE,  #
        ACTIVATION=activation,  #
        FP8_OUTPUT=out_type == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    # print(f"matmul_kernel_persistent, {M=}, {N=}, {K=}, best config: {matmul_kernel_persistent.best_config}")
    return c


@triton.jit
def matmul_kernel_tma_fused_tma_store(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    M,
    N,
    K,  #
    stride_cm,
    stride_cn,  #
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,
    NUM_SMS: tl.constexpr,
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
    ki = -1

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0

    n_step = NUM_SMS % num_pid_n
    m_step = NUM_SMS // num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            curr_m_step = tl.where(pid_n + n_step >= num_pid_n, m_step + 1, m_step)
            curr_n_step = tl.where(pid_n + n_step >= num_pid_n, n_step - num_pid_n, n_step)
            pid_m = pid_m + curr_m_step
            pid_n = pid_n + curr_n_step

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

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


@triton.jit
def matmul_kernel_tma_fused_tma_store_grouped(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    M,
    N,
    K,  #
    stride_cm,
    stride_cn,  #
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,
    NUM_SMS: tl.constexpr,
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

    for i in range(0, k_tiles * tiles_per_SM):
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


def matmul_tma_persistent(a, b, desc_a, desc_b, kernel, c=None, desc_c=None, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    N, K = b.shape
    # Allocates output.

    assert c.dtype in [torch.float16, torch.float8_e4m3fn], "Unsupported output type"

    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    kernel[grid](
        desc_a,
        desc_b,
        desc_c,
        M,
        N,
        K,
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE,
        FP8_OUTPUT=c.dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return c


def verification(K, fp8_output):
    M = 8192
    N = 8192
    out_type = torch.float8_e4m3fn if fp8_output else torch.float16

    # a = np.random.randint(0, 256, (M, K), dtype=np.uint8)
    # a = torch.tensor(a, device="cuda", dtype=torch.float8_e4m3fn)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    # b = np.random.randint(0, 256, (K, N), dtype=np.uint8)
    # b = torch.tensor(b, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    c = torch.zeros((M, N), device="cuda", dtype=out_type)

    b = b.T.contiguous()

    TMA_SIZE = 128

    desc_a = np.empty(TMA_SIZE, dtype=np.int8)
    desc_b = np.empty(TMA_SIZE, dtype=np.int8)
    desc_c = np.empty(TMA_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(a.data_ptr(), M, K, BLOCK_SIZE_M, BLOCK_SIZE_K,
                                                              a.element_size(), desc_a)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(b.data_ptr(), N, K, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                                              b.element_size(), desc_b  # transposed
                                                              )
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(c.data_ptr(), M, N, BLOCK_SIZE_M, BLOCK_SIZE_N,
                                                              c.element_size(), desc_c)

    desc_a = torch.tensor(desc_a, device="cuda")
    desc_b = torch.tensor(desc_b, device="cuda")
    desc_c = torch.tensor(desc_c, device="cuda")

    torch_result = torch.matmul(a.to(torch.float16), b.to(torch.float16).T)
    triton_result = matmul(a, b.T, out_type=out_type)
    time.sleep(1)
    triton_result_pers = matmul_persistent(a, b.T, out_type=out_type)
    time.sleep(1)
    triton_result_tma = matmul_tma_persistent(a, b, desc_a, desc_b, matmul_kernel_tma_fused_tma_store_grouped, c=c,
                                              desc_c=desc_c)

    print(torch_result)
    print(triton_result)
    print(triton_result_pers)
    print(triton_result_tma)

    # if torch.allclose(
    #     triton_result.to(torch.float16), triton_result_tma.to(torch.float16), atol=0.125, rtol=0
    # ):
    #     print("✅ Triton and Torch match for K=", K)
    # else:
    #     print("❌ Triton and Torch differ for K=", K)


parser = argparse.ArgumentParser()
parser.add_argument("-K", type=int, default=4096)
parser.add_argument("--fp8_output", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
verification(args.K, args.fp8_output)
