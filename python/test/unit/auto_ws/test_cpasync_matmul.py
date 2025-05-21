import pytest
import torch

import triton
import triton.language as tl
import triton.testing_autows as utils
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def matmul_kernel(  #
        a_ptr, b_ptr, bias_ptr, output_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = accumulator.to(tl.float16)
    bias_ptrs = bias_ptr + offs_bn[None, :]
    bias = tl.load(bias_ptrs)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(output_ptrs, acc + bias)


@pytest.mark.parametrize("M", [256, 1024, 8192])
@pytest.mark.parametrize("N", [256, 1024, 4096])
@pytest.mark.parametrize("K", [128, 1024, 2048])
@pytest.mark.parametrize("BLOCK_M", [128])
@pytest.mark.parametrize("BLOCK_N", [128, 256])
@pytest.mark.parametrize("BLOCK_K", [64, 128])
@pytest.mark.parametrize("NUM_WARPS", [4])
def test_simple_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, device):
    NUM_STAGES = 3
    torch.manual_seed(42)
    dtype = torch.float16
    dtype_dst = torch.float16
    a = utils.generate_input((M, K), dtype)
    b = utils.generate_input((K, N), dtype)
    bias = utils.generate_input((1, N), dtype_dst)
    output = torch.empty((M, N), dtype=dtype_dst, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)

    try:
        k = matmul_kernel[grid](a, b, bias, output, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), output.stride(0),
                                output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, num_stages=NUM_STAGES, num_warps=NUM_WARPS,
                                enable_warp_specialization=True)
    except triton.runtime.errors.OutOfResources:
        pytest.skip()

    utils.verify_matmul(a, b.T.contiguous(), output, bias)


@triton.jit
def simple_persistent_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
                             stride_bk, stride_bn,  #
                             stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                             BLOCK_SIZE_K: tl.constexpr,  #
                             GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    tile_id_c = start_pid - NUM_SMS  # remat value to use in the epilogue
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in tl.range(0, k_tiles * tiles_per_SM):
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
            tile_id_c += NUM_SMS
            group_id = tile_id_c // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id_c % group_size_m)
            pid_n = (tile_id_c % num_pid_in_group) // group_size_m

            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            if (c_ptr.dtype == tl.float8e4nv):
                c = accumulator.to(tl.float8e4nv)
            else:
                c = accumulator.to(tl.float16)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

@pytest.mark.parametrize("M", [256, 1024, 8192])
@pytest.mark.parametrize("N", [256, 1024, 4096])
@pytest.mark.parametrize("K", [128, 1024, 2048])
@pytest.mark.parametrize("BLOCK_M", [128])
@pytest.mark.parametrize("BLOCK_N", [128, 256])
@pytest.mark.parametrize("BLOCK_K", [64, 128])
@pytest.mark.parametrize("NUM_WARPS", [4])
def test_simple_persistent_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, device):
    NUM_STAGES = 3
    torch.manual_seed(42)
    dtype = torch.float16
    a = utils.generate_input((M, K), dtype)
    b = utils.generate_input((K, N), dtype)
    dtype_dst = torch.float16
    output = torch.empty((M, N), dtype=dtype_dst, device=device)
    NUM_SMS = utils.get_num_sms()
    grid = (min(NUM_SMS, (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))),)

    try:
        k = simple_persistent_kernel[grid](a, b, output, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), output.stride(0),
                                           output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, 8, NUM_SMS, num_stages=NUM_STAGES, num_warps=NUM_WARPS,
                                           enable_warp_specialization=True)
    except triton.runtime.errors.OutOfResources:
        pytest.skip()

    # print(k.asm["ttgir"])
    utils.verify_matmul(a, b.T.contiguous(), output)


@triton.jit
def matmul_kernel_mixed_tma_cpasync(  #
        a_desc, b_ptr, output_ptr,  #
        M, N, K,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_m_tma = pid_m * BLOCK_M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    offs_k_tma = 0
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, tl.cdiv(K, BLOCK_K)):
        a = a_desc.load([offs_m_tma, offs_k_tma])
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, acc=accumulator)
        b_ptrs += BLOCK_K * stride_bk
        offs_k_tma += BLOCK_K

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(output_ptrs, accumulator.to(tl.float16))


@pytest.mark.parametrize("M", [256, 1024, 8192])
@pytest.mark.parametrize("N", [256, 1024, 4096])
@pytest.mark.parametrize("K", [128, 1024, 2048])
@pytest.mark.parametrize("BLOCK_M", [128])
@pytest.mark.parametrize("BLOCK_N", [128, 256])
@pytest.mark.parametrize("BLOCK_K", [64, 128])
@pytest.mark.parametrize("NUM_WARPS", [4])
def test_mixed_tma_cpasync(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, device):
    NUM_STAGES = 3
    torch.manual_seed(42)
    dtype = torch.float16
    a = utils.generate_input((M, K), dtype)
    b = utils.generate_input((K, N), dtype)
    dtype_dst = torch.float16
    output = torch.empty((M, N), dtype=dtype_dst, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])

    try:
        k = matmul_kernel_mixed_tma_cpasync[grid](a_desc, b, output, M, N, K, b.stride(0), b.stride(1), output.stride(0),
                                                  output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, num_stages=NUM_STAGES, num_warps=NUM_WARPS,
                                                  enable_warp_specialization=True)
    except triton.runtime.errors.OutOfResources:
        pytest.skip()

    # print(k.asm["ttgir"])
    utils.verify_matmul(a, b.T.contiguous(), output)
