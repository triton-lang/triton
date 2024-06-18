# End-to-end tests to check the correctness of the pipeliner

import pytest
import torch
import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def check_capabilities():
    if is_cuda():
        cc = torch.cuda.get_device_capability()
        if cc[0] < 8:
            pytest.skip("CUDA 8.0+ required")


@triton.jit
def matmul_kernel(  #
        a_ptr, b_ptr, output_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=NUM_STAGES):
        mask_a = (offs_am[:, None] < M) & (offs_k[None, :] + k * BLOCK_SIZE_K < K)
        mask_b = ((offs_k[:, None] + k * BLOCK_SIZE_K) < K) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(output_ptrs, accumulator, mask=mask_c)


@triton.jit
def vecadd_kernel(a_ptr, b_ptr, output_ptr, n_elements, num_blocks, BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * num_blocks
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    for _ in tl.range(0, num_blocks, num_stages=NUM_STAGES):
        mask = offsets < n_elements
        x = tl.load(a_ptr + offsets, mask=mask)
        y = tl.load(b_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
        offsets += BLOCK_SIZE


def test_pipeline_matmul(device):
    check_capabilities()
    M, N, K = 4096, 4096, 4096
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    NUM_STAGES = 4
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)
    output = torch.empty((M, N), dtype=torch.float16, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    handler = matmul_kernel[grid](a, b, output, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                  output.stride(0), output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES=NUM_STAGES)
    ref_out = torch.matmul(a, b).to(torch.float16)
    torch.testing.assert_close(ref_out, output)
    if is_cuda():
        ttgir = handler.asm["ttgir"]
        # 1. check async
        assert ttgir.count("triton_gpu.async_copy_global_to_local") != 0, "async copy not found"
        # 2. check number of stages
        assert ttgir.count(f"num = {NUM_STAGES} : i32") != 0, "num_stages not match"
        # 3. check alloc
        assert ttgir.count("triton_gpu.local_alloc") == 2, "alloc number not match"
        # 4. check dot
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 9:
            ttgir.count("triton_nvidia_gpu.warp_group_dot") != 0, "warp_group_dot not found"
        else:
            ttgir.count("triton_gpu.dot") != 0, "dot not found"


def test_pipeline_vecadd(device):
    check_capabilities()
    SIZE = 4096
    NUM_BLOCKS = 4
    BLOCK_SIZE = 256
    NUM_STAGES = 3
    a = torch.randn(SIZE, dtype=torch.float16, device=device)
    b = torch.randn(SIZE, dtype=torch.float16, device=device)
    output = torch.empty(SIZE, dtype=torch.float16, device=device)
    grid = (triton.cdiv(SIZE, NUM_BLOCKS * BLOCK_SIZE), 1)
    handler = vecadd_kernel[grid](a, b, output, SIZE, NUM_BLOCKS, BLOCK_SIZE, NUM_STAGES)
    ref_out = a + b
    torch.testing.assert_close(ref_out, output)
    if is_cuda():
        ttgir = handler.asm["ttgir"]
        # 1. check async
        assert ttgir.count("triton_gpu.async_copy_global_to_local") != 0, "async copy not found"
        # 2. check number of stages
        assert ttgir.count(f"num = {NUM_STAGES} : i32") != 0, "num_stages not match"
        # 3. check alloc
        assert ttgir.count("triton_gpu.local_alloc") == 2, "alloc number not match"
