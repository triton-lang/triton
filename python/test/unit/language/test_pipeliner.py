# End-to-end tests to check the correctness of the pipeliner

import pytest
import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_cuda_tma_available():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


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
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr):
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
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        mask_a = (offs_am[:, None] < M) & (offs_k[None, :] + k * BLOCK_K < K)
        mask_b = ((offs_k[:, None] + k * BLOCK_K) < K) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    accumulator = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(output_ptrs, accumulator, mask=mask_c)


@triton.jit
def matmul_kernel_tma(  #
        a_ptr, b_ptr, output_ptr,  #
        M, N, K,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M) % M
    offs_bn = (pid_n * BLOCK_N) % N
    offs_am = tl.multiple_of(offs_am, BLOCK_M)
    offs_bn = tl.multiple_of(offs_bn, BLOCK_N)
    offs_k = 0
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl._experimental_descriptor_load(a_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K], tl.float16)
        b = tl._experimental_descriptor_load(b_ptr, [offs_k, offs_bn], [BLOCK_K, BLOCK_N], tl.float16)
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_K
    accumulator = accumulator.to(tl.float16)
    tl._experimental_descriptor_store(output_ptr, accumulator, [offs_am, offs_bn])


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
    M, N, K = 512, 512, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    NUM_STAGES = 4
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)
    output = torch.empty((M, N), dtype=torch.float16, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    if is_cuda_tma_available():
        a_tma = triton.tools.experimental_descriptor.create_2d_tma_descriptor(a.data_ptr(), M, K, BLOCK_M, BLOCK_K,
                                                                              a.element_size())
        b_tma = triton.tools.experimental_descriptor.create_2d_tma_descriptor(b.data_ptr(), K, N, BLOCK_K, BLOCK_N,
                                                                              b.element_size())
        output_tma = triton.tools.experimental_descriptor.create_2d_tma_descriptor(output.data_ptr(), M, N, BLOCK_M,
                                                                                   BLOCK_N, output.element_size())
        handler = matmul_kernel_tma[grid](a_tma, b_tma, output_tma, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                          NUM_STAGES=NUM_STAGES)
    else:
        handler = matmul_kernel[grid](a, b, output, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                      output.stride(0), output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K,
                                      NUM_STAGES=NUM_STAGES)
    ref_out = torch.matmul(a, b)
    atol = 1e-2 if is_hip_mi200() else None
    # Bigger tolerance for AMD MI200 devices.
    # MI200 devices use reduced precision fp16 and bf16 and flush input and
    # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    rtol = 1e-2 if is_hip_mi200() else None
    torch.testing.assert_close(ref_out, output, atol=atol, rtol=rtol)
    if is_cuda():
        ttgir = handler.asm["ttgir"]
        if is_cuda_tma_available():
            assert ttgir.count("triton_nvidia_gpu.async_tma_copy_global_to_local") != 0, "async tma copy not found"
            assert ttgir.count(f"num = {NUM_STAGES} : i32") == 0, "num_stages not match"
            # a_tma, b_tma, output_tma, barriar
            assert ttgir.count("triton_gpu.local_alloc") == 4, "alloc number not match"
            assert ttgir.count("triton_nvidia_gpu.barrier_expect") != 0, "barrier_expect not found"
            assert ttgir.count("triton_nvidia_gpu.wait_barrier") != 0, "wait_barrier not found"
            assert ttgir.count("triton_nvidia_gpu.warp_group_dot") != 0, "warp_group_dot not found"
        else:
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


@pytest.mark.parametrize("ROW_COUNT", [0, 1, 2, 3])
@pytest.mark.parametrize("NUM_STAGES", [1, 2, 3, 4, 5])
def test_pipeline_epilogue(ROW_COUNT, NUM_STAGES, device):

    @triton.jit
    def kernel_up(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                  NUM_STAGES: tl.constexpr):
        row_step = tl.num_programs(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        for row_idx in tl.range(0, n_rows, row_step, num_stages=NUM_STAGES):
            row_start_ptr = input_ptr + row_idx * input_row_stride
            input_ptrs = row_start_ptr + col_offsets
            val = tl.load(input_ptrs, mask=mask, other=-float('inf'))
            val += 1.0
            output_row_start_ptr = output_ptr + row_idx * output_row_stride
            output_ptrs = output_row_start_ptr + col_offsets
            tl.store(output_ptrs, val, mask=mask)

    width = ROW_COUNT
    depth = 78
    x = torch.zeros(width, depth, device=device)
    y0 = torch.rand_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    kernel_up[(1, )](y0, x, x.stride(0), y0.stride(0), n_rows, n_cols, BLOCK_SIZE, NUM_STAGES)
    assert (y0 == torch.ones_like(x)).all()
