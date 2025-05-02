import pytest
import torch

import triton
import triton.language as tl
import triton.testing_autows as utils
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def matmul_kernel_tma(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc_ptr.load([offs_am, offs_k])
        b = b_desc_ptr.load([offs_k, offs_bn])
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    accumulator = accumulator.to(tl.float16)
    c_desc_ptr.store([offs_am, offs_bn], accumulator)


@pytest.mark.parametrize("M", [128, 256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("N", [256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("K", [128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("BLOCK_M", [128])
@pytest.mark.parametrize("BLOCK_N", [128, 256])
@pytest.mark.parametrize("BLOCK_K", [64, 128])
@pytest.mark.parametrize("DTYPE", ["fp16"])
@pytest.mark.parametrize(
    ("ENABLE_WARP_SPECIALIZATION", "NUM_WARPS", "NUM_STAGES", "MMA_DEPTH"),
    [(True, 4, 3, 2), (True, 8, 5, 3), (True, 8, 7, 4), (False, 8, 3, 2)],
)
def test_experimental_tma_matmul(
    M,
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    DTYPE,
    NUM_WARPS,
    NUM_STAGES,
    MMA_DEPTH,
    ENABLE_WARP_SPECIALIZATION,
):
    utils.common_test_setup(ENABLE_WARP_SPECIALIZATION, NUM_WARPS)

    dtype = torch.float8_e4m3fn if DTYPE == "fp8" else torch.float16
    A = utils.generate_input((M, K), dtype)
    B = utils.generate_input((K, N), dtype)
    C = torch.empty((M, N), dtype=dtype, device="cuda")

    desc_a = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K])
    desc_b = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_K, BLOCK_N])
    desc_c = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_M, BLOCK_N])

    if ENABLE_WARP_SPECIALIZATION:
        smem_buffers = [
            NUM_STAGES * BLOCK_M * BLOCK_K * utils.dtype_size(dtype),
            NUM_STAGES * BLOCK_N * BLOCK_K * utils.dtype_size(dtype),
            BLOCK_M * BLOCK_N * utils.dtype_size(dtype),
            NUM_STAGES * 8,
            NUM_STAGES * 8,
        ]
        if utils.is_sm10x():
            smem_buffers.append(8) # TMEM full
            smem_buffers.append(8) # TMEM empty
    else:
        smem_buffers = [
            NUM_STAGES * BLOCK_M * BLOCK_K * utils.dtype_size(dtype),
            NUM_STAGES * BLOCK_N * BLOCK_K * utils.dtype_size(dtype),
            NUM_STAGES * 8,
        ]
        if utils.is_sm10x():
            smem_buffers.append(MMA_DEPTH * 8)
    shared_memory = utils.compute_shared_memory(smem_buffers)

    utils.init_check_shared_memory_hook(matmul_kernel_tma, shared_memory)

    try:
        kernel = matmul_kernel_tma[
            (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)
        ](
            desc_a,
            desc_b,
            desc_c,
            M,
            N,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_warps=NUM_WARPS,
            mma_depth=MMA_DEPTH,
            num_stages=NUM_STAGES,
            enable_warp_specialization=ENABLE_WARP_SPECIALIZATION,
        )
    except triton.runtime.errors.OutOfResources:
        assert shared_memory > utils.get_shared_memory()
        return
    finally:
        utils.clear_check_shared_memory_hook()
    assert shared_memory <= utils.get_shared_memory()

    utils.verify_matmul(A, B.T.contiguous(), C)
