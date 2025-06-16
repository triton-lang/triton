import pytest
import torch

import triton
import triton.language as tl
import triton.testing_autows as utils
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def matmul_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    dtype = tl.float8e4nv if DTYPE == "fp8" else tl.float16
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    accumulator = accumulator.to(dtype)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator)


@pytest.mark.parametrize("M", [128, 256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("N", [256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("K", [128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("BLOCK_M", [128])
@pytest.mark.parametrize("BLOCK_N", [128, 256])
@pytest.mark.parametrize("BLOCK_K", [64, 128])
@pytest.mark.parametrize("DTYPE", ["fp8", "fp16"])
@pytest.mark.parametrize(
    ("ENABLE_WARP_SPECIALIZATION", "NUM_WARPS", "NUM_STAGES", "MMA_DEPTH"),
    [(True, 4, 3, 2), (True, 8, 5, 3), (True, 8, 7, 4), (False, 8, 3, 2)],
)
def test_experimental_matmul(
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

    if BLOCK_K == 128 and NUM_STAGES > 4:
        pytest.skip(
            "FIXME: fails with unspecified launch failure for NUM_STAGES <= 2 when BLOCK_K=128"
        )

    dtype = utils.torch_dtype(DTYPE)
    A = utils.generate_input((M, K), dtype)
    B = utils.generate_input((N, K), dtype)
    C = torch.empty((M, N), dtype=dtype, device="cuda")

    desc_a = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K])
    desc_b = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_N, BLOCK_K])

    # calculate shared memory usage of kernel
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

    # In some cases, allocated buffers have gaps in the offsets between them
    # utils.init_check_shared_memory_hook(matmul_kernel, shared_memory)

    try:
        matmul_kernel[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
            desc_a,
            desc_b,
            C,
            M,
            N,
            K,
            C.stride(0),
            C.stride(1),
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            DTYPE,
            num_warps=NUM_WARPS,  # 4 warps (one warp group) for producer and 8 warps for consumer
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

    utils.verify_matmul(A, B, C)


@triton.jit
def matmul_scale_rhs_kernel(
    a_desc,
    b_desc,
    b_scale_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
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
    offs_k_scales = tl.arange(0, BLOCK_SIZE_K)
    dtype = tl.float16
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    b_scale_ptr = b_scale_ptr + offs_k_scales[None, :]
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        scales = tl.load(b_scale_ptr)
        b *= scales
        accumulator = tl.dot(a, b.T, acc=accumulator)
        offs_k += BLOCK_SIZE_K
        b_scale_ptr += BLOCK_SIZE_K

    accumulator = accumulator.to(dtype)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator)


@pytest.mark.parametrize("M", [128, 1024])
@pytest.mark.parametrize("N", [256, 1024])
@pytest.mark.parametrize("K", [128, 256])
@pytest.mark.parametrize("BLOCK_M", [128])
@pytest.mark.parametrize("BLOCK_N", [128])
@pytest.mark.parametrize("BLOCK_K", [64])
@pytest.mark.parametrize(("NUM_WARPS", "NUM_STAGES"),[(4, 3)])
def test_matmul_rhs_scale(
    M,
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    NUM_WARPS,
    NUM_STAGES,
):
    utils.common_test_setup(True, NUM_WARPS)

    dtype = utils.torch_dtype("fp16")
    A = utils.generate_input((M, K), dtype)
    B = utils.generate_input((N, K), dtype)
    C = torch.empty((M, N), dtype=dtype, device="cuda")
    B_scales = torch.randn((1, K), dtype=dtype, device="cuda")

    desc_a = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K])
    desc_b = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_N, BLOCK_K])

    out = matmul_scale_rhs_kernel[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
        desc_a,
        desc_b,
        B_scales,
        C,
        M,
        N,
        K,
        C.stride(0),
        C.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        enable_warp_specialization=True,
        force_membar=True,
    )

    utils.verify_matmul(A, B * B_scales, C)


@triton.jit
def matmul_simt_lhs_kernel(
    a_desc,
    b_desc,
    a_scale_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
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
    offs_k_scales = tl.arange(0, BLOCK_SIZE_K)
    dtype = tl.float16
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    a_scale_ptr = a_scale_ptr + offs_k_scales[None, :]
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        scales = tl.load(a_scale_ptr)
        a *= scales
        accumulator = tl.dot(a, b.T, acc=accumulator)
        offs_k += BLOCK_SIZE_K
        a_scale_ptr += BLOCK_SIZE_K

    accumulator = accumulator.to(dtype)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator)


@pytest.mark.parametrize("M", [128, 1024])
@pytest.mark.parametrize("N", [256, 1024])
@pytest.mark.parametrize("K", [128, 256])
@pytest.mark.parametrize("BLOCK_M", [128])
@pytest.mark.parametrize("BLOCK_N", [128])
@pytest.mark.parametrize("BLOCK_K", [64])
# FIXME: cannot have both True and False due to how monkeypatch works
@pytest.mark.parametrize("PROMOTE_LHS_TMEM", [True])
@pytest.mark.parametrize(("NUM_WARPS", "NUM_STAGES"),[(4, 3)])
def test_matmul_simt_lhs(
    M,
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    NUM_WARPS,
    NUM_STAGES,
    PROMOTE_LHS_TMEM,
    monkeypatch,
):
    utils.common_test_setup(True, NUM_WARPS)

    dtype = utils.torch_dtype("fp16")
    A = utils.generate_input((M, K), dtype)
    B = utils.generate_input((N, K), dtype)
    C = torch.empty((M, N), dtype=dtype, device="cuda")
    A_scales = torch.randn((1, K), dtype=dtype, device="cuda")

    desc_a = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K])
    desc_b = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_N, BLOCK_K])

    if PROMOTE_LHS_TMEM and (not utils.is_sm10x() or monkeypatch is None):
        pytest.skip()

    if PROMOTE_LHS_TMEM:
        monkeypatch.setenv("ALLOW_LHS_TMEM_LAYOUT_CONVERSION", "1")

    out = matmul_simt_lhs_kernel[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
        desc_a,
        desc_b,
        A_scales,
        C,
        M,
        N,
        K,
        C.stride(0),
        C.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        enable_warp_specialization=True,
        force_membar=True,
    )

    utils.verify_matmul(A * A_scales, B, C)

    ttgir = out.asm["ttgir"]

    if PROMOTE_LHS_TMEM:
        assert "nvws.group.simt" in ttgir
        assert f"ttng.tmem_alloc {{aref_buffer}} : () -> !ttg.memdesc<2x{BLOCK_M}x{BLOCK_K}xf16" in ttgir
    else:
        assert not "nvws.group.simt" in ttgir
