import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def matmul_no_scf_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    c = tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)


@pytest.mark.parametrize('SHAPE,NUM_WARPS,TRANS_A,TRANS_B', [
    (shape, num_warps, trans_a, trans_b)
    for shape in [
        [128, 256, 32],
        # [256, 128, 16],
        [128, 16, 32],
        [32, 128, 64],
        [128, 128, 64],
        [64, 128, 128],
    ]
    for num_warps in [2, 4]
    for trans_a in [False, True]
    for trans_b in [False, True]
])
def test_gemm_no_scf(SHAPE, NUM_WARPS, TRANS_A, TRANS_B):
    SIZE_M, SIZE_N, SIZE_K = SHAPE
    if (TRANS_A):
        a = torch.randn((SIZE_K, SIZE_M), device='cuda', dtype=torch.float16).T
    else:
        a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=torch.float16)

    if (TRANS_B):
        b = torch.randn((SIZE_N, SIZE_K), device='cuda', dtype=torch.float16).T
    else:
        b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=torch.float16)

    c = torch.empty((SIZE_M, SIZE_N), device=a.device, dtype=torch.float32)
    grid = lambda META: (1, )
    matmul_no_scf_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c,
                               stride_am=a.stride(0), stride_ak=a.stride(1),
                               stride_bk=b.stride(0), stride_bn=b.stride(1),
                               stride_cm=c.stride(0), stride_cn=c.stride(1),
                               M=SIZE_M, N=SIZE_N, K=SIZE_K,
                               num_warps=NUM_WARPS)
    golden = torch.matmul(a, b)
    torch.set_printoptions(profile="full")
    assert_close(c, golden, rtol=1e-3, atol=1e-3, check_dtype=False)


@pytest.mark.parametrize('SHAPE,NUM_WARPS,TRANS_A,TRANS_B', [
    (shape, num_warps, trans_a, trans_b)
    for shape in [
        [64, 128, 128],
        [128, 128, 128],
        [16, 16, 32],
        [32, 16, 64],
        [32, 16, 64],
    ]
    for num_warps in [1, 2, 4]
    for trans_a in [False, True]
    for trans_b in [False, True]
])
def test_gemm_no_scf_int8(SHAPE, NUM_WARPS, TRANS_A, TRANS_B):
    guard_for_volta(is_int8=True)

    SIZE_M, SIZE_N, SIZE_K = SHAPE

    if (TRANS_A):
        a = torch.randint(-5, 5, (SIZE_K, SIZE_M), device='cuda', dtype=torch.int8).T
    else:
        a = torch.randint(-5, 5, (SIZE_M, SIZE_K), device='cuda', dtype=torch.int8)

    if (TRANS_B):
        b = torch.randint(-5, 5, (SIZE_N, SIZE_K), device='cuda', dtype=torch.int8).T
    else:
        b = torch.randint(-5, 5, (SIZE_K, SIZE_N), device='cuda', dtype=torch.int8)

    c = torch.empty((SIZE_M, SIZE_N), device=a.device, dtype=torch.int32)

    grid = lambda META: (1, )
    matmul_no_scf_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c,
                               stride_am=a.stride(0), stride_ak=a.stride(1),
                               stride_bk=b.stride(0), stride_bn=b.stride(1),
                               stride_cm=c.stride(0), stride_cn=c.stride(1),
                               M=SIZE_M, N=SIZE_N, K=SIZE_K,
                               num_warps=NUM_WARPS)

    aa = a.cpu()
    bb = b.cpu()
    golden = torch.matmul(aa.float(), bb.float()).int()
    torch.set_printoptions(profile="full")
    torch.testing.assert_close(c.cpu(), golden, check_dtype=False)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator)


def get_variant_golden(a, b):
    SIZE_M = a.shape[0]
    SIZE_K = a.shape[1]
    SIZE_N = b.shape[1]
    assert a.shape[1] == b.shape[0]
    zero_M_K = torch.zeros((SIZE_M, SIZE_K)).cuda()
    zero_3M_K = torch.zeros((3 * SIZE_M, SIZE_K)).cuda()
    zero_K_N = torch.zeros((SIZE_K, SIZE_N)).cuda()
    zero_3K_N = torch.zeros((3 * SIZE_K, SIZE_N)).cuda()
    a_padded = torch.cat((a, zero_M_K, zero_M_K), 0)
    a_padded = torch.cat((a_padded, zero_3M_K, zero_3M_K), 1)
    b_padded = torch.cat((b, zero_K_N, zero_K_N), 0)
    b_padded = torch.cat((b_padded, zero_3K_N, zero_3K_N), 1)
    c_padded = torch.matmul(a_padded, b_padded)
    return c_padded[:SIZE_M, :SIZE_N]

# It's not easy to get a proper error threshold in different size
# Here the gemm calculation is padded to a different size in order to get
# a variant version of the golden result. And the error between golden and
# golden_variant provide reference on selecting the proper rtol / atol.


def get_proper_err(a, b, golden):
    golden_variant = get_variant_golden(a, b)
    golden_diff = golden - golden_variant
    golden_abs_err = torch.max(torch.abs(golden_diff)).item()
    golden_rel_err = torch.max(torch.abs(golden_diff / golden)).item()
    return (golden_abs_err, golden_rel_err)


@pytest.mark.parametrize('SIZE_M,SIZE_N,SIZE_K,NUM_WARPS,BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K,TRANS_A,TRANS_B', [
    # Non-forloop
    [64, 32, 64, 4, 64, 32, 64, False, False],
    [128, 64, 128, 4, 128, 64, 128, False, False],
    [16, 16, 16, 16, 16, 16, 16, False, False],  # wpt overflow issue
    # K-Forloop
    # [16, 16, 64, 4, 8, 8, 8, False, False],  # Wrap threads
    [32, 32, 64, 4, 32, 32, 32, False, False],  # Single shared encoding
    [16, 16, 128, 4, 16, 16, 16, False, False],  # Single shared encoding and small k
    [64, 32, 128, 4, 64, 32, 64, False, False],
    [128, 16, 128, 4, 128, 16, 32, False, False],
    [32, 16, 128, 4, 32, 16, 32, False, False],
    [32, 64, 128, 4, 32, 64, 32, False, False],
    [32, 128, 256, 4, 32, 128, 64, False, False],
    [64, 128, 64, 4, 64, 128, 32, False, False],
    [64, 64, 128, 4, 64, 64, 32, False, False],
    [128, 128, 64, 4, 128, 128, 32, False, False],
    [128, 128, 128, 4, 128, 128, 32, False, False],
    [128, 128, 256, 4, 128, 128, 64, False, False],
    [128, 256, 128, 4, 128, 256, 32, False, False],
    [256, 128, 64, 4, 256, 128, 16, False, False],
    [128, 64, 128, 4, 128, 64, 32, False, False],
    [16, 16, 64, 4, 16, 16, 16, False, False],
    [32, 32, 64, 4, 32, 32, 32, False, False],
    # trans
    [128, 64, 128, 4, 128, 64, 32, True, False],
    [128, 64, 128, 4, 128, 64, 32, False, True],
])
def test_gemm(SIZE_M, SIZE_N, SIZE_K, NUM_WARPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, TRANS_A, TRANS_B):

    if (TRANS_A):
        a = torch.randn((SIZE_K, SIZE_M), device='cuda', dtype=torch.float16).T
    else:
        a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=torch.float16)

    if (TRANS_B):
        b = torch.randn((SIZE_N, SIZE_K), device='cuda', dtype=torch.float16).T
    else:
        b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=torch.float16)

    c = torch.empty((SIZE_M, SIZE_N), device=a.device, dtype=torch.float32)
    grid = lambda META: (1, )
    matmul_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c,
                        stride_am=a.stride(0), stride_ak=a.stride(1),
                        stride_bk=b.stride(0), stride_bn=b.stride(1),
                        stride_cm=c.stride(0), stride_cn=c.stride(1),
                        M=a.shape[0], N=b.shape[1], K=a.shape[1],
                        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                        num_warps=NUM_WARPS)
    golden = torch.matmul(a, b)
    golden_abs_err, golden_rel_err = get_proper_err(a, b, golden)
    torch.set_printoptions(profile="full")
    assert_close(c, golden, rtol=max(1e-4, 1.5 * golden_rel_err), atol=max(1e-4, 1.5 * golden_abs_err), check_dtype=False)


@pytest.mark.parametrize('M,N,K,num_warps,block_M,block_N,block_K,allow_tf32', [
    [32, 32, 16, 4, 32, 32, 16, False],
    [32, 32, 16, 4, 32, 32, 16, True],
    [32, 16, 16, 4, 32, 32, 16, False],
    [32, 16, 16, 4, 32, 32, 16, True],
    [127, 41, 43, 4, 32, 32, 16, False],
    [127, 41, 43, 4, 32, 32, 16, True],
    [128, 8, 8, 4, 32, 32, 16, False],
    [128, 8, 8, 4, 32, 32, 16, True]
])
def test_gemm_fp32(M, N, K, num_warps, block_M, block_N, block_K, allow_tf32):
    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        ALLOW_TF32: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        # num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
            b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
            a = tl.load(a_ptrs, a_mask, other=0.0)
            b = tl.load(b_ptrs, b_mask, other=0.0)
            accumulator += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            offs_k += BLOCK_SIZE_K

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, c_mask)

    guard_for_volta(is_tf32=allow_tf32)

    # Configure the pytorch counterpart
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](a, b, c,
                        M, N, K,
                        stride_am=a.stride(0), stride_ak=a.stride(1),
                        stride_bk=b.stride(0), stride_bn=b.stride(1),
                        stride_cm=c.stride(0), stride_cn=c.stride(1),
                        BLOCK_SIZE_M=block_M, BLOCK_SIZE_N=block_N, BLOCK_SIZE_K=block_K, ALLOW_TF32=allow_tf32)

    golden = torch.matmul(a, b)
    golden_abs_err, golden_rel_err = get_proper_err(a, b, golden)
    if allow_tf32:
        # TF32 is not accurate enough
        torch.testing.assert_close(c, golden, rtol=max(1e-2, 1.5 * golden_rel_err), atol=max(1e-2, 1.5 * golden_abs_err))
    else:
        torch.testing.assert_close(c, golden, rtol=max(1e-4, 1.5 * golden_rel_err), atol=max(1e-4, 1.5 * golden_abs_err))


def guard_for_volta(is_int8=False, is_tf32=False):
    '''
    Tell whether the test case is valid on Volta GPU.
    Some features are WIP, so the corresponding support are missing.
    '''
    capability = torch.cuda.get_device_capability()
    is_on_Volta = capability[0] < 8
    # TODO[Superjomn]: Remove the constraints below when features are ready
    is_feature_supported = not (is_int8 or is_tf32)

    if is_on_Volta:
        if (not is_feature_supported):
            pytest.skip("Not valid on Volta")