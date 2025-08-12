import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl


def test_chained_matmul(device):
    # Regression test for issue #1601
    def chained_matmul_reference(a, b, c):
        intermediate = torch.einsum('MK,NK->MN', a, b)
        return torch.einsum('MN,NK->MK', intermediate, c)

    @triton.jit
    def chained_matmul_kernel(A,  # shape: (m, k)
                              B,  # shape: (n, k)
                              C,  # shape: (n, k)
                              out,  # shape: (m, k)
                              m, n, k: tl.constexpr,  #
                              block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):

        tl.static_assert(block_k == k, f"expected block_k == k but got {block_k} != {k}")

        block_ix = tl.program_id(0)
        a_tile = (block_ix * block_m + tl.arange(0, block_m))[:, None] * block_k \
            + tl.arange(0, block_k)[None, :]

        a = tl.load(A + a_tile, mask=a_tile < m * k, other=0.0)

        acc = tl.zeros([block_m, block_k], dtype=tl.float32)

        for loop_block_start in range(0, n, block_n):
            bc_tile = (loop_block_start + tl.arange(0, block_n))[:, None] * block_k \
                + tl.arange(0, block_k)[None, :]
            b = tl.load(B + bc_tile, mask=bc_tile < n * k, other=0.0)

            intermediate = tl.dot(a, tl.trans(b))
            intermediate_mask = ((loop_block_start + tl.arange(0, block_n)) < n)[None, :] \
                * (tl.arange(0, block_m) < m)[:, None]

            intermediate = tl.where(intermediate_mask, intermediate, 0.0)

            c = tl.load(C + bc_tile, mask=bc_tile < n * k)

            acc += tl.dot(intermediate.to(A.dtype.element_ty), c)

        tl.store(out + a_tile, acc.to(A.dtype.element_ty), mask=a_tile < m * k)

    m, n, k = 32, 64, 128
    block_m, block_n, block_k = 16, 32, k

    grid = (triton.cdiv(m, block_m), )
    a = torch.randint(low=0, high=2, size=(m, k), dtype=torch.float16, device=device)
    b = torch.randint(low=0, high=2, size=(n, k), dtype=torch.float16, device=device)
    c = torch.randint_like(b, low=0, high=2)
    triton_result = torch.zeros_like(a)

    torch_result = chained_matmul_reference(a, b, c)
    chained_matmul_kernel[grid](
        a, b, c, triton_result, m, n, k,  #
        block_m=block_m, block_n=block_n, block_k=block_k)

    assert (torch_result == triton_result).all()


def test_vecmat(device):

    @triton.jit
    def batched_vecmat(
            # inputs
            A,  # shape: [dim_m, dim_k]
            B,  # shape: [dim_m, dim_n, dim_k]
            # dimensions
        dim_m, dim_n, dim_k,
            # outputs
            output,
            # block information
            block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
        m_index = tl.program_id(0)
        n_index = tl.program_id(1)
        # Output tile
        output_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_n \
            + (n_index * block_n + tl.arange(0, block_n))[None, :]

        vecmat = tl.zeros([block_m, block_n], dtype=A.dtype.element_ty)
        k_blocks = dim_k // block_k
        for k_index in range(k_blocks):
            # Load A tile
            a_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_k \
                + (k_index * block_k + tl.arange(0, block_k))[None, :]
            a = tl.load(A + a_tile)

            # Load B tile, transposed to [n, m, k] in order to broadcast A on a
            # leading dimension.
            b_tile = (m_index * block_m + tl.arange(0, block_m))[None, :, None] * dim_n * dim_k \
                + (n_index * block_n + tl.arange(0, block_n))[:, None, None] * dim_k \
                + (k_index * block_k + tl.arange(0, block_k))[None, None, :]
            b = tl.load(B + b_tile)

            expanded_a, _ = tl.broadcast(a, b)
            vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))

        tl.store(output + output_tile, vecmat)

    M, N, K = 128, 128, 128
    block_m, block_n, block_k = 16, 32, 64

    rs = RandomState(17)
    A_vec = rs.randint(0, 4, (M, K)).astype('float32')
    B_vec = rs.randint(0, 4, (M, N, K)).astype('float32')
    A = A_vec
    B = B_vec

    A_tri = torch.tensor(A, device=device)
    B_tri = torch.tensor(B, device=device)
    C_tri = torch.zeros((M, N), dtype=torch.float32, device=device)

    grid = (M // block_m, N // block_n)

    batched_vecmat[grid](
        A_tri, B_tri, M, N, K, C_tri,  #
        block_m=block_m, block_n=block_n, block_k=block_k,  #
        num_warps=4, num_stages=1)

    A_expanded = A[:, np.newaxis, :]
    A_broadcasted = np.broadcast_to(A_expanded, (M, N, K))
    AB = A_broadcasted * B
    C_ref = np.sum(AB, axis=2)

    np.testing.assert_allclose(C_ref, C_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


@pytest.mark.parametrize("type",
                         ["pre_load", "post_load", "post_pre_mixed", "post_load_two_iters", "post_load_three_iters"])
def test_iv_dependent_matmul(type, device):

    @triton.jit
    def kernel(a_ptr, b_ptr, c_ptr,  #
               M, N, K,  #
               stride_am, stride_ak,  #
               stride_bk, stride_bn,  #
               stride_cm, stride_cn,  #
               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
               type: tl.constexpr):
        pid = tl.program_id(axis=0)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptr = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptr = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        a_ptrs = a_ptr
        b_ptrs = b_ptr
        if type == "post_load_two_iters":
            a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
            b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
        elif type == "post_load_three_iters":
            a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
            b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
            a_ptrs_next_next = a_ptr + 2 * BLOCK_SIZE_K * stride_ak
            b_ptrs_next_next = b_ptr + 2 * BLOCK_SIZE_K * stride_bk

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            if type == "pre_load":
                a_ptrs = a_ptr + k * BLOCK_SIZE_K * stride_ak
                b_ptrs = b_ptr + k * BLOCK_SIZE_K * stride_bk
            elif type == "post_pre_mixed":
                a_ptrs = a_ptr + k * BLOCK_SIZE_K * stride_ak
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator += tl.dot(a, b)
            if type == "post_load":
                a_ptrs = a_ptr + (k + 1) * BLOCK_SIZE_K * stride_ak
                b_ptrs = b_ptr + (k + 1) * BLOCK_SIZE_K * stride_bk
            elif type == "post_pre_mixed":
                b_ptrs = b_ptr + (k + 1) * BLOCK_SIZE_K * stride_bk
            elif type == "post_load_two_iters":
                a_ptrs = a_ptrs_next
                b_ptrs = b_ptrs_next
                a_ptrs_next = a_ptr + (k + 2) * BLOCK_SIZE_K * stride_ak
                b_ptrs_next = b_ptr + (k + 2) * BLOCK_SIZE_K * stride_bk
            elif type == "post_load_three_iters":
                a_ptrs = a_ptrs_next
                b_ptrs = b_ptrs_next
                a_ptrs_next = a_ptrs_next_next
                b_ptrs_next = b_ptrs_next_next
                a_ptrs_next_next = a_ptr + (k + 3) * BLOCK_SIZE_K * stride_ak
                b_ptrs_next_next = b_ptr + (k + 3) * BLOCK_SIZE_K * stride_bk
        c = accumulator.to(tl.float16)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    M = 256
    K = 256
    N = 256
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 32

    a = torch.rand((M, K), device=device)
    b = torch.rand((K, N), device=device)

    torch_output = torch.mm(a, b)
    triton_output = torch.empty_like(torch_output, device=torch_output.device)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    num_stages = 4 if type == "post_load_three_iters" else 3
    kernel[grid](
        a, b, triton_output, M, N, K,  #
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),  #
        triton_output.stride(0), triton_output.stride(1),  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, type=type,  #
        num_stages=num_stages)
    torch.testing.assert_close(torch_output, triton_output, rtol=1e-2, atol=1e-2)


def test_reverse_range(device):

    @triton.jit
    def kernel(in_ptr, out_ptr):
        x0 = tl.arange(0, 512)
        tmp0 = tl.load(in_ptr + (512 - x0))
        tl.store(out_ptr + x0, tmp0)

    data = torch.randn((516, ), dtype=torch.float32, device=device)
    res = torch.empty((512, ), dtype=torch.float32, device=device)
    kernel[(1, )](data, res)
    ref = torch.flip(data[1:513], [0])
    assert (res == ref).all()


@triton.jit
def _triton_cummax_helper_fn(arg0_0, arg0_1, arg1_0, arg1_1):
    tmp0 = arg0_0 > arg1_0
    tmp1 = arg0_0 == arg1_0
    tmp2 = arg0_1 > arg1_1
    tmp3 = tmp1 & tmp2
    tmp4 = tmp0 | tmp3
    tmp5 = tl.where(tmp4, arg0_0, arg1_0)
    tmp6 = tl.where(tmp4, arg0_1, arg1_1)
    return tmp5, tmp6


def test_inductor_cummax_bool(device):

    @triton.jit
    def triton_(in_ptr0, out_ptr0, out_ptr1, XBLOCK: tl.constexpr):
        offset = tl.arange(0, XBLOCK)
        tmp0 = tl.load(in_ptr0 + offset).to(tl.int1)
        tmp1 = tmp0.to(tl.int1)
        tmp3 = offset.to(tl.int64)
        tmp5, tmp6, = tl.associative_scan((
            tmp1,
            tmp3,
        ), 0, _triton_cummax_helper_fn)
        tl.store(out_ptr0 + offset, tmp5)
        tl.store(out_ptr1 + offset, tmp6)

    a = torch.randn((64, ), device=device) > 0
    values = torch.empty((64, ), dtype=torch.bool, device=device)
    indices = torch.empty((64, ), dtype=torch.int64, device=device)
    ref = torch.cummax(a, dim=0)

    triton_[(1, )](a, values, indices, 64)
    torch.testing.assert_close(ref.values, values)
    torch.testing.assert_close(ref.indices, indices)
