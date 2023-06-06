import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl

# Tests matmul with different argument types and memory layouts.
@pytest.mark.parametrize("lhs_row_major", [False, True])
@pytest.mark.parametrize("rhs_row_major", [False, True])
@pytest.mark.parametrize(("lhs_dtype", "rhs_dtype"),
    [("float16", "float16"), ("float16", "float8"),
     ("float16", "int8"), ("float8", "float16"),
     ("float8", "float8"), ("int8", "float16"),
     ("int8", "int8")])
def test_matmul(lhs_dtype, lhs_row_major, rhs_dtype, rhs_row_major):
    if (lhs_row_major and not rhs_row_major and lhs_dtype == rhs_dtype and
          (lhs_dtype == "float8" or lhs_dtype == "int8")):
        pytest.skip("Fails because of a known issue: "
            "https://github.com/openai/triton/issues/1397#issuecomment-1564409407")

    def matmul_reference(a, b):
        return torch.matmul(a, b)

    def f8_to_f16(x):
        assert x.is_contiguous(), "Kernel only works for contiguous tensors"
        @triton.jit
        def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            x = tl.load(X + offs, mask=mask)
            y = x.to(tl.float16)
            tl.store(Y + offs, y, mask=mask)
        ret = torch.empty(x.shape, dtype=torch.float16, device=x.device)
        grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
        kernel[grid](ret, triton.reinterpret(x, tl.float8e5), ret.numel(), BLOCK_SIZE=1024)
        return ret

    def f16_to_f8(x):
        assert x.is_contiguous(), "Kernel only works for contiguous tensors"
        @triton.jit
        def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            x = tl.load(X + offs, mask=mask)
            y = x.to(tl.float8e5)
            tl.store(Y + offs, y, mask=mask)
        ret = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
        kernel[grid](triton.reinterpret(ret, tl.float8e5), x, x.numel(), BLOCK_SIZE=1024)
        return ret

    @triton.jit
    def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        IS_LHS_FP8: tl.constexpr,
        IS_RHS_FP8: tl.constexpr,
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
        # See above `Pointer Arithmetics` section for details
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
            if IS_LHS_FP8:
              a = a.to(tl.float8e5, bitcast=True)
            a = a.to(tl.float16)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            if IS_RHS_FP8:
              b = b.to(tl.float8e5, bitcast=True)
            b = b.to(tl.float16)
            # We accumulate along the K dimension.
            accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        c = accumulator.to(tl.float16)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    m, n, k = 32, 64, 128
    block_m, block_n, block_k = 16, 32, 64
    group_size_m = 8

    def get_input(n, m, dtype, row_major):
      if not row_major:
        return get_input(m, n, dtype, row_major=True).T

      res = torch.randn((n, m), device='cuda', dtype=torch.float16)
      if dtype == "int8":
        res = res.to(torch.int8)
      elif dtype == "float8":
        res = f16_to_f8(res)
      return res

    torch.manual_seed(0)
    a = get_input(m, k, lhs_dtype, lhs_row_major)
    b = get_input(k, n, rhs_dtype, rhs_row_major)

    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    triton_output = torch.empty((m, n), device=a.device, dtype=torch.float16)
    matmul_kernel[grid](
        a, b, triton_output,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        triton_output.stride(0), triton_output.stride(1),
        block_m, block_n, block_k,
        group_size_m,
        IS_LHS_FP8=(lhs_dtype == "float8"),
        IS_RHS_FP8=(rhs_dtype == "float8")
    )

    def maybe_upcast(x, dtype):
      if not x.is_contiguous():
          return maybe_upcast(x.T, dtype).T

      if dtype == "float8":
          return f8_to_f16(x)
      else:
          return x.to(torch.float16)

    torch_output = torch.matmul(maybe_upcast(a, lhs_dtype), maybe_upcast(b, rhs_dtype))
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2)


def test_chained_matmul():
    # Regression test for issue #1601
    def chained_matmul_reference(a, b, c):
        intermediate = torch.einsum('MK,NK->MN', a, b)
        return torch.einsum('MN,NK->MK', intermediate, c)

    @triton.jit
    def chained_matmul_kernel(
            A,  # shape: (m, k)
            B,  # shape: (n, k)
            C,  # shape: (n, k)
            out,  # shape: (m, k)
            m, n, k: tl.constexpr,
            block_m: tl.constexpr,
            block_n: tl.constexpr,
            block_k: tl.constexpr):

        tl.static_assert(block_k == k,
                         f"expected block_k == k but got {block_k} != {k}")

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

    grid = (triton.cdiv(m, block_m),)
    a = torch.randint(low=0, high=2, size=(m, k), dtype=torch.float16,
                      device='cuda')
    b = torch.randint(low=0, high=2, size=(n, k), dtype=torch.float16,
                      device='cuda')
    c = torch.randint_like(b, low=0, high=2)
    triton_result = torch.zeros_like(a)

    torch_result = chained_matmul_reference(a, b, c)
    chained_matmul_kernel[grid](a, b, c, triton_result, m, n, k,
                                block_m=block_m, block_n=block_n,
                                block_k=block_k)

    assert (torch_result == triton_result).all()


def test_vecmat():
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
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr
    ):
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

    A_tri = torch.tensor(A, device='cuda')
    B_tri = torch.tensor(B, device='cuda')
    C_tri = torch.zeros((M, N), dtype=torch.float32, device='cuda')

    grid = (M // block_m, N // block_n)

    batched_vecmat[grid](A_tri, B_tri, M, N, K, C_tri,
                         block_m=block_m, block_n=block_n, block_k=block_k,
                         num_warps=4, num_stages=1)

    A_expanded = A[:, np.newaxis, :]
    A_broadcasted = np.broadcast_to(A_expanded, (M, N, K))
    AB = A_broadcasted * B
    C_ref = np.sum(AB, axis=2)

    np.testing.assert_allclose(C_ref, C_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


@pytest.mark.parametrize("type", ["pre_load", "post_load", "post_pre_mixed", "post_load_two_iters", "post_load_three_iters"])
def test_iv_dependent_matmul(type):
    @triton.jit
    def kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        type: tl.constexpr
    ):
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

    a = torch.rand((M, K), device='cuda')
    b = torch.rand((K, N), device='cuda')

    torch_output = torch.mm(a, b)
    triton_output = torch.empty_like(
        torch_output, device=torch_output.device)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    num_stages = 4 if type == "post_load_three_iters" else 3
    kernel[grid](a, b, triton_output, M, N, K, a.stride(0), a.stride(1),
                 b.stride(0), b.stride(1), triton_output.stride(0), triton_output.stride(1),
                 BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                 type=type, num_stages=num_stages)
    torch.testing.assert_allclose(torch_output, triton_output, rtol=1e-2, atol=1e-2)
