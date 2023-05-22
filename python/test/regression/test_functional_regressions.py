import torch

import triton
import triton.language as tl


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
