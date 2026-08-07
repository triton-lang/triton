"""
Regression test for issue #10229: pipeliner crashes when TensormapCreateOp
is inside a loop with tl.dot and cannot be predicated by the pipeliner.

The minimal trigger is: a single-level loop that contains both
tl.make_tensor_descriptor and tl.dot. The pipeliner tries to predicate
TensormapCreateOp in prologue/epilogue phases, crashing without a fallback.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer, is_cuda
from typing import Optional


@pytest.mark.interpreter
@pytest.mark.skipif(not is_cuda(), reason="Requires NVIDIA GPU with TMA support")
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper (sm90+) or newer for TMA")
def test_tensormap_create_in_loop_with_dot(device):
    """Minimal reproducer for #10229: descriptor creation inside a single-level
    loop with tl.dot. The pipeliner must predicate TensormapCreateOp in
    prologue/kernel/epilogue phases. Without a fallback, it crashes."""

    BLOCK_M: tl.constexpr = 16
    BLOCK_N: tl.constexpr = 64

    @triton.jit
    def kernel(
        out_ptr,
        a_ptr,
        b_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        m_off = pid_m * BLOCK_M

        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[K, N],
            strides=[N, 1],
            block_shape=[BLOCK_K, BLOCK_N],
        )

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            # Create a new descriptor each iteration. This triggers #10229
            # because the pipeliner encounters TensormapCreateOp inside the
            # loop body and needs a fallback to predicate it.
            a_desc = tl.make_tensor_descriptor(
                a_ptr + m_off * K + k,
                shape=[BLOCK_M, BLOCK_K],
                strides=[K, 1],
                block_shape=[BLOCK_M, BLOCK_K],
            )
            a = a_desc.load([0, 0]).to(tl.float32)
            b = b_desc.load([k, 0]).to(tl.float32)
            acc += tl.dot(a, b)

        out_offs = (
            m_off * N
            + tl.arange(0, BLOCK_M)[:, None] * N
            + tl.arange(0, BLOCK_N)[None, :]
        )
        mask = (
            (m_off + tl.arange(0, BLOCK_M)[:, None] < M)
            & (tl.arange(0, BLOCK_N)[None, :] < N)
        )
        tl.store(out_ptr + out_offs, acc.to(tl.float16), mask=mask)

    M, N, K = 32, 64, 64
    BLOCK_K: tl.constexpr = 16
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C_ref = torch.mm(A.float(), B.float()).to(torch.float16)
    C = torch.zeros((M, N), dtype=torch.float16, device=device)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)

    grid = (triton.cdiv(M, BLOCK_M),)
    kernel[grid](
        C, A, B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
    )

    torch.testing.assert_close(C.float(), C_ref.float(), rtol=1e-2, atol=1e-2)
