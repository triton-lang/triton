"""
Regression test for issue #10229: pipeliner crashes when TensormapCreateOp
is inside a loop with tl.dot, because TensormapCreateOp did not implement
PredicatedOpInterface.

The minimal trigger is: a single-level loop that contains both
tl.make_tensor_descriptor and tl.dot. The pipeliner tries to predicate
TensormapCreateOp in prologue/epilogue phases, crashing without the interface.
"""

import pytest
import torch
import numpy as np

import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer, is_interpreter, is_cuda
from typing import Optional


@pytest.mark.interpreter
@pytest.mark.skipif(not is_cuda(), reason="Requires NVIDIA GPU with TMA support")
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper (sm90+) or newer for TMA")
def test_tensormap_create_in_loop_with_dot(device):
    """Minimal reproducer for #10229: descriptor creation inside a single-level
    loop with tl.dot. The pipeliner must predicate TensormapCreateOp in
    prologue/kernel/epilogue phases. Without PredicatedOpInterface, it crashes."""

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
            # Create a new descriptor each iteration — this triggers #10229
            # because the pipeliner encounters TensormapCreateOp inside the
            # loop body and needs PredicatedOpInterface to predicate it.
            a_desc = tl.make_tensor_descriptor(
                a_ptr + m_off * K + k,
                shape=[BLOCK_M, BLOCK_K],
                strides=[K, 1],
                block_shape=[BLOCK_M, BLOCK_K],
            )
            a = a_desc.load([0, 0]).to(tl.float32)
            b = b_desc.load([k, 0]).to(tl.float32)
            acc += tl.dot(a, b)

        out_offs = m_off * N + tl.arange(0, BLOCK_M)[:, None] * N + tl.arange(0, BLOCK_N)[None, :]
        mask = (m_off + tl.arange(0, BLOCK_M)[:, None] < M) & (tl.arange(0, BLOCK_N)[None, :] < N)
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


@pytest.mark.interpreter
@pytest.mark.skipif(not is_cuda(), reason="Requires NVIDIA GPU with TMA support")
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper (sm90+) or newer for TMA")
def test_tensormap_create_paged_qk(device):
    """Paged KV cache QK^T computation: per-page descriptor creation inside
    a loop with tl.dot. Each page maps via a page table, and a new descriptor
    is created per page iteration with a different base address."""

    PAGE_SIZE: tl.constexpr = 16
    BLOCK_M: tl.constexpr = 16
    HEAD_SIZE: tl.constexpr = 64
    NUM_PAGES: tl.constexpr = 4

    @triton.jit
    def paged_qk_kernel(
        out_ptr,
        q_ptr,
        k_cache_ptr,
        page_table_ptr,
        stride_page,
        stride_head,
        num_pages: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)

        # Load single Q token
        q = tl.load(q_ptr + tl.arange(0, HEAD_SIZE)).to(tl.float32)
        q_vec = q.reshape(1, HEAD_SIZE)

        acc = tl.zeros([1, BLOCK_M], dtype=tl.float32)

        for page_idx in range(num_pages):
            # Map logical page → physical page
            phys_page = tl.load(page_table_ptr + page_idx)
            base = k_cache_ptr + phys_page * stride_page

            # Create descriptor for this page's K block — triggers #10229
            k_desc = tl.make_tensor_descriptor(
                base,
                shape=[PAGE_SIZE, HEAD_SIZE],
                strides=[stride_head, 1],
                block_shape=[BLOCK_M, HEAD_SIZE],
            )

            K_block = k_desc.load([0, 0]).to(tl.float32)
            K_T = tl.trans(K_block)
            qk = tl.dot(q_vec, K_T)
            acc += qk

        # Store accumulated result for this program
        out_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = out_offsets < num_pages * PAGE_SIZE
        tl.store(out_ptr + out_offsets, acc.reshape(BLOCK_M), mask=mask)

    num_pages = NUM_PAGES
    page_table = torch.tensor([0, 3, 7, 2], dtype=torch.int32, device=device)
    k_cache = torch.randn((8, PAGE_SIZE, HEAD_SIZE), dtype=torch.float16, device=device)
    q = torch.randn((HEAD_SIZE,), dtype=torch.float16, device=device)
    out = torch.zeros((BLOCK_M,), dtype=torch.float32, device=device)

    # Reference: sum of QK^T across all pages (first BLOCK_M tokens per page)
    q_f32 = q.float()
    ref = torch.zeros((BLOCK_M,), dtype=torch.float32, device=device)
    for i in range(num_pages):
        phys = page_table[i].item()
        K_block = k_cache[phys, :BLOCK_M, :].float()  # (BLOCK_M, HEAD_SIZE)
        qk = q_f32 @ K_block.T  # (1, HEAD_SIZE) @ (HEAD_SIZE, BLOCK_M) → (BLOCK_M)
        ref += qk

    stride_page = PAGE_SIZE * HEAD_SIZE
    stride_head = HEAD_SIZE

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)

    paged_qk_kernel[(1,)](
        out, q, k_cache, page_table,
        stride_page, stride_head,
        num_pages, PAGE_SIZE, BLOCK_M, HEAD_SIZE,
    )

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)