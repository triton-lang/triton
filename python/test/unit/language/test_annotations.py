from __future__ import annotations

import pytest
import torch

import triton
import triton.language as tl

@pytest.mark.xfail("From future import annotations causes tl.constexpr annotations to fail")
def test_from_future_annotations():
    N = 1024
    src = torch.empty(N, device='cuda')
    dst = torch.empty(N, device='cuda')

    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)
    _kernel[(1,)](dst=dst, src=src, N=N, BLOCK_SIZE=32)
