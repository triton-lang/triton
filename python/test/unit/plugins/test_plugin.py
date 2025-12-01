import torch

import pytest
import os

import triton
import triton.language as tl
from triton import knobs
import custom_stages


@pytest.mark.parametrize(None, [None])
@triton.jit
def kernel1(BLOCK_SIZE: tl.constexpr):
    return


@pytest.mark.parametrize(None, [None])
@triton.jit
def kernel2(BLOCK_SIZE: tl.constexpr):
    return


def test_op(capfd, device: str):
    if os.environ.get('LLVM_BUILD_SHARED_LIBS', '0') == '0':
        return

    size = 98432
    x = torch.rand(size, device=device)
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    h = kernel1[grid](BLOCK_SIZE=1024)
    assert "tt.func public @foo" not in h.asm["ttgir"]

    knobs.runtime.add_stages_inspection_hook = custom_stages.inspect_stages_hook
    h = kernel2[grid](BLOCK_SIZE=1024)
    assert "tt.func public @foo" in h.asm["ttgir"]

    knobs.runtime.add_stages_inspection_hook = None
    h = kernel2[grid](BLOCK_SIZE=1024)
    assert "tt.func public @foo" not in h.asm["ttgir"]
