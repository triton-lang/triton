import torch

import pytest
import os

import triton
import triton.language as tl
from triton import knobs
import custom_stages


@pytest.mark.parametrize(None, [None])
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(output_ptr + offsets, output)


def test_op(capfd, device: str):
    if os.environ.get('LLVM_BUILD_SHARED_LIBS', '0') == '0':
        return
    os.environ['TRITON_ALWAYS_COMPILE'] = '1'

    size = 1024
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(output.numel(), meta['BLOCK_SIZE']), )

    knobs.runtime.add_stages_inspection_hook = custom_stages.inspect_stages_hook_dialect
    h = add_kernel[grid](x, y, output, BLOCK_SIZE=1024)

    if os.environ.get('TRITON_KERNEL_OVERRIDE', '0') == '0':
        return

    assert 'plugin.magic' in h.asm["ttir"]
    assert 'gpu.thread_id  x' in h.asm["ttgir"]
