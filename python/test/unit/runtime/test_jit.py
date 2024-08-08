import itertools
import pytest
import torch

import triton
import triton.language as tl


def test_pre_call_hooks(device):

    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    class MyTensor(torch.Tensor):
        pass

    def my_hook(kernel, *args, **kwargs):
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, MyTensor):
                raise Exception("MyTensor is not allowed")

    add_kernel.add_pre_run_hook(my_hook)

    x = torch.randn(4, device=device)
    y = MyTensor(x)
    out = torch.zeros_like(x)
    with pytest.raises(Exception):
        add_kernel[(4, )](x, y, out, 4, 4)


def test_pre_call_hooks_can_stop_execution(device):
    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    called = 0

    def my_hook(kernel, *args, **kwargs):
        nonlocal called
        called += 1
        return True

    add_kernel.add_pre_run_hook(my_hook)
    x = torch.ones(4, device=device)
    y = torch.ones(4, device=device)
    out = torch.zeros_like(x)
    add_kernel[(4, )](x, y, out, 4, 4)
    assert called == 1
    assert torch.allclose(out, torch.zeros_like(out))


def test_pre_call_hooks_can_stop_execution_autotuned(device):
    N = 1024
    src = torch.empty(N, device=device)
    dst = torch.empty(N, device=device)
    dst_clone = dst.clone()

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.autotune(configs=configs, key=['N'], warmup=1, rep=1)
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    called = 0

    def my_hook(kernel, *args, grid, warmup, **kwargs):
        nonlocal called
        called += 1
        return True

    _kernel.add_pre_run_hook(my_hook)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel[grid](dst, src, N)

    assert called == 1
    assert torch.allclose(dst, torch.zeros_like(dst_clone))
