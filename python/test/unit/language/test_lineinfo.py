import triton
import triton.language as tl
import tempfile
import subprocess
import pytest

import torch


@triton.jit
def kernel_single(X,
                  Y,
                  BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def device_inline(x):
    return x + x


@triton.jit
def kernel_call(X,
                Y,
                BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = device_inline(x)
    tl.store(Y + tl.arange(0, BLOCK), y)


@triton.jit(noinline=True)
def device_noinline(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = x + x
    tl.store(Y + tl.arange(0, BLOCK), y)


@triton.jit
def kernel_call_noinline(X, Y, BLOCK: tl.constexpr):
    device_noinline(X, Y, BLOCK)


func_types = ["single", "call", "call_noinline"]


@pytest.mark.parametrize("func", func_types)
def test_line_info(func: str):
    try:
        subprocess.check_output(["nvdisasm", "-h"])
    except BaseException:
        pytest.skip("nvdisasm is not available")
    shape = (128, )
    x = torch.arange(0, shape[0], dtype=torch.int32, device='cuda')
    y = torch.zeros(shape, dtype=x.dtype, device="cuda")
    kernel_info = {}
    if func == "single":
        kernel_info = kernel_single[(1,)](x, y, BLOCK=shape[0])
    elif func == "call":
        kernel_info = kernel_call[(1,)](x, y, BLOCK=shape[0])
    elif func == "call_noinline":
        kernel_info = kernel_call_noinline[(1,)](x, y, BLOCK=shape[0])

    fd, path = tempfile.mkstemp()
    with open(fd, 'wb') as cubin:
        cubin.write(kernel_info.asm['cubin'])
    # Check if nvdisasm is available
    subprocess.check_output(["nvdisasm", "-h"])
    asm = subprocess.check_output(["nvdisasm", "-g", path])
    if func == "single":
        assert "kernel_single" in str(asm)
    elif func == "call":
        assert "kernel_call" in str(asm)
    elif func == "call_noinline":
        assert "kernel_call_noinline" in str(asm)
