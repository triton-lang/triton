import sys

import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def kernel_device_assert(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.device_assert(x == 0, "x != 0")
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_assert(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    assert x == 0, "x != 0"
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_static_assert(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.static_assert(BLOCK == 128, "BLOCK != 128")
    tl.store(Y + tl.arange(0, BLOCK), x)


def test_assert(func: str):
    shape = (128, )
    x = torch.arange(0, shape[0], dtype=torch.int32, device='cuda')
    y = torch.zeros(shape, dtype=x.dtype, device="cuda")
    if func == "device_assert":
        kernel_device_assert[(1,)](x, y, BLOCK=shape[0])
    elif func == "assert":
        kernel_assert[(1,)](x, y, BLOCK=shape[0])
    elif func == "static_assert":
        kernel_static_assert[(1,)](x, y, BLOCK=shape[0])
    assert_close(y, x)


if __name__ == "__main__":
    test_assert(sys.argv[1])
