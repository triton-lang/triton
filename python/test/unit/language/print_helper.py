import sys

import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def kernel_device_print(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.device_print("", x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_print(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    print("", x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_static_print(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.static_print(x)
    tl.store(Y + tl.arange(0, BLOCK), x)


def test_print(func: str, data_type: str):
    shape = (128, )
    # limit the range of integers so that the sum does not overflow
    x = torch.arange(0, shape[0], dtype=torch.int32, device='cuda').to(getattr(torch, data_type))
    y = torch.zeros(shape, dtype=x.dtype, device="cuda")
    if func == "device_print":
        kernel_device_print[(1,)](x, y, BLOCK=shape[0])
    elif func == "print":
        kernel_print[(1,)](x, y, BLOCK=shape[0])
    elif func == "static_print":
        kernel_static_print[(1,)](x, y, BLOCK=shape[0])
    assert_close(y, x)


if __name__ == "__main__":
    test_print(sys.argv[1], sys.argv[2])
