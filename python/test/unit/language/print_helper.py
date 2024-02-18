import sys
import uuid

import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def kernel_device_print(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.device_print("x: ", x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_device_print_hex(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.device_print("x: ", x, hex=True)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_print(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    # Triton should add a space after this prefix.
    print("x:", x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_device_print_large(
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    x = tl.full([BLOCK_M, BLOCK_N], 1, tl.int32)
    # Triton should change this prefix to "x: ".
    tl.device_print("x ", x)


@triton.jit
def kernel_print_multiple_args(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = tl.full((BLOCK, ), 1, tl.int32)
    print("", x, y)


@triton.jit
def kernel_device_print_multiple_args(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = tl.full((BLOCK, ), 1, tl.int32)
    tl.device_print("", x, y)
    tl.store(Y + tl.arange(0, BLOCK), y)


@triton.jit
def kernel_static_print(X, Y, BLOCK: tl.constexpr, PLACEHOLDER: tl.constexpr):
    # This function takes an extra value as a tl.constexpr so this kernel is not
    # cached.  This way the static print is run every time.
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.static_print("", x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_no_arg_print():
    print("", tl.program_id(0))


@triton.jit
def kernel_print_no_arg():
    print("no arg")


def test_print(func: str, data_type: str):
    shape = (128, )
    x = torch.arange(0, shape[0], dtype=torch.int32, device='cuda').to(getattr(torch, data_type))
    y = torch.zeros(shape, dtype=x.dtype, device="cuda")
    if func == "device_print":
        kernel_device_print[(1, )](x, y, BLOCK=shape[0])
    elif func == "print":
        kernel_print[(1, )](x, y, BLOCK=shape[0])
    elif func == "device_print_large":
        kernel_device_print_large[(1, 2)](BLOCK_M=64, BLOCK_N=128)
    elif func == "print_multiple_args":
        kernel_print_multiple_args[(1, )](x, y, BLOCK=shape[0])
    elif func == "device_print_multiple_args":
        kernel_device_print_multiple_args[(1, )](x, y, BLOCK=shape[0])
    elif func == "static_print":
        kernel_static_print[(1, )](x, y, BLOCK=shape[0], PLACEHOLDER=uuid.uuid4())
    elif func == "no_arg_print":
        kernel_no_arg_print[(1, )](num_warps=4)
    elif func == "print_no_arg":
        kernel_print_no_arg[(1, )](num_warps=4)
    elif func == "device_print_hex":
        kernel_device_print_hex[(1, )](x, y, BLOCK=shape[0])
    else:
        assert f"Unknown kernel: {func}"

    if func != "print_no_arg" and func != "no_arg_print" and func != "device_print_large" and \
       func != "print_multiple_args" and func != "device_print_multiple_args":
        assert_close(y, x)


if __name__ == "__main__":
    test_print(sys.argv[1], sys.argv[2])
