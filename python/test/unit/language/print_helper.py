import sys
import uuid

import torch
from torch.testing import assert_close

import triton
import triton.language as tl


def get_current_target_warp_size():
    return triton.runtime.driver.active.get_current_target().warp_size


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
def kernel_device_print_scalar(SCALAR):
    x = tl.load(SCALAR)
    # Triton should add a space after this prefix.
    print("x:", x)


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


@triton.jit
def kernel_print_pointer(X, Y, BLOCK: tl.constexpr):
    tl.device_print("ptr ", X + tl.arange(0, BLOCK))


def test_print(func: str, data_type: str, device: str):
    N = 128  # This value should match with test_print in test_subprocess.py.
    # TODO(antiagainst): Currently the warp count is chosen to make sure wedon't have multiple
    # threads printing duplicated messages due to broadcasting. Improve print op lowering logic
    # to filter out duplicated data range.
    num_warps = N // get_current_target_warp_size()

    x = torch.arange(0, N, dtype=torch.int32, device=device).to(getattr(torch, data_type))
    y = torch.zeros((N, ), dtype=x.dtype, device=device)
    if func == "device_print":
        kernel_device_print[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_scalar":
        scalar = torch.tensor(42, dtype=x.dtype, device="cuda")
        kernel_device_print_scalar[(1, )](scalar, num_warps=num_warps)
    elif func == "device_print_negative":
        x = -x
        kernel_device_print[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_uint":
        x = torch.arange((1 << 31), (1 << 31) + N, device=device).to(getattr(torch, data_type))
        kernel_device_print[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "print":
        kernel_print[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_large":
        kernel_device_print_large[(1, 2)](BLOCK_M=64, num_warps=num_warps, BLOCK_N=N)
    elif func == "print_multiple_args":
        kernel_print_multiple_args[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_multiple_args":
        kernel_device_print_multiple_args[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "static_print":
        kernel_static_print[(1, )](x, y, num_warps=num_warps, BLOCK=N, PLACEHOLDER=uuid.uuid4())
    elif func == "no_arg_print":
        kernel_no_arg_print[(1, )](num_warps=num_warps)
    elif func == "print_no_arg":
        kernel_print_no_arg[(1, )](num_warps=num_warps)
    elif func == "device_print_hex":
        kernel_device_print_hex[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_pointer":
        kernel_print_pointer[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    else:
        assert f"Unknown kernel: {func}"

    if func != "print_no_arg" and func != "no_arg_print" and func != "device_print_large" and \
       func != "print_multiple_args" and func != "device_print_multiple_args" and \
       func != "device_print_pointer" and func != "device_print_scalar":
        assert_close(y, x)


if __name__ == "__main__":
    fn = globals()[sys.argv[1]]
    fn(*sys.argv[2:])
