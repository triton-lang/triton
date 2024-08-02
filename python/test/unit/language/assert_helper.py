import sys

import torch
from torch.testing import assert_close

import triton
import triton.language as tl


def get_current_target_warp_size():
    return triton.runtime.driver.active.get_current_target().warp_size


@triton.jit
def kernel_device_assert(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.device_assert(x == 0, "x != 0")
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_assert_passes(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    # Trivial assert, should not be an error.
    tl.device_assert(0 == 0, "x != 0")
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit(debug=False)
def kernel_device_assert_no_debug(X, Y, BLOCK: tl.constexpr):
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


def test_assert(func: str, device: str):
    N = 128  # This value should match with test_print in test_subprocess.py.
    num_warps = N // get_current_target_warp_size()

    x = torch.arange(0, N, dtype=torch.int32, device='cuda')
    y = torch.zeros((N, ), dtype=x.dtype, device="cuda")
    if func == "device_assert":
        kernel_device_assert[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    if func == "device_assert_passes":
        # Assert passes; no error.
        kernel_assert_passes[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "no_debug":
        # TRITON_DEBUG=1 can override the debug flag
        kernel_device_assert_no_debug[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "assert":
        kernel_assert[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "static_assert":
        kernel_static_assert[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "double_assert":
        # Launching a different kernel after the first one asserted used to
        # segfault.  What seems to have happened is:
        #  - The first kernel is enqueued but doesn't run yet.
        #  - We go to launch the second kernel.  Because this is the first time
        #    we're running it, we have to load the kernel into the GPU.
        #  - Loading the kernel takes some time, during which the first launch
        #    completes.
        #  - Now the GPU is in an error state.  We need to detect this inside
        #    the kernel-launch/loading code and bail out properly.  If we don't,
        #    we segfault.
        kernel_device_assert[(1, )](x, y, num_warps=num_warps, BLOCK=N)
        kernel_assert_passes[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    assert_close(y, x)
    # To guarantee driver completes all the jobs for
    torch.cuda.synchronize()


@triton.jit
def jit_device_assert_none(x):
    tl.device_assert(x == 0, "x != 0")


@triton.jit(debug=True)
def jit_device_assert_true(x):
    tl.device_assert(x == 0, "x != 0")


@triton.jit(debug=False)
def jit_device_assert_false(x):
    tl.device_assert(x == 0, "x != 0")


@triton.jit
def kernel_device_assert_nested(X, Y, BLOCK: tl.constexpr, jit_debug: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    if jit_debug == "true":
        jit_device_assert_true(x)
    elif jit_debug == "false":
        jit_device_assert_false(x)
    else:
        jit_device_assert_none(x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit(debug=True)
def kernel_device_assert_nested_true(X, Y, BLOCK: tl.constexpr, jit_debug: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    if jit_debug == "true":
        jit_device_assert_true(x)
    elif jit_debug == "false":
        jit_device_assert_false(x)
    else:
        jit_device_assert_none(x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit(debug=False)
def kernel_device_assert_nested_false(X, Y, BLOCK: tl.constexpr, jit_debug: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    if jit_debug == "true":
        jit_device_assert_true(x)
    elif jit_debug == "false":
        jit_device_assert_false(x)
    else:
        jit_device_assert_none(x)
    tl.store(Y + tl.arange(0, BLOCK), x)


def test_assert_nested(caller: str, callee: str, device: str):
    N = 128  # This value should match with test_print in test_subprocess.py.
    num_warps = N // get_current_target_warp_size()

    x = torch.arange(0, N, dtype=torch.int32, device=device)
    y = torch.zeros((N, ), dtype=x.dtype, device=device)
    if caller == "none":
        kernel_device_assert_nested[(1, )](x, y, num_warps=num_warps, BLOCK=N, jit_debug=callee)
    elif caller == "true":
        kernel_device_assert_nested_true[(1, )](x, y, num_warps=num_warps, BLOCK=N, jit_debug=callee)
    elif caller == "false":
        kernel_device_assert_nested_false[(1, )](x, y, num_warps=num_warps, BLOCK=N, jit_debug=callee)
    assert_close(y, x)


if __name__ == "__main__":
    fn = globals()[sys.argv[1]]
    fn(*sys.argv[2:])
