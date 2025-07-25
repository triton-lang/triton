"""
Introduction to Gluon
=====================

Gluon is a GPU programming language based on the same compiler stack as Triton.
But compared to Triton, Gluon is a lower-level language that gives the user more
control and responsibility when implementing kernels.

In this tutorial series, we will cover everything from the basics of writing
GPU kernels in Gluon, modern (NVIDIA) GPU hardware features, to advanced
performance optimization techniques. And at the end, we will use what we have
learned to assemble an efficient GEMM (General Matrix Multiply) kernel. These
tutorials will assume that you already have basic familiarity with Triton.

At a high level, Gluon and Triton share many similarities. Both implement
a tile-based SPMD (Single Program Multiple Data) programming model, where a tile
represents an N-dimensional array whose elements are distributed over a single
"program". Both represent computations as operations over these tiles. And both
are embedded Python DSLs that share the same frontend.

Triton, however, is a high-level language that hides many of the details of
implementing a GPU kernel from the user. The specific distribution of tile
elements, i.e. its "layout", is hidden from the user and managed by the
compiler. Similarly, the Triton compiler manages memory, data movement, and
scheduling (asynchronity and synchronization), and it abstracts away the
underlying hardware features.

Getting these details right is important to the performance of the kernel. While
the Triton compiler does a fairly good job of generating efficient code for a
wide range of kernels, it can often fall short. When this happens, there is
little the user can do to significantly improve performance since all the
details are hidden.

In Gluon, all of these details (and more) are exposed to the user. This means
writing Gluon kernels requires a deeper understanding of GPU hardware and the
many aspects of GPU programming, but it also enables writing more performant
kernels by finely controlling these low-level details.
"""

# %%
# We will first go over the basics of defining a Gluon kernel and writing its
# launcher. It is practically the same as defining a Triton kernel. Use the
# `@gluon.jit` decorator to declare a Gluon kernel, and it can be invoked from
# Python using the same interface as a `@triton.jit` kernel.

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# %%
# Let's illustrate this using a trivial kernel. This kernel copies a scalar from
# one point in global memory to another.


@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)


# %%
# Write the launcher. PyTorch tensors are converted to global memory pointers
# when passed to Gluon kernels, much like Triton kernels. And the grid is
# specified in the same way.


def copy_scalar(input, output):
    # Launch a single program.
    grid = (1, )
    copy_scalar_kernel[grid](input, output, num_warps=1)


# %%
# Let's test the kernel. You can run the test with `pytest 01-intro.py`.


def test_copy_scalar():
    input = torch.tensor([42.0], device="cuda")
    output = torch.empty_like(input)
    copy_scalar(input, output)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


# %%
# We can write a kernel with hyperparameters passed as constexpr arguments in
# much the same way as Triton. This is a trivial memcpy kernel implemented by
# subtiling the tensors into 1D blocks, where each program processes one block.


@gluon.jit
def memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    # Each program processes the addresses [pid, pid + BLOCK_X), clamped into
    # the range [0, xnumel).
    pid = gl.program_id(0)
    start = pid * XBLOCK
    end = min(start + XBLOCK, xnumel)
    for i in range(start, end):
        value = gl.load(in_ptr + i)
        gl.store(out_ptr + i, value)


def memcpy(input, output, XBLOCK=128):
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    memcpy_kernel[grid](input, output, xnumel, XBLOCK, num_warps=1)


@pytest.mark.parametrize("XBLOCK", [64])
@pytest.mark.parametrize("xnumel", [40, 500])
def test_memcpy(XBLOCK, xnumel):
    torch.manual_seed(0)
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    memcpy(input, output, XBLOCK)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


# %%
# Gluon hyperparameters can be autotuned like Triton as well. Let's autotune
# `XBLOCK` for the memcpy kernel as an example.


@triton.autotune(
    configs=[triton.Config({"XBLOCK": 2**i}, num_warps=1) for i in range(8, 14)],
    key=["xnumel"],
)
@gluon.jit
def memcpy_kernel_autotune(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK)


def memcpy_autotune(input, output):
    xnumel = input.numel()

    def grid(META):
        return (triton.cdiv(xnumel, META["XBLOCK"]), )

    memcpy_kernel_autotune[grid](input, output, xnumel)


# %%
# Run this with `TRITON_PRINT_AUTOTUNING=1 python 01-intro.py` to see which
# `XBLOCK` gets selected. On B200, the best `XBLOCK` ends up being 2048 to copy
# 4 GB of data at about 332 GB/s, far from the 8 TB/s peak bandwidth of the GPU.
#
# Since performance is the main motiviation for writing kernels in Gluon, let's
# spend some time discussing that. First, the obvious problem is we are not
# fully utilizing the parallelism of the GPU. Each Gluon "program" corresponds
# to a thread block (CTA) on the GPU, and while the GPU can execute many CTAs
# at once, our kernel copies 1 scalar element at a time per CTA.
#
# In order to copy many elements at once, we need to load and store tiles, but
# that will require picking a layout and understanding which layouts perform
# better than others. In the next tutorial, we will cover the basics of layouts
# in Gluon and how they can affect performance.

if __name__ == "__main__":
    torch.manual_seed(0)
    xnumel = 4 << 30
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)

    fn = lambda: memcpy_autotune(input, output)
    ms = triton.testing.do_bench(fn)
    gbytes = xnumel * input.element_size() >> 30
    print("ms  ", round(ms, 2))
    print("GB/s", round(gbytes / (ms * 1e-3), 2))
