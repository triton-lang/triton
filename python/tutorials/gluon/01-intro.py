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

import torch
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


def test_kernel():
    input = torch.tensor([42.0], device="cuda")
    output = torch.empty_like(input)
    copy_scalar(input, output)
    torch.testing.assert_close(input, output, atol=0, rtol=0)
