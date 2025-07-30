"""
Async Copy in Gluon
===================

Modern GPUs provide asynchronous instructions for long-running operations like
global memory reads and writes. Asynchronous operations allow overlapping memory
transactions with compute, also known as "pipelining".

Asynchronous instructions vary by GPU vendor and architecture, so this tutorial
focuses on NVIDIA GPUs. On NVIDIA GPUs, async copies transfer data between
global memory and shared memory, unlike `ld.global` and `st.global` which
directly write to and read from the register file.
"""

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

# %%
# Let's reimplement the 1D memcpy using `cp.async`. Since the memcpy does not
# perform compute work, we can keep the data in shared memory.


@gluon.jit
def memcpy_1d_cpasync_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)

    mask = offsets < xnumel
    in_ptrs = in_ptr + offsets
