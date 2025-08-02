"""
Warp Specialization
===================

This tutorial covers warp specialization. In typical GPU kernels, all warps are
executing cooperatively, meaning they perform parts of the same task. Warp
specialization, however, is a technique where different warps in the kernel are
doing completely different tasks.

Warp specialization is typically used to overlap execution of different parts
of the kernel. This is useful overlapping async operations with finer
granularity than software pipelining, and we can overlap non-async operations
that exercise different parts of the hardware without relying on precise
SASS-level instruction interleaving.

However, warp specialization comes at the cost of additional synchronization
overhead, potentially higher shared memory usage for communicating data, and
higher overall register pressure.

Warp specialization in Gluon is only supported on Hopper and newer GPUs.
"""

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared

# %%
# Let's revisit
