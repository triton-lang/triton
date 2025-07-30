"""
Warp-Group MMA
==============

WGMMA (also known as MMAv3) is a Hopper-specific instruction for performing
matrix multiply-accumulate operations using the Tensor Cores. WGMMA instructions
are asynchronous, meaning they can be pipelined.

In this tutorial, we will cover how to use WGMMAs in Gluon. We will build a
simple matmul kernel to demonstrate practical uses of WGMMA, and show an example
where WGMMAs can be pipelined for better performance.
"""

import pytest
import torch
import triton
import importlib
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
)


def is_hopper():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


if __name__ == "__main__" and not is_hopper():
    raise RuntimeError("This tutorial requires a Hopper NVIDIA GPU")

# %%
# Let's illustrate WGMMA with a trivial example. This kernel will be launched
# with grid size (1, ) and it performs MMA on a small matrix.
