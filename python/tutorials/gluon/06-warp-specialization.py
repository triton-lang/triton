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

# import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier  # , fence_async_shared


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")

# %%
# Let's revisit our elementwise add kernel and implement a warp-specialized
# version. In a warp-specialized kernel, groups of warps that perform a specific
# task are called "partitions", and each can have a different number of warps
# and registers.
#
# First, we need to decide what the partitions will be and how many registers
# they will get. One of the benefits of warp specialization is that partitions
# that don't have tensor values only need to use 1 warp and often very few
# registers. For example, we can have one partition that just issues async TMA
# loads and one partition that just issues TMA stores, each with 1 warp and 24,
# the minimum number of registers we can assign to a warp.
#
# Then we have one compute partition, with either 4 or 8 warps, which performs
# the vector addition. Estimating the right register allocation is difficult,
# and often involves trial and error, profiling, and autotuning. We will need to
# use mbarriers to signal between the partitions using producer-consumer pairs.
#
# To write a warp-specialized kernel, we need to write a separate function for
# each partition. One of the partitions must be chosen as the "default"
# partition and it always has the same number of warps as `num_warps` passed to
# the kernel. The other partitions, i.e. the "worker" partitions, can have
# different numbers of warps. The signature of the worker partition functions
# must all be the same. Only the default partition can accept tensor arguments.
#
# Quickly sketch out the partitions: load partition will fetch inputs to smem
# and signal the compute partition. The compute partition will consume the
# operands and send them to the store partition over smem.


@gluon.jit
def load_partition(descs, barriers, buffers, xoff, numel, YBLOCK: gl.constexpr):
    # Unpack the arguments.
    a_desc, b_desc, c_desc = descs
    load_empty_bars, load_ready_bars, c_empty_bar, c_ready_bar = barriers
    a_bufs, b_bufs, c_bufs = buffers
    xnumel, ynumel = numel

    num_buffers: gl.constexpr = a_bufs.type.shape[0]

    # All the partitions need to have the same number of inner loop iterations.
    for i in range(gl.cdiv(ynumel, YBLOCK)):
        index = i % num_buffers
        phase = i // num_buffers & 1
        a_buf = a_bufs.index(index)
        b_buf = b_bufs.index(index)
        load_empty_bar = load_empty_bars.index(index)
        load_ready_bar = load_ready_bars.index(index)

        # Wait for the current buffers to be empty. Recall that mbarriers are
        # initialized to phase 1 complete, so we wait starting with phase 1 to
        # allow the producer to begin filling the pipeline.
        mbarrier.wait(load_empty_bar, phase ^ 1)

        # Okay, a_buf and b_buf are empty. Issue the TMA loads, and have them
        # signal the operand buffers as ready when they complete.
        yoff = i * YBLOCK
        mbarrier.expect(load_ready_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_local(a_desc, [xoff, yoff], load_ready_bar, a_buf)
        tma.async_copy_global_to_local(b_desc, [xoff, yoff], load_ready_bar, b_buf)


@gluon.jit
def store_partition(descs, barriers, buffers, xoff, numel,  #
                    YBLOCK: gl.constexpr):
    a_desc, b_desc, c_desc = descs
    load_empty_bars, load_ready_bars, c_empty_bar, c_ready_bar = barriers
    a_bufs, b_bufs, c_bufs = buffers
    xnumel, ynumel = numel

    # This partition consumes the addition result, passed over smem, and stores
    # them to global memory.
    # num_buffers: gl.constexpr = c_bufs.type.shape[0]

    # Because `tma.store_wait` accepts a constexpr


@gluon.jit
def elementwise_add_warp_specialized(  #
        a_desc, b_desc, c_desc,  #
        xnumel, ynumel, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pass
