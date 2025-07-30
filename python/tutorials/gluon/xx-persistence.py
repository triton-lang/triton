"""
Persistent Kernels
==================

So far, we have defined kernels such that one programs handles one block of work
and we span all the work using the grid dimensions. This creates a large number
of programs, and we rely on the GPU to schedule the work. The primary benefit is
the GPU will dynamically load-balance the work across its SMs.

However, this approach has downsides. The scheduler incurs an overhead, and the
GPU is not aware of the memory access patterns of the kernels. This also
prevents overlapping across blocks of work, as the GPU waits until kernels have
fully exited before issuing more work.

In this tutorial, we will implement persistent kernels and explore performance
aspects like load balancing and data locality.
"""

import pytest
import torch
import importlib
import os
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# %%
# In the last tutorial, we implemented memcpy. An input shape of (32K, 64K)
# and XBLOCK=YBLOCK=128 yields grind dimensions of (256, 512). That's a lot of
# CTAs to schedule!
#
# By locking the grid size to the number of SMs, we launch one kernel per SM
# among which we statically distribute the work. This way, the kernels "persist"
# until all the work is done.

# Re-use code from the last tutorial.
if "PYTEST_VERSION" in os.environ:
    utils = importlib.import_module(".02-layouts", package="gluon")
else:
    utils = importlib.import_module("02-layouts")


@gluon.jit
def memcpy_2d_persistent_kernel(in_ptr, out_ptr,  #
                                xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
                                layout_in: gl.constexpr, layout_out: gl.constexpr,  #
                                XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, NUM_SMS: gl.constexpr):
    sm_id = gl.program_id(0)

    # Compute the amount of work assigned to each SM.
    num_pid_x = gl.cdiv(xnumel, XBLOCK)
    num_pid_y = gl.cdiv(ynumel, YBLOCK)
    num_pid = num_pid_x * num_pid_y
    pid_per_sm = gl.cdiv(num_pid, NUM_SMS)

    pid_start = sm_id * pid_per_sm
    pid_end = min(pid_start + pid_per_sm, num_pid)

    # The persistent outer loop iterates over the available work.
    for pid in range(pid_start, pid_end):
        # Map the linearized pid to a specific block along the inner dimension.
        pid_y = pid % num_pid_y
        pid_x = pid // num_pid_y

        # The inner loop looks the same as the non-persistent memcpy.
        start_x = pid_x * XBLOCK
        start_y = pid_y * YBLOCK
        mask_in, in_offsets = utils.get_mask_and_offsets(start_x, start_y, xnumel, ynumel,  #
                                                         xstride_in, ystride_in, XBLOCK, YBLOCK, layout_in)
        mask_out, out_offsets = utils.get_mask_and_offsets(start_x, start_y, xnumel, ynumel,  #
                                                           xstride_out, ystride_out, XBLOCK, YBLOCK, layout_out)

        value = gl.load(in_ptr + in_offsets, mask=mask_in)
        value = gl.convert_layout(value, layout_out)
        gl.store(out_ptr + out_offsets, value, mask=mask_out)


def memcpy_2d_persistent(input, output, num_warps=4, maxnreg=None, occupancy=1):
    assert input.shape == output.shape, "input and output must have the same shape"
    XBLOCK = 128
    YBLOCK = 128
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count * occupancy

    layout_in = utils.get_layout_for_gmem_access(input, num_warps)
    layout_out = utils.get_layout_for_gmem_access(output, num_warps)
    grid = (NUM_SMS, )
    return memcpy_2d_persistent_kernel[grid](  #
        input, output,  #
        input.shape[0], input.shape[1],  #
        *input.stride(), *output.stride(),  #
        layout_in, layout_out,  #
        XBLOCK, YBLOCK, NUM_SMS, num_warps=num_warps, maxnreg=maxnreg)


@pytest.mark.parametrize("xnumel, ynumel", [(400, 500)])
@pytest.mark.parametrize("transpose_in, transpose_out", [(True, False), (False, True)])
def test_memcpy_2d_persistent(xnumel, ynumel, transpose_in, transpose_out):
    torch.manual_seed(0)
    if transpose_in:
        input = torch.randn((ynumel, xnumel), device="cuda").T
    else:
        input = torch.randn((xnumel, ynumel), device="cuda")
    if transpose_out:
        output = torch.empty((ynumel, xnumel), device="cuda").T
    else:
        output = torch.empty((xnumel, ynumel), device="cuda")
    memcpy_2d_persistent(input, output)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


if __name__ == "__main__":
    print("Benchmarking persistent memcpy")
    print("==============================")
    input = torch.randn((32 * 1024, 64 * 1024), device="cuda")
    output = torch.empty_like(input)
    k, throughput = utils.bench_memcpy_impl(input, output, memcpy_2d_persistent)
    print(f"Persistent memcpy: {throughput:.3f} TB/s")
    _, throughput = utils.bench_memcpy_impl(input, output, utils.memcpy_2d_inout)
    print(f"Non-persistent memcpy: {throughput:.3f} TB/s")

# %%
# We can compare our persistent kernel with the non-persistent version from the
# previous tutorial.
#
# ```
# Persistent memcpy: 2.216 TB/s
# Non-persistent memcpy: 2.854 TB/s
# ```
#
# It's slower! Running both kernels through the `ncu` profiler, we see this:
#
# | kernel         | active warps per SM | occupancy     | registers/thread |
# |----------------|---------------------|---------------|------------------|
# | persistent     |  3.99 /  8          |  6.23 / 12.50 | 174              |
# | non-persistent | 11.74 / 12          | 17.84 / 18.75 | 168              |
#
# The non-persistent kernel has higher theoretical occupancy and gets closer to
# it. High occupancy allows the SM to swap between CTAs to hide instruction
# latency, almost like CPU hyper-threading. This is especially important because
# `LGD.E` and `STG.E` are synchronous instructions.
#
# The problem is we schedule exactly `NUM_SMS` CTAs, each with 4 warps. Since
# at least Ampere, SMs can process 4 warps in parallel. That means when 1
# persistent warp stalls, there is nothing to switch to.
#
# The non-persistent kernel happens to use 168 registers per thread, which is
# maximum to achieve 3 CTAs per SM. We can manually limit the registers of the
# persistent kernel. It is unlikely to trigger spilling as the compiler can find
# 6 registers by reordering instructions. Then, we will schedule 3 times as many
# persistent kernels.

if __name__ == "__main__":
    impl = partial(memcpy_2d_persistent, maxnreg=168, occupancy=900)
    _, throughput = utils.bench_memcpy_impl(input, output, impl)
    print(f"Persistent (3x) memcpy: {throughput:.3f} TB/s")
