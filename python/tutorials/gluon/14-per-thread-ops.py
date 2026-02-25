"""
Per-Thread Operations
=====================

In previous tutorials, we covered layouts and how they affect performance. In
this tutorial, we explore a powerful technique enabled by linear layouts:
**per-thread operations**. These are operations that execute entirely within a
single thread's registers, requiring no inter-thread communication (no warp
shuffles, no shared memory).

Per-thread operations are useful in performance-critical inner loops such as
attention softmax, where partial reductions can avoid the cost of full
cross-thread reductions until the final step. They are also the foundation of
per-thread quantization techniques like those used in SageAttention.

The key idea is: by inspecting a layout's linear representation, we can identify
which tensor dimensions are distributed across registers vs. lanes vs. warps.
Reducing along a register-only dimension is entirely local to each thread.

Prerequisites: familiarity with Gluon layouts (Tutorial 02) and basic Gluon
kernel structure (Tutorial 01).
"""

# %%
# Inspecting Layouts with Linear Layouts
# ----------------------------------------
#
# Every Gluon layout can be converted to a ``DistributedLinearLayout``, which
# explicitly lists the bases at each level of the GPU hierarchy: register, lane,
# warp, and block. This tells us exactly how tensor elements map to hardware.
#
# The ``to_linear_layout`` function converts any layout to its linear layout
# equivalent. This is a ``constexpr`` operation: it runs at compile time and has
# zero runtime cost.
#
# Consider this 2D blocked layout for a 128x64 tensor with 4 warps:
#
# .. code-block:: python
#
#     layout = gl.BlockedLayout(
#         size_per_thread=[1, 8],
#         threads_per_warp=[32, 1],
#         warps_per_cta=[4, 1],
#         order=[1, 0],
#     )
#
# Converting it to a linear layout yields:
#
# .. code-block:: python
#
#     gl.DistributedLinearLayout(
#         reg_bases=[[0, 1], [0, 2], [0, 4]],
#         lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
#         warp_bases=[[32, 0], [64, 0]],
#         block_bases=[],
#         shape=[128, 64],
#     )
#
# Each basis is a 2-element vector ``[dim0, dim1]`` telling us which tensor
# dimension that hardware bit addresses. Notice:
#
# - The 3 register bases all have the form ``[0, X]`` -- they only address
#   dimension 1. This means each thread owns 2^3 = 8 consecutive elements along
#   dimension 1.
# - The 5 lane bases all have the form ``[X, 0]`` -- they address dimension 0.
# - The 2 warp bases also address dimension 0.
#
# Reducing along dimension 1 therefore only touches register-local data -- no
# shuffles or shared memory needed.
#
# Now consider a different layout for the same tensor:
#
# .. code-block:: python
#
#     layout = gl.BlockedLayout(
#         size_per_thread=[1, 1],
#         threads_per_warp=[1, 32],
#         warps_per_cta=[4, 1],
#         order=[1, 0],
#     )
#
# Its linear layout has:
#
# .. code-block:: python
#
#     reg_bases=[[0, 1]],  # 1 register basis for dim 1
#     lane_bases=[[0, 2], [0, 4], [0, 8], [0, 16], [0, 32]],  # 5 lane bases for dim 1!
#
# Here dimension 1 is spread across *lanes*. Reducing along dimension 1 now
# requires warp shuffles. Same tensor, same reduction, but much more expensive
# because of the layout choice.

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# %%
# Analyzing Register Ownership
# -----------------------------
#
# To use per-thread operations effectively, we need to know how many elements
# each thread owns along each dimension. Here is a ``constexpr`` helper that
# computes this from any layout:


@gluon.constexpr_function
def reg_size_per_dim(layout, shape):
    """Compute the number of register-local elements per dimension.

    For a DistributedLinearLayout, each register basis with a nonzero entry in
    dimension d doubles the register-local size along d. Returns a list where
    entry i is the register-local size along dimension i.
    """
    ll = gl.to_linear_layout(layout, shape)
    rank = len(ll.shape)
    sizes = [1] * rank
    for basis in ll.reg_bases:
        for d in range(rank):
            if basis[d] != 0:
                sizes[d] *= 2
    return sizes


# %%
# For the first layout above (``size_per_thread=[1, 8]``), this returns
# ``[1, 8]``: each thread owns 1 element along dim 0 and 8 along dim 1.
# For the second layout (``size_per_thread=[1, 1]``), it returns ``[1, 2]``:
# only 2 elements per thread along dim 1, with the rest on other lanes.
#
# This information is crucial: reducing along a dimension where
# ``reg_size == block_size`` is entirely thread-local and free.

# %%
# Register-Only Row Reduction
# ----------------------------
#
# Let's write a kernel that performs a row reduction (sum along dim 1) on a 2D
# tensor. When the layout places all of dim 1 in registers, the reduction
# requires no inter-thread communication.


@gluon.jit
def row_reduce_sum_kernel(in_ptr, out_ptr,  #
                          xnumel, ynumel, xstride, ystride,  #
                          layout: gl.constexpr, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    start_x = pid * XBLOCK

    # Load 2D tile using SliceLayout for 1D indices, same pattern as Tutorial 02
    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    indices_y = gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))

    offsets = xstride * indices_x[:, None] + ystride * indices_y[None, :]
    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    x = gl.load(in_ptr + offsets, mask=mask, other=0.0)

    # Row-wise sum. When the layout places all of dim 1 in registers,
    # the compiler generates no warp shuffles for this reduction.
    result = gl.sum(x, axis=1)

    # Store the 1D result. Use the same SliceLayout as indices_x so the
    # pointer layout matches the result layout (which is also a slice of
    # the parent layout along dim 1).
    out_mask = indices_x < xnumel
    gl.store(out_ptr + indices_x, result, mask=out_mask)


def row_reduce_sum(input, output, XBLOCK, YBLOCK, layout, num_warps):
    xnumel, ynumel = input.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )
    row_reduce_sum_kernel[grid](  #
        input, output, xnumel, ynumel,  #
        *input.stride(),  #
        layout, XBLOCK, YBLOCK, num_warps=num_warps)


# %%
# Test with a layout where dim 1 is entirely in registers. With
# ``size_per_thread=[1, YBLOCK]``, every element along dim 1 belongs to one
# thread, so ``gl.sum(x, axis=1)`` compiles to pure register adds.


@pytest.mark.parametrize("YBLOCK", [8, 16, 32])
@pytest.mark.parametrize("xnumel", [50, 128])
@pytest.mark.parametrize("num_warps", [4])
def test_row_reduce_register_only(YBLOCK, xnumel, num_warps):
    torch.manual_seed(0)
    ynumel = YBLOCK  # dim 1 exactly fits the block
    input = torch.randn((xnumel, ynumel), device="cuda")
    output = torch.empty(xnumel, device="cuda")
    XBLOCK = 128

    # Each thread owns the entire row: size_per_thread=[1, YBLOCK]
    layout = gl.BlockedLayout([1, YBLOCK], [32, 1], [num_warps, 1], [0, 1])
    row_reduce_sum(input, output, XBLOCK, YBLOCK, layout, num_warps)

    expected = input.sum(dim=1)
    torch.testing.assert_close(output[:xnumel], expected, atol=1e-4, rtol=1e-4)


# %%
# Now test with a layout where dim 1 is spread across lanes. The result is
# identical, but the compiler must generate warp shuffles to communicate
# partial sums between threads.


@pytest.mark.parametrize("YBLOCK", [32])
@pytest.mark.parametrize("xnumel", [50, 128])
@pytest.mark.parametrize("num_warps", [4])
def test_row_reduce_cross_thread(YBLOCK, xnumel, num_warps):
    torch.manual_seed(0)
    ynumel = YBLOCK
    input = torch.randn((xnumel, ynumel), device="cuda")
    output = torch.empty(xnumel, device="cuda")
    XBLOCK = 128

    # Dim 1 distributed across lanes: threads_per_warp=[1, 32]
    layout = gl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0])
    row_reduce_sum(input, output, XBLOCK, YBLOCK, layout, num_warps)

    expected = input.sum(dim=1)
    torch.testing.assert_close(output[:xnumel], expected, atol=1e-4, rtol=1e-4)


# %%
# Both tests produce the same result, but the register-only version generates
# simpler code. You can verify this by examining the PTX or SASS output:
# the register-only version will have no ``shfl`` instructions for the
# reduction.

# %%
# Partial Reduction via Reshape
# ------------------------------
#
# In many cases, the layout distributes a dimension across *both* registers and
# lanes (or warps). For example, an MMA output layout might give each thread
# 16 elements along dim 1, with the remaining distributed across 4 lanes.
#
# We can perform a *partial* reduction over just the register-local portion
# by reshaping the tensor to separate the register and non-register parts of
# the dimension, then reducing along only the register axis.


@gluon.jit
def partial_reduce_sum(x, dim: gl.constexpr, reg_size: gl.constexpr):
    """Partially reduce x along `dim`, summing only the register-local portion.

    Reshapes dim from size N to [N // reg_size, reg_size], reduces the inner
    axis. The reshape is free when the register bases are contiguous (which
    they are for BlockedLayout).
    """
    shape = x.shape
    M: gl.constexpr = shape[0]
    N: gl.constexpr = shape[1]
    GROUPS: gl.constexpr = N // reg_size
    x_3d = x.reshape([M, GROUPS, reg_size])
    return gl.sum(x_3d, axis=2)


@gluon.jit
def partial_reduce_max(x, dim: gl.constexpr, reg_size: gl.constexpr):
    """Partially reduce x along `dim`, taking max over the register-local portion."""
    shape = x.shape
    M: gl.constexpr = shape[0]
    N: gl.constexpr = shape[1]
    GROUPS: gl.constexpr = N // reg_size
    x_3d = x.reshape([M, GROUPS, reg_size])
    return gl.max(x_3d, axis=2)


# %%
# This is the core technique for efficient attention softmax: in the inner loop
# over key blocks, maintain a partially-reduced running max and sum, deferring
# the full cross-thread reduction until after the loop.

# %%
# Reduce-Broadcast Pattern
# -------------------------
#
# Attention kernels need to broadcast a reduced value (like a row-wise max)
# back to the original tile shape. In Gluon, broadcasting has a constraint:
# the operand must have a layout compatible with the reduction source.
#
# The standard solution is to *replay the reduction* on the original tensor to
# obtain a value with the correct reduced layout, then use ``convert_layout``
# to substitute in the actual values, and ``expand_dims`` to restore the
# original rank. This is the pattern described in the issue (#8580):


@gluon.jit
def reduce_broadcast(x, y, axes: gl.constexpr):
    """Broadcast y to the shape of x by replaying x's reduction.

    Given tensor x and tensor y (with the shape that results from reducing x
    along `axes`), produce a tensor with x's shape and layout containing y's
    values.

    The trick: reduce x along the same axes to get a tensor with the correct
    layout, then swap in y's values via convert_layout (assert_trivial), and
    expand_dims to restore the original shape.
    """
    keep_dims: gl.constexpr = False
    length: gl.constexpr = len(axes)
    # Forward: reduce x to get the target layout
    x_reduced = x
    for i in gl.static_range(length):
        x_reduced = gl.max(x_reduced, axes[length - i - 1], keep_dims=keep_dims)
    # Swap in y's values (layouts must be trivially compatible)
    y_converted = gl.convert_layout(y.reshape(x_reduced.shape), x_reduced.type.layout, assert_trivial=True)
    # Backward: expand dims to restore x's shape
    for i in gl.static_range(length):
        y_converted = y_converted.expand_dims(axes[i])
    return y_converted


# %%
# With ``reduce_broadcast``, the softmax correction step in an attention inner
# loop becomes straightforward:
#
# .. code-block:: python
#
#     @gluon.jit
#     def softmax_correction(acc, m_i, m_ij):
#         # m_i, m_ij have the shape from reducing acc along dim 1
#         alpha = gl.exp2(m_i - m_ij)
#         # Broadcast alpha back to acc's shape and multiply
#         acc = acc * reduce_broadcast(acc, alpha, gl.constexpr((1,)))
#         return acc
#
# When dim 1 is register-only in the layout, *both* the reduction and the
# broadcast are entirely thread-local -- no communication at any step.

# %%
# Let's write a complete kernel that demonstrates the reduce-broadcast pattern
# with a simple "subtract row max" operation, which is the first step of
# numerically-stable softmax.


@gluon.jit
def subtract_row_max_kernel(in_ptr, out_ptr,  #
                            xnumel, ynumel, xstride, ystride,  #
                            layout: gl.constexpr, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    start_x = pid * XBLOCK

    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    indices_y = gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))

    offsets = xstride * indices_x[:, None] + ystride * indices_y[None, :]
    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    x = gl.load(in_ptr + offsets, mask=mask, other=-float("inf"))

    # Row-wise max
    row_max = gl.max(x, axis=1)

    # Broadcast max back and subtract
    row_max_bc = reduce_broadcast(x, row_max, gl.constexpr((1, )))
    result = x - row_max_bc

    gl.store(out_ptr + offsets, result, mask=mask)


def subtract_row_max(input, output, XBLOCK, YBLOCK, layout, num_warps):
    xnumel, ynumel = input.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )
    subtract_row_max_kernel[grid](  #
        input, output, xnumel, ynumel,  #
        *input.stride(),  #
        layout, XBLOCK, YBLOCK, num_warps=num_warps)


@pytest.mark.parametrize("YBLOCK", [8, 16])
@pytest.mark.parametrize("xnumel", [50, 128])
@pytest.mark.parametrize("num_warps", [4])
def test_subtract_row_max(YBLOCK, xnumel, num_warps):
    torch.manual_seed(0)
    ynumel = YBLOCK
    input = torch.randn((xnumel, ynumel), device="cuda")
    output = torch.empty_like(input)
    XBLOCK = 128

    # Register-only layout along dim 1
    layout = gl.BlockedLayout([1, YBLOCK], [32, 1], [num_warps, 1], [0, 1])
    subtract_row_max(input, output, XBLOCK, YBLOCK, layout, num_warps)

    expected = input - input.max(dim=1, keepdim=True).values
    torch.testing.assert_close(output[:xnumel, :], expected, atol=1e-5, rtol=1e-5)


# %%
# Per-Thread Quantization
# ------------------------
#
# SageAttention (https://arxiv.org/abs/2411.10958) introduced per-thread
# quantization: instead of computing one scale per row (which requires a
# cross-thread reduction), each thread computes its own scale from the elements
# it holds in registers. This dramatically reduces quantization overhead because
# the max-abs computation is entirely thread-local.
#
# With linear layouts, we can implement this generically: find the absolute max
# of each thread's register-local elements along the reduction dimension,
# compute a scale, quantize, and later dequantize with the same per-thread
# scale.
#
# The key insight from SageAttention is that we do not need a *global* scale
# for each row. Since the MMA (matrix multiply-accumulate) instruction
# redistributes elements across threads anyway, the dequantization can use the
# per-thread scale from the *producing* thread. In practice, this means we
# do not need to understand any tensor core layout details -- we just compute
# the scale per thread and carry it through.
#
# The implementation reuses our partial reduce and reduce-broadcast helpers:
#
# .. code-block:: python
#
#     @gluon.jit
#     def per_thread_quantize(x, layout, reg_n):
#         # Per-thread max-abs (register-only, no communication)
#         abs_max = partial_reduce_max(gl.abs(x), 1, reg_n)
#         scale = 127.0 / gl.maximum(abs_max, 1e-12)
#
#         # Broadcast scale back and quantize
#         scale_bc = reduce_broadcast(x, scale, gl.constexpr((1,)))
#         x_quantized = (x * scale_bc).to(gl.int8)
#
#         return x_quantized, scale
#
# The ``partial_reduce_max`` computes the max over just the register-local
# portion of dim 1. The ``reduce_broadcast`` pushes the per-thread scale
# back to every element that thread owns. Both operations are zero-cost in
# terms of inter-thread communication.
#
# After the MMA, the dequantization step uses the same per-thread scale:
#
# .. code-block:: python
#
#     x_dequantized = x_quantized.to(gl.float32) / scale_bc

# %%
# Layout Selection for Per-Thread Operations
# --------------------------------------------
#
# The effectiveness of per-thread operations depends entirely on the layout.
# Here are some guidelines:
#
# - **Maximize register ownership along the reduction dimension.** Use
#   ``size_per_thread=[1, K]`` where K is as large as practical.
#
# - **Match the MMA output layout.** In attention kernels, the QK scores tile
#   comes from a matrix multiply. The MMA layout determines which elements
#   each thread owns. Use ``to_linear_layout`` to examine the MMA accumulator
#   layout and design partial reductions that target the register-local
#   dimensions.
#
# - **Use ``reg_size_per_dim`` to verify.** Before writing a performance-
#   critical kernel, check that the target dimension's register size matches
#   your expectations.
#
# For example, a typical MMA layout for a 128x64 accumulator might have:
#
# .. code-block:: text
#
#     reg_bases: [[0, 1], [8, 0], [0, 8], [0, 16], [0, 32], [64, 0]]
#     lane_bases: [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]]
#     warp_bases: [[16, 0], [32, 0]]
#
# Register bases ``[0, 1], [0, 8], [0, 16], [0, 32]`` address dimension 1
# (4 bases = 16 elements per thread), while ``[8, 0], [64, 0]`` address
# dimension 0. Reducing along dim 1 over the register-local portion reduces
# 16 elements per thread with no communication.
#
# The remaining dim 1 elements (on different lanes via ``[0, 2], [0, 4]``)
# need a small warp reduction after the loop -- a factor of 4 instead of 64.

# %%
# Summary
# --------
#
# - **Linear layouts** reveal exactly how tensor elements map to registers,
#   lanes, warps, and blocks. Use ``to_linear_layout`` to inspect any layout.
# - **Per-thread operations** exploit register-local dimensions to perform
#   reductions and broadcasts without inter-thread communication.
# - **Partial reduction** via ``reshape`` + ``reduce`` isolates the register-
#   local portion of a dimension, enabling efficient inner-loop accumulators
#   in attention kernels.
# - The **reduce-broadcast pattern** replays a reduction to produce operands
#   with broadcast-compatible layouts.
# - **Per-thread quantization** (SageAttention) computes scales from
#   register-local elements, avoiding cross-thread max reductions entirely.
# - These techniques compose naturally with Gluon's existing primitives:
#   ``reshape``, ``reduce``, ``convert_layout``, and ``expand_dims``.
