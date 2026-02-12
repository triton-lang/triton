"""
TMA im2col mode and Convolution via Implicit GEMM
===================================================

This tutorial explains NVIDIA TMA's im2col mode and then applies it to an
end-to-end convolution kernel.

We cover:
1) TMA im2col fundamentals on a 4-D NHWC tensor
2) How TensorDescriptorIm2Col parameters control access boundaries
3) Practical loading patterns (basic, padded, shifted-offset, multi-batch)
4) How 2D convolution works (the sliding window)
5) The im2col algorithm that reshapes convolution into GEMM
6) A convolution kernel using TMA im2col + MMA (works on both Hopper and Blackwell)

We re-use the MMA abstraction from ``07-persistence.py`` so that the same kernel
runs on Hopper (WGMMA) and Blackwell (tcgen05 MMA) without code duplication.

Prerequisites:
  - ``04-tma.py``: TMA basics (tensor descriptors, async copies)
  - ``05-wgmma.py``: Hopper warp-group MMA (WGMMA) basics
  - ``06-tcgen05.py``: tcgen05 MMA for matrix multiplication on Blackwell
  - ``07-persistence.py``: MMA abstraction (WGMMA / MMAv5) and persistent kernels

Reference:
https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-mode

TMA im2col Mode Parameters:
    The following descriptor fields and boundary rules are specific to
    TensorDescriptorIm2Col in im2col mode.

Block Shape:
    block_shape = [pixelsPerColumn, channelsPerPixel]
    - pixelsPerColumn: number of pixels from spatial dims [N, H, W]
    - channelsPerPixel: number of channels per pixel from [C]

TensorDescriptor Parameters:
    - Input tensor: [N, H, W, C] in NHWC format
    - pixel_box_lower_corner: [H_lo, W_lo] offset from (0, 0)
    - pixel_box_upper_corner: [H_hi, W_hi] offset from [H - 1, W - 1]
    - element_strides: [1, 1, 1, 1] for contiguous element access
    - padding: "zero" or "nan" for out-of-bounds fills

Access Boundary:
    pixel_box_lower_corner, pixel_box_upper_corner, and offsets define the
    spatial access window:
    - Lower bound = pixel_box_lower_corner + offsets
    - Upper bound = [H - 1, W - 1] + pixel_box_upper_corner + offsets
    - Bounds are interpreted as closed (inclusive): [lower, upper]

    The input tensor in global memory is NOT padded. When accessing outside
    [0, 0] to [H-1, W-1], TMA fills with the padding value.

async_copy_global_to_shared_im2col:
    async_copy_global_to_shared_im2col(tensor_desc, coord, offsets, barrier, result)
    - coord: [batch_idx, start_h, start_w, channel_start] start coords
    - offsets: [h_offset, w_offset] spatial offsets (i16)

    Starting position (the first pixel): (batch_idx, start_h + h_offset, start_w + w_offset, channel_start)
"""

import importlib
import pytest
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor, TensorDescriptorIm2Col
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma

t7 = importlib.import_module("07-persistence")

if __name__ == "__main__" and not t7.is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")

# %%
# Tutorial Motivation
# ===================
#
# Before diving into kernels, we first explain the concept of
# **im2col convolution**: rewriting convolution as GEMM by flattening each
# sliding-window patch into a row.
#
# Then we motivate why **TMA im2col mode** is helpful for this pattern:
# it performs convolution-style address generation and padding handling in
# hardware, so the kernel uses simpler indexing and can spend more resources on
# MMA compute.
#
# %%
# What is a 2D Convolution?
# =======================
#
# A 2D convolution slides a small filter (kernel) across an input image and
# computes a weighted sum at every position. For a single-channel example
# with a 3x3 input and a 2x2 filter (stride=1, no padding), the output is 2x2:
#
# .. code-block:: text
#
#     Input (3x3):        Filter (2x2):       Output (2x2):
#     +---+---+---+       +----+----+          +----+----+
#     | a | b | c |       | w0 | w1 |          | y0 | y1 |
#     +---+---+---+       +----+----+          +----+----+
#     | d | e | f |       | w2 | w3 |          | y2 | y3 |
#     +---+---+---+       +----+----+          +----+----+
#     | g | h | i |
#     +---+---+---+
#
# Each output element is the dot product of the filter with a region of the input:
#
# .. code-block:: text
#
#     y0 = a*w0 + b*w1 + d*w2 + e*w3
#     y1 = b*w0 + c*w1 + e*w2 + f*w3
#     y2 = d*w0 + e*w1 + g*w2 + h*w3
#     y3 = e*w0 + f*w1 + h*w2 + i*w3
#
# With **multiple input channels** (Ci) and **multiple output channels** (Co):
#
# .. code-block:: text
#
#     y[n, oh, ow, co] = SUM over (r, s, ci) of:
#         input[n, oh*stride + r - pad, ow*stride + s - pad, ci] * weight[co, r, s, ci]
#
# This nested summation over (r, s, ci) is the key insight for converting
# convolution into matrix multiplication.
#
# %%
# The im2col Algorithm
# ====================
#
# **im2col** (image to column) rearranges the input so that convolution becomes
# a standard matrix multiplication (GEMM). The idea is:
#
# 1. For each output position (n, oh, ow), extract the R x S x Ci input
#    patch that the filter overlaps with.
# 2. Flatten this patch into a single row of length ``R * S * Ci``.
# 3. Stack all such rows into a matrix **A** of shape ``[M, K]``.
# 4. Reshape the filter into a matrix **W** of shape ``[Co, K]``.
# 5. The output is simply ``A @ W^T`` (a GEMM!).
#
# where ``M = N * out_h * out_w`` and ``K = R * S * Ci``.
#
# .. code-block:: text
#
#     Example: 3x3 input, 2x2 filter, stride=1, no padding => 2x2 output
#
#     Step 1: Extract patches for each output position
#     -------------------------------------------------
#     y0 at (0,0): patch = [a, b, d, e]
#     y1 at (0,1): patch = [b, c, e, f]
#     y2 at (1,0): patch = [d, e, g, h]
#     y3 at (1,1): patch = [e, f, h, i]
#
#     Step 2: Stack patches into im2col matrix A (M=4, K=4)
#     -----------------------------------------------------
#              K = R*S*Ci = 4
#              <------------->
#          A = | a  b  d  e |  <- y0     ^
#              | b  c  e  f |  <- y1     | M = 4
#              | d  e  g  h |  <- y2     |
#              | e  f  h  i |  <- y3     v
#
#     Note: overlapping patches share input elements (e.g., 'e' appears in all 4 rows).
#
#     Step 3: Reshape filter into weight matrix W (Co=1, K=4)
#     -------------------------------------------------------
#          W = | w0 w1 w2 w3 |
#
#     Step 4: Output = A @ W^T
#     ------------------------
#                        | w0 |     | y0 |
#          A  @  W^T  =  | w1 |  =  | y1 |
#          (4x4)(4x1)    | w2 |     | y2 |
#                        | w3 |     | y3 |
#                                   (4x1)
#
#     With Co output channels, W is (Co x K) and Output is (M x Co).
#
# Why TMA im2col mode is helpful for convolution:
#   - In the matrix view above, each row of **A** is one output position's patch.
#   - The indexing term ``oh*stride + r - pad`` / ``ow*stride + s - pad`` must be
#     evaluated for every filter tap and channel block in the inner loop.
#   - Software address generation and bounds handling increase integer ALU work
#     and register pressure right where we want to maximize MMA throughput.
#   - ``TensorDescriptorIm2Col`` encodes stride/padding/pixel-box once, and TMA
#     hardware computes the gather addresses and out-of-bounds padding fills.
#   - The kernel can focus on MMA scheduling while using simple ``coord`` +
#     ``offsets`` loads, which is both cleaner and typically faster.
#
#   Example (multiple TMA im2col loads -> rows of A):
#   .. code-block:: text
#
#       Use the same setup as above: 3x3 input, 2x2 filter, stride=1, pad=0.
#
#       Input (same symbols as the matrix example):
#             W=0 W=1 W=2
#       H=0    a   b   c
#       H=1    d   e   f
#       H=2    g   h   i
#
#       For each output position (oh, ow), TMA im2col uses:
#         coord = [n, oh*stride - pad, ow*stride - pad, ci_blk]
#
#       Then one load is issued per filter offset (r, s):
#         offsets=[0,0], [0,1], [1,0], [1,1]
#
#       (oh=0, ow=0), coord=[n,0,0,ci_blk] -> first column of A:
#         [ a,
#           b,
#           d,
#           e ]
#       The width/channel dimension is Ci.
#       (oh=0, ow=1), coord=[n,0,1,ci_blk] -> [b, c, e, f] = second column of A
#       (oh=1, ow=0), coord=[n,1,0,ci_blk] -> [d, e, g, h] = third column of A
#       (oh=1, ow=1), coord=[n,1,1,ci_blk] -> [e, f, h, i] = fourth column of A
#       This is exactly the "image to column" idea: each output-position patch
#       from the image is flattened and stored as one column of A.
#
#       Stacking these multiple loads across output positions forms A:
#         | a  b  d  e |
#         | b  c  e  f |
#         | d  e  g  h |
#         | e  f  h  i |
#
#       In the real kernel, this is executed in the K-loop as one
#       async_copy_global_to_shared_im2col per (r, s, ci_block).
#
# %%
# Shared Kernel for All Examples
# ==============================
#
# All examples use the same kernel structure - only the TensorDescriptor
# configuration and load coordinates differ.


@gluon.jit
def tma_im2col_kernel(in_desc, out_desc, coord_n: int, coord_h: int, coord_w: int, coord_c: int, offset_h: tl.constexpr,
                      offset_w: tl.constexpr):
    """Generic im2col kernel with configurable coordinates and offsets."""
    smem = ttgl.allocate_shared_memory(in_desc.dtype, in_desc.block_shape, in_desc.layout)

    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, in_desc.block_type.nbytes)

    # TMA im2col load with specified coordinates and offsets
    tma.async_copy_global_to_shared_im2col(
        in_desc,
        [coord_n, coord_h, coord_w, coord_c],
        [offset_h, offset_w],
        bar,
        smem,
    )

    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    tma.async_copy_shared_to_global(out_desc, [0, 0], smem)
    tma.store_wait(pendings=0)


def run_tma_im2col(title, pixel_box_lower_corner, pixel_box_upper_corner, coord, offsets):
    """
    Shared function for all im2col examples.

    Args:
        title: Description to print
        pixel_box_lower_corner: [H_lo, W_lo] lower boundary for TMA access
        pixel_box_upper_corner: [H_hi, W_hi] upper boundary offset for TMA access
        coord: [N, H, W, C] starting coordinates
        offsets: [h_offset, w_offset] spatial offsets
    """
    # Create input tensor with values 1-16 and output tensor
    inp = torch.arange(1, 17, device="cuda", dtype=torch.float32).unsqueeze(1).repeat(1, 32).reshape(1, 4, 4, 32)
    out = torch.zeros(16, 32, device="cuda", dtype=torch.float32)

    block_shape = [16, 32]
    layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2)

    in_desc = TensorDescriptorIm2Col.from_tensor(
        inp,
        block_shape,
        layout,
        padding="zero",
        element_strides=[1, 1, 1, 1],
        pixel_box_lower_corner=pixel_box_lower_corner,
        pixel_box_upper_corner=pixel_box_upper_corner,
    )
    out_desc = TensorDescriptor.from_tensor(out, block_shape, layout)

    tma_im2col_kernel[(1, )](in_desc, out_desc, *coord, *offsets, num_warps=1)

    print(title)
    print("\nLoaded data (pixel_id, value):")
    for pixel_id in range(16):
        value = int(out[pixel_id, 0].item())
        print(f"  pixel {pixel_id:2d}: {value}")

    return inp, out


# %%
# Example 1: Basic Loading (No Padding)
# =====================================
#
# Load all 16 pixels (4x4 grid) from a [1, 4, 4, 32] tensor.
# - coord = [0, 0, 0, 0], offsets = [0, 0]
# - Access boundary: H in [0, 3], W in [0, 3] (original tensor)


def run_tma_im2col_simple():
    inp, out = run_tma_im2col(
        title="Example 1: Basic loading (no padding)",
        pixel_box_lower_corner=[0, 0],
        pixel_box_upper_corner=[0, 0],
        coord=[0, 0, 0, 0],
        offsets=[0, 0],
    )
    torch.testing.assert_close(out, inp.reshape(16, 32), atol=0, rtol=0)


@pytest.mark.skipif(not t7.is_hopper_or_newer(), reason="Requires Hopper")
def test_tma_im2col_simple():
    run_tma_im2col_simple()


# %%
# Example 1 Explanation:
# ----------------------
# Configuration:
#   - pixel_box_lower_corner = [0, 0], pixel_box_upper_corner = [0, 0] (no padding)
#   - coord = [0, 0, 0, 0], offsets = [0, 0]
#   - Access boundary: H in [0, 3], W in [0, 3]
#
# Expected output:
# ::
#
#     Loaded data: pixel 0->1, pixel 1->2, ..., pixel 15->16
#
#     Input [1, 4, 4, 32] spatial layout:     Output [16, 32] pixel order:
#     Channel dimension is contiguous: each pixel stores 32 channels in order.
#
#     +----+----+----+----+                   pixel  0 = (H=0,W=0) -> value 1
#     |  1 |  2 |  3 |  4 |  H=0              pixel  1 = (H=0,W=1) -> value 2
#     +----+----+----+----+                   ...
#     |  5 |  6 |  7 |  8 |  H=1              pixel 15 = (H=3,W=3) -> value 16
#     +----+----+----+----+
#     |  9 | 10 | 11 | 12 |  H=2
#     +----+----+----+----+
#     | 13 | 14 | 15 | 16 |  H=3
#     +----+----+----+----+
#       W=0  W=1  W=2  W=3

if __name__ == "__main__":
    run_tma_im2col_simple()
    print("\n" + "=" * 50 + "\n")

# %%
# ```
# Example 1: Basic loading (no padding)
#
# Loaded data (pixel_id, value):
#   pixel  0: 1
#   pixel  1: 2
#   pixel  2: 3
#   pixel  3: 4
#   pixel  4: 5
#   pixel  5: 6
#   pixel  6: 7
#   pixel  7: 8
#   pixel  8: 9
#   pixel  9: 10
#   pixel 10: 11
#   pixel 11: 12
#   pixel 12: 13
#   pixel 13: 14
#   pixel 14: 15
#   pixel 15: 16
#
# ==================================================
# ```

# %%
# Example 2: Loading from Padded Region
# =====================================
#
# With pixel_box_lower_corner=[-1,-1] and pixel_box_upper_corner=[-1,-1],
# access boundary is H in [-1, 2], W in [-1, 2].
# Negative coordinates are outside tensor -> filled with padding (0).


def run_tma_im2col_padded():
    _, out = run_tma_im2col(
        title="Example 2: Loading from padded region",
        pixel_box_lower_corner=[-1, -1],
        pixel_box_upper_corner=[-1, -1],
        coord=[0, -1, -1, 0],
        offsets=[0, 0],
    )
    # Expected: row H=-1 all padded, then each row has W=-1 padded
    expected = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11], device="cuda", dtype=torch.float32)
    torch.testing.assert_close(out[:, 0], expected, atol=0, rtol=0)


@pytest.mark.skipif(not t7.is_hopper_or_newer(), reason="Requires Hopper")
def test_tma_im2col_padded():
    run_tma_im2col_padded()


# %%
# Example 2 Explanation:
# ----------------------
# Configuration:
#   - pixel_box_lower_corner = [-1, -1], pixel_box_upper_corner = [-1, -1]
#   - coord = [0, -1, -1, 0], offsets = [0, 0]
#
# Access Boundary with offsets=[0, 0]:
#   - Lower = pixel_box_lower_corner + offsets = [-1, -1] + [0, 0]
#   - Upper = [H - 1, W - 1] + pixel_box_upper_corner + offsets = [3, 3] + [-1, -1] + [0, 0]
#   - Result: H in [-1, 2], W in [-1, 2]
#   - Negative coordinates are outside tensor -> filled with padding (0)
#
# Expected output:
# ::
#
#     pixel  0: 0  (H=-1, W=-1) padded     pixel  8: 0  (H=1, W=-1) padded
#     pixel  1: 0  (H=-1, W=0)  padded     pixel  9: 5  (H=1, W=0)  actual
#     pixel  2: 0  (H=-1, W=1)  padded     pixel 10: 6  (H=1, W=1)  actual
#     pixel  3: 0  (H=-1, W=2)  padded     pixel 11: 7  (H=1, W=2)  actual
#     pixel  4: 0  (H=0, W=-1)  padded     pixel 12: 0  (H=2, W=-1) padded
#     pixel  5: 1  (H=0, W=0)   actual     pixel 13: 9  (H=2, W=0)  actual
#     pixel  6: 2  (H=0, W=1)   actual     pixel 14: 10 (H=2, W=1)  actual
#     pixel  7: 3  (H=0, W=2)   actual     pixel 15: 11 (H=2, W=2)  actual
#
# Visualization:
# ::
#
#     Original 4x4 tensor:              Access boundary H in [-1,2], W in [-1,2]:
#     Note: padded cells are not stored in global memory; TMA fills them on load.
#
#          W=0  W=1  W=2  W=3                 W=-1  W=0  W=1  W=2
#         +----+----+----+----+              +----+----+----+----+
#     H=0 |  1 |  2 |  3 |  4 |          H=-1|  0 |  0 |  0 |  0 | <- padded
#         +----+----+----+----+              +----+----+----+----+
#     H=1 |  5 |  6 |  7 |  8 |          H=0 |  0 |  1 |  2 |  3 |
#         +----+----+----+----+              +----+----+----+----+
#     H=2 |  9 | 10 | 11 | 12 |          H=1 |  0 |  5 |  6 |  7 |
#         +----+----+----+----+              +----+----+----+----+
#     H=3 | 13 | 14 | 15 | 16 |          H=2 |  0 |  9 | 10 | 11 |
#         +----+----+----+----+              +----+----+----+----+
#                                             ^
#                                          padded (W<0)

if __name__ == "__main__":
    run_tma_im2col_padded()
    print("\n" + "=" * 50 + "\n")

# %%
# ```
# Example 2: Loading from padded region
#
# Loaded data (pixel_id, value):
#   pixel  0: 0
#   pixel  1: 0
#   pixel  2: 0
#   pixel  3: 0
#   pixel  4: 0
#   pixel  5: 1
#   pixel  6: 2
#   pixel  7: 3
#   pixel  8: 0
#   pixel  9: 5
#   pixel 10: 6
#   pixel 11: 7
#   pixel 12: 0
#   pixel 13: 9
#   pixel 14: 10
#   pixel 15: 11
#
# ==================================================
# ```

# %%
# Example 3: Offset Shifts Access Boundary
# ========================================
#
# Same TensorDescriptor as Example 2, but with offsets=[1, 1].
# This shifts the access boundary from [-1, 2] to [0, 3], covering the original tensor.


def run_tma_im2col_offset():
    inp, out = run_tma_im2col(
        title="Example 3: Offset shifts access to original tensor",
        pixel_box_lower_corner=[-1, -1],
        pixel_box_upper_corner=[-1, -1],
        coord=[0, -1, -1, 0],
        offsets=[1, 1],
    )
    # Offset shifts access to exactly the original tensor [0,3] x [0,3]
    torch.testing.assert_close(out, inp.reshape(16, 32), atol=0, rtol=0)


@pytest.mark.skipif(not t7.is_hopper_or_newer(), reason="Requires Hopper")
def test_tma_im2col_offset():
    run_tma_im2col_offset()


# %%
# Example 3 Explanation:
# ----------------------
# Configuration:
#   - pixel_box_lower_corner = [-1, -1], pixel_box_upper_corner = [-1, -1]
#   - coord = [0, -1, -1, 0], offsets = [1, 1]
#
# Access Boundary with offsets=[1, 1]:
#   - Lower = pixel_box_lower_corner + offsets = [-1, -1] + [1, 1]
#   - Upper = [H - 1, W - 1] + pixel_box_upper_corner + offsets = [3, 3] + [-1, -1] + [1, 1]
#   - Result: H in [0, 3], W in [0, 3] -> exactly the original tensor
#
# Comparison (Example 2 vs Example 3):
# ::
#
#     Example 2 (offsets=[0,0]):           Example 3 (offsets=[1,1]):
#     Access H in [-1, 2], W in [-1, 2]    Access H in [0, 3], W in [0, 3]
#
#          W=-1  W=0  W=1  W=2                  W=0  W=1  W=2  W=3
#         +----+----+----+----+              +----+----+----+----+
#     H=-1|  0 |  0 |  0 |  0 |          H=0 |  1 |  2 |  3 |  4 |
#         +----+----+----+----+              +----+----+----+----+
#     H=0 |  0 |  1 |  2 |  3 |          H=1 |  5 |  6 |  7 |  8 |
#         +----+----+----+----+              +----+----+----+----+
#     H=1 |  0 |  5 |  6 |  7 |          H=2 |  9 | 10 | 11 | 12 |
#         +----+----+----+----+              +----+----+----+----+
#     H=2 |  0 |  9 | 10 | 11 |          H=3 | 13 | 14 | 15 | 16 |
#         +----+----+----+----+              +----+----+----+----+
#
#     Key insight: offsets=[1,1] shifts the 4x4 access window from
#     the padded region to exactly cover the original tensor data.

if __name__ == "__main__":
    run_tma_im2col_offset()
    print("\n" + "=" * 50 + "\n")

# %%
# ```
# Example 3: Offset shifts access to original tensor
#
# Loaded data (pixel_id, value):
#   pixel  0: 1
#   pixel  1: 2
#   pixel  2: 3
#   pixel  3: 4
#   pixel  4: 5
#   pixel  5: 6
#   pixel  6: 7
#   pixel  7: 8
#   pixel  8: 9
#   pixel  9: 10
#   pixel 10: 11
#   pixel 11: 12
#   pixel 12: 13
#   pixel 13: 14
#   pixel 14: 15
#   pixel 15: 16
#
# ==================================================
# ```

# %%
# Example 4: Loading Across Multiple Batches
# ==========================================
#
# This example shows loading pixels that span across multiple images (batches).
# Input: [2, 4, 4, 32] - 2 batches of 4x4 images with 32 channels
# Starting at coord=[0, 1, 3, 0] (batch 0, H=1, W=3) and loading 16 pixels
# wraps from batch 0 into batch 1.


def run_tma_im2col_multi_batch():
    # Input: [2, 4, 4, 32] - 2 batches of 4x4 images (32 pixels total)
    # Values 1-32 for easy tracking
    inp = torch.arange(1, 33, device="cuda", dtype=torch.float32).unsqueeze(1).repeat(1, 32).reshape(2, 4, 4, 32)
    out = torch.zeros(16, 32, device="cuda", dtype=torch.float32)

    block_shape = [16, 32]  # load 16 pixels
    layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2)

    in_desc = TensorDescriptorIm2Col.from_tensor(
        inp,
        block_shape,
        layout,
        padding="zero",
        element_strides=[1, 1, 1, 1],
        pixel_box_lower_corner=[0, 0],
        pixel_box_upper_corner=[0, 0],
    )
    out_desc = TensorDescriptor.from_tensor(out, block_shape, layout)

    # coord=[0, 1, 3, 0] - start from batch 0, H=1, W=3, C=0
    # This is pixel index 1*4 + 3 = 7 in batch 0
    tma_im2col_kernel[(1, )](in_desc, out_desc, 0, 1, 3, 0, 0, 0, num_warps=1)

    print("Example 4: Loading across multiple batches")
    print("Input shape: [2, 4, 4, 32] (2 batches of 4x4 images)")
    print("Starting at: batch=0, H=1, W=3 (pixel 8 in input)")
    print("\nLoaded data (pixel_id, value):")
    for pixel_id in range(16):
        value = int(out[pixel_id, 0].item())
        print(f"  pixel {pixel_id:2d}: value {value:2d}")

    # Expected: values 8-16 from batch 0, then 17-23 from batch 1
    expected = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], device="cuda",
                            dtype=torch.float32)
    torch.testing.assert_close(out[:, 0], expected, atol=0, rtol=0)


@pytest.mark.skipif(not t7.is_hopper_or_newer(), reason="Requires Hopper")
def test_tma_im2col_multi_batch():
    run_tma_im2col_multi_batch()


# %%
# Example 4 Explanation:
# ----------------------
# Configuration:
#   - Input: [2, 4, 4, 32] - 2 batches of 4x4 images
#   - block_shape = [16, 32] - load 16 pixels
#   - coord = [0, 1, 3, 0] - start at batch 0, H=1, W=3
#
# Input layout (values 1-32):
# ::
#
#     Batch 0 (4x4):                      Batch 1 (4x4):
#     [x] = accessed pixel                 [x] = accessed pixel
#
#          W=0  W=1  W=2  W=3                  W=0  W=1  W=2  W=3
#         +----+----+----+----+              +----+----+----+----+
#     H=0 |  1 |  2 |  3 |  4 |          H=0 |[17]|[18]|[19]|[20]|
#         +----+----+----+----+              +----+----+----+----+
#     H=1 |  5 |  6 |  7 | [8]| <-start  H=1 |[21]|[22]|[23]| 24 |
#         +----+----+----+----+              +----+----+----+----+
#     H=2 |[ 9]|[10]|[11]|[12]|          H=2 | 25 | 26 | 27 | 28 |
#         +----+----+----+----+              +----+----+----+----+
#     H=3 |[13]|[14]|[15]|[16]|          H=3 | 29 | 30 | 31 | 32 |
#         +----+----+----+----+              +----+----+----+----+
#
# Starting at (N=0, H=1, W=3), loading 16 pixels in row-major order:
#   pixel  0: (N=0, H=1, W=3) -> value 8   (start position)
#   pixel  1: (N=0, H=2, W=0) -> value 9
#   pixel  2: (N=0, H=2, W=1) -> value 10
#   pixel  3: (N=0, H=2, W=2) -> value 11
#   pixel  4: (N=0, H=2, W=3) -> value 12
#   pixel  5: (N=0, H=3, W=0) -> value 13
#   pixel  6: (N=0, H=3, W=1) -> value 14
#   pixel  7: (N=0, H=3, W=2) -> value 15
#   pixel  8: (N=0, H=3, W=3) -> value 16  (last pixel in batch 0)
#   pixel  9: (N=1, H=0, W=0) -> value 17  (wraps to batch 1)
#   pixel 10: (N=1, H=0, W=1) -> value 18
#   pixel 11: (N=1, H=0, W=2) -> value 19
#   pixel 12: (N=1, H=0, W=3) -> value 20
#   pixel 13: (N=1, H=1, W=0) -> value 21
#   pixel 14: (N=1, H=1, W=1) -> value 22
#   pixel 15: (N=1, H=1, W=2) -> value 23
#
# Key insight: Loading wraps from batch 0 into batch 1 seamlessly,
# enabling efficient access patterns for convolution operations.

if __name__ == "__main__":
    run_tma_im2col_multi_batch()
    print("\n" + "=" * 50 + "\n")

# %%
# ```
# Example 4: Loading across multiple batches
# Input shape: [2, 4, 4, 32] (2 batches of 4x4 images)
# Starting at: batch=0, H=1, W=3 (pixel 8 in input)
#
# Loaded data (pixel_id, value):
#   pixel  0: value  8
#   pixel  1: value  9
#   pixel  2: value 10
#   pixel  3: value 11
#   pixel  4: value 12
#   pixel  5: value 13
#   pixel  6: value 14
#   pixel  7: value 15
#   pixel  8: value 16
#   pixel  9: value 17
#   pixel 10: value 18
#   pixel 11: value 19
#   pixel 12: value 20
#   pixel 13: value 21
#   pixel 14: value 22
#   pixel 15: value 23
#
# ==================================================
# ```

# %%
# Example 5: Multi-Batch with Padded Access Boundary
# ==================================================
#
# This example combines multi-batch loading with padding.
# With pixel_box_lower_corner=[-1,-1] and pixel_box_upper_corner=[-1,-1],
# the access boundary is H in [-1, 2], W in [-1, 2] per batch.
# Starting at H=1, W=2 loads pixels that wrap across batches within
# this constrained boundary.


def run_tma_im2col_multi_batch_padded():
    # Input: [2, 4, 4, 32] - 2 batches of 4x4 images (32 pixels total)
    inp = torch.arange(1, 33, device="cuda", dtype=torch.float32).unsqueeze(1).repeat(1, 32).reshape(2, 4, 4, 32)
    out = torch.zeros(16, 32, device="cuda", dtype=torch.float32)

    block_shape = [16, 32]
    layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2)

    in_desc = TensorDescriptorIm2Col.from_tensor(
        inp,
        block_shape,
        layout,
        padding="zero",
        element_strides=[1, 1, 1, 1],
        pixel_box_lower_corner=[-1, -1],
        pixel_box_upper_corner=[-1, -1],
    )
    out_desc = TensorDescriptor.from_tensor(out, block_shape, layout)

    # coord=[0, 1, 2, 0] - start from batch 0, H=1, W=2, C=0
    tma_im2col_kernel[(1, )](in_desc, out_desc, 0, 1, 2, 0, 0, 0, num_warps=1)

    print("Example 5: Multi-batch with padded access boundary")
    print("Input shape: [2, 4, 4, 32], pixel_box_lower_corner=[-1,-1], "
          "pixel_box_upper_corner=[-1,-1]")
    print("Access boundary per batch: H in [-1, 2], W in [-1, 2]")
    print("Starting at: batch=0, H=1, W=2")
    print("\nLoaded data (pixel_id, value):")
    for pixel_id in range(16):
        value = int(out[pixel_id, 0].item())
        print(f"  pixel {pixel_id:2d}: {value}")

    # Expected: 7, then padding + data pattern as described in documentation
    expected = torch.tensor([7, 0, 9, 10, 11, 0, 0, 0, 0, 0, 17, 18, 19, 0, 21, 22], device="cuda", dtype=torch.float32)
    torch.testing.assert_close(out[:, 0], expected, atol=0, rtol=0)


@pytest.mark.skipif(not t7.is_hopper_or_newer(), reason="Requires Hopper")
def test_tma_im2col_multi_batch_padded():
    run_tma_im2col_multi_batch_padded()


# %%
# Example 5 Explanation:
# ----------------------
# Configuration:
#   - Input: [2, 4, 4, 32] - 2 batches of 4x4 images
#   - pixel_box_lower_corner = [-1, -1], pixel_box_upper_corner = [-1, -1]
#   - Lower = pixel_box_lower_corner + offsets = [-1, -1] + [0, 0]
#   - Upper = [H - 1, W - 1] + pixel_box_upper_corner + offsets = [3, 3] + [-1, -1] + [0, 0]
#   - Access boundary per batch: H in [-1, 2], W in [-1, 2] (4x4 shifted region)
#   - coord = [0, 1, 2, 0] - start at batch 0, H=1, W=2
#
# Access boundary visualization (per batch):
# ::
#
#     Batch 0 - Original 4x4 tensor:     Batch 0 - Access boundary H in [-1,2], W in [-1,2]:
#     [x] = accessed pixel                [x] = accessed pixel, [0] = accessed padding
#
#          W=0  W=1  W=2  W=3                 W=-1  W=0  W=1  W=2
#         +----+----+----+----+              +----+----+----+----+
#     H=0 |  1 |  2 |  3 |  4 |          H=-1|  0 |  0 |  0 |  0 |
#         +----+----+----+----+              +----+----+----+----+
#     H=1 |  5 |  6 | [7]|  8 |          H=0 |  0 |  1 |  2 |  3 |
#         +----+----+----+----+              +----+----+----+----+
#     H=2 | [9]|[10]|[11]| 12 |          H=1 |  0 |  5 |  6 | [7]| <-start
#         +----+----+----+----+              +----+----+----+----+
#     H=3 | 13 | 14 | 15 | 16 |          H=2 | [0]| [9]|[10]|[11]|
#         +----+----+----+----+              +----+----+----+----+
#
#     Batch 1 - Original 4x4 tensor:     Batch 1 - Access boundary H in [-1,2], W in [-1,2]:
#     [x] = accessed pixel                [x] = accessed pixel, [0] = accessed padding
#
#          W=0  W=1  W=2  W=3                 W=-1  W=0  W=1  W=2
#         +----+----+----+----+              +----+----+----+----+
#     H=0 |[17]|[18]|[19]| 20 |          H=-1| [0]| [0]| [0]| [0]| <- accessed padding
#         +----+----+----+----+              +----+----+----+----+
#     H=1 |[21]|[22]| 23 | 24 |          H=0 | [0]|[17]|[18]|[19]|
#         +----+----+----+----+              +----+----+----+----+
#     H=2 | 25 | 26 | 27 | 28 |          H=1 | [0]|[21]|[22]| 23 |
#         +----+----+----+----+              +----+----+----+----+
#     H=3 | 29 | 30 | 31 | 32 |          H=2 |  0 | 25 | 26 | 27 |
#         +----+----+----+----+              +----+----+----+----+
#
# Starting at H=1, W=2 (value 7), loading 16 pixels within the boundary:
#   pixel  0: (N=0, H=1, W=2) -> value 7   (start position)
#   pixel  1: (N=0, H=2, W=-1) -> value 0  (padded, W=-1)
#   pixel  2: (N=0, H=2, W=0) -> value 9
#   pixel  3: (N=0, H=2, W=1) -> value 10
#   pixel  4: (N=0, H=2, W=2) -> value 11
#   pixel  5: (N=1, H=-1, W=-1) -> value 0 (wraps to batch 1, padded)
#   pixel  6: (N=1, H=-1, W=0) -> value 0  (padded)
#   pixel  7: (N=1, H=-1, W=1) -> value 0  (padded)
#   pixel  8: (N=1, H=-1, W=2) -> value 0  (padded)
#   pixel  9: (N=1, H=0, W=-1) -> value 0  (padded)
#   pixel 10: (N=1, H=0, W=0) -> value 17
#   pixel 11: (N=1, H=0, W=1) -> value 18
#   pixel 12: (N=1, H=0, W=2) -> value 19
#   pixel 13: (N=1, H=1, W=-1) -> value 0  (padded)
#   pixel 14: (N=1, H=1, W=0) -> value 21
#   pixel 15: (N=1, H=1, W=1) -> value 22
#
# Key insight: The access boundary constrains the 4x4 window per batch.
# When wrapping to the next batch, the boundary resets and padded regions
# (H=-1 or W=-1) are filled with zeros. This is useful for convolution
# operations, where it allows seamlessly processing pixel data across
# image boundaries in a single block.

if __name__ == "__main__":
    run_tma_im2col_multi_batch_padded()
    print("\n" + "=" * 50 + "\n")

# %%
# ```
# Example 5: Multi-batch with padded access boundary
# Input shape: [2, 4, 4, 32], pixel_box_lower_corner=[-1,-1], pixel_box_upper_corner=[-1,-1]
# Access boundary per batch: H in [-1, 2], W in [-1, 2]
# Starting at: batch=0, H=1, W=2
#
# Loaded data (pixel_id, value):
#   pixel  0: 7
#   pixel  1: 0
#   pixel  2: 9
#   pixel  3: 10
#   pixel  4: 11
#   pixel  5: 0
#   pixel  6: 0
#   pixel  7: 0
#   pixel  8: 0
#   pixel  9: 0
#   pixel 10: 17
#   pixel 11: 18
#   pixel 12: 19
#   pixel 13: 0
#   pixel 14: 21
#   pixel 15: 22
#
# ==================================================
# ```

# %%
# Implicit GEMM with TMA im2col
# ==============================
#
# **Explicit im2col** physically constructs the matrix A in memory. This is
# simple but wastes memory because overlapping patches duplicate input data.
# For an R x S filter, each input element may be copied up to R * S times!
#
# **Implicit GEMM** avoids materializing A. During the GEMM K-loop, it computes
# input addresses on the fly for each filter position ``(r, s)`` and channel
# block. On Hopper+ GPUs, TMA provides a dedicated **im2col mode** that performs
# this address generation in hardware. We configure one
# ``TensorDescriptorIm2Col`` (see the launcher), and each TMA load receives a
# filter offset ``[r, s]`` that selects which kernel position to gather. TMA
# then handles output-position strides, zero-padding for out-of-bounds
# accesses, and batch-boundary wrapping automatically, with no register-level
# index arithmetic.
#
# Pseudocode:
#
# ```
# # GEMM dimensions
# M = N * out_h * out_w        # output spatial positions
# N_gemm = Co                  # output channels
# K = R * S * Ci               # reduction dimension
#
# for each M-tile (output positions):
#     for each N-tile (output channels):
#         acc = zeros(BLOCK_M, BLOCK_N)
#
#         for r in range(R):                  # filter height
#             for s in range(S):              # filter width
#                 for ci_block in range(Ci // BLOCK_K):
#                     # Input tile via TMA im2col:
#                     # input[batch, out_y*stride+r-pad, out_x*stride+s-pad, ci_block*BLOCK_K:...]
#                     tma.async_copy_global_to_shared_im2col(...)
#                     A_tile = a_smem         # [BLOCK_M, BLOCK_K]
#
#                     # Weight tile via standard TMA:
#                     # weight[co_start:..., r, s, ci_block*BLOCK_K:...]
#                     tma.async_copy_global_to_shared(...)
#                     B_tile = b_smem         # [BLOCK_N, BLOCK_K]
#
#                     # Matrix multiply-accumulate (details omitted here)
#                     acc += A_tile @ B_tile^T
#
#         # Store output tile via TMA
#         tma.async_copy_shared_to_global(...)
# ```
#
# Key insight: we never materialize the full im2col matrix; address generation
# happens inside the K-loop via TMA im2col.
#
# %%
# Gluon Kernel
# ============
#
# The kernel below implements a single-buffered implicit GEMM convolution.
# ``store_output_tile`` is factored out as a helper; everything else is
# inline so the reader can follow the im2col algorithm top-to-bottom.
#
# ``MMAImpl`` (from ``07-persistence.py``) lets the same kernel run on Hopper
# (WGMMA, accumulator in registers) and Blackwell (tcgen05, accumulator in
# tensor memory) without any code changes.
#
# Convolution data layout convention for this section:
#   - Input:  NHWC  ``[N, H, W, Ci]``
#   - Weight: ``[Co, R, S, Ci]``  (output-channels first)
#   - Output: NHWC  ``[N, out_h, out_w, Co]``


@gluon.jit
def decompose_m_offset(pid_m, BLOCK_M: ttgl.constexpr, out_h: ttgl.constexpr, out_w: ttgl.constexpr):
    """Map M-tile index to (offs_m, batch, out_y, out_x) for TMA im2col coordinates."""
    offs_m = pid_m * BLOCK_M
    batch_id = offs_m // (out_h * out_w)
    m_residual = offs_m % (out_h * out_w)
    out_y = m_residual // out_w
    out_x = m_residual % out_w
    return offs_m, batch_id, out_y, out_x


@gluon.jit
def init_accumulator(in_desc, weight_desc, MMAImpl, dtype, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                     num_warps: ttgl.constexpr):
    """Allocate shared-memory tiles, MMA accumulator, and TMA barrier."""
    a_smem = ttgl.allocate_shared_memory(dtype, in_desc.block_shape, in_desc.layout)
    b_smem = ttgl.allocate_shared_memory(dtype, weight_desc.block_shape, weight_desc.layout)
    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    tma_bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar, count=1)
    return a_smem, b_smem, mma, tma_bar


@gluon.jit
def store_output_tile(mma, dtype, out_desc, offs_m, offs_n):
    """Wait for the final MMA, downcast, and write the output tile via TMA."""
    mma = mma.wait_num_outstanding(0)
    acc, mma = mma.take_result()
    c_smem = ttgl.allocate_shared_memory(dtype, out_desc.block_shape, out_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(out_desc, [offs_m, offs_n], c_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def conv2d_im2col_kernel(
    in_desc,
    weight_desc,
    out_desc,
    R: ttgl.constexpr,
    S: ttgl.constexpr,
    Ci: ttgl.constexpr,
    out_h: ttgl.constexpr,
    out_w: ttgl.constexpr,
    stride_h: ttgl.constexpr,
    stride_w: ttgl.constexpr,
    pad_h: ttgl.constexpr,
    pad_w: ttgl.constexpr,
    MMAImpl: ttgl.constexpr,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    num_warps: ttgl.constexpr,
):
    """
    Implicit GEMM convolution kernel using TMA im2col + MMA.
    Works on both Hopper (WGMMA) and Blackwell (tcgen05) via MMAImpl.
    """
    dtype: ttgl.constexpr = in_desc.dtype

    # for each M-tile / N-tile:
    pid_m, pid_n = ttgl.program_id(0), ttgl.program_id(1)
    offs_m, batch_id, out_y, out_x = decompose_m_offset(pid_m, BLOCK_M, out_h, out_w)

    # acc = zeros(BLOCK_M, BLOCK_N)
    a_smem, b_smem, mma, tma_bar = init_accumulator(in_desc, weight_desc, MMAImpl, dtype, BLOCK_M, BLOCK_N, num_warps)
    phase = 0

    # for r in range(R): for s in range(S): for ci_blk in range(ceil(Ci/BLOCK_K))
    # When Ci is not a multiple of BLOCK_K, last channel block is partial; TMA zero-fills OOB.
    ci_num_blocks = ttgl.cdiv(Ci, BLOCK_K)
    total_k_iters = R * S * ci_num_blocks
    for k_iter in range(total_k_iters):
        ci_block, rs_idx = k_iter % ci_num_blocks, k_iter // ci_num_blocks
        r, s = rs_idx // S, rs_idx % S

        # A = load_input[batch, oh*stride+r-pad, ow*stride+s-pad, ci_blk*BLOCK_K:…]
        # Equivalent TMA-im2col mapping:
        #   coord   = [batch_id, out_y*stride_h-pad_h, out_x*stride_w-pad_w, ci_block*BLOCK_K]
        #   offsets = [r, s]
        # TMA applies offsets to the spatial coords, so start is:
        #   [batch_id, out_y*stride_h-pad_h+r, out_x*stride_w-pad_w+s, ci_block*BLOCK_K]
        mbarrier.expect(tma_bar, in_desc.block_type.nbytes + weight_desc.block_type.nbytes)
        tma.async_copy_global_to_shared_im2col(
            in_desc,
            [batch_id, out_y * stride_h - pad_h, out_x * stride_w - pad_w, ci_block * BLOCK_K],
            [r.to(tl.int16), s.to(tl.int16)],
            tma_bar,
            a_smem,
        )
        # B = load_weight[co_start:…, k_offset:k_offset+BLOCK_K].  Uses Ci (not
        # ci_num_blocks*BLOCK_K) as the per-(r,s) stride so offsets stay aligned with the
        # flat (Co, R*S*Ci) layout.  When the last ci block bleeds into the next (r,s)
        # group, those weight elements are multiplied by zero-filled input channels (TMA
        # zero-fills input channels past Ci), so the result is still correct.
        k_offset = r * S * Ci + s * Ci + ci_block * BLOCK_K
        tma.async_copy_global_to_shared(weight_desc, [pid_n * BLOCK_N, k_offset], tma_bar, b_smem)
        mbarrier.wait(tma_bar, phase=phase)

        # acc += A @ B^T
        mma = mma.wait_num_outstanding(0)
        mma = mma.issue_async_mma(a_smem, b_smem.permute((1, 0)))

        phase ^= 1

    mbarrier.invalidate(tma_bar)

    # store output[...] = acc
    store_output_tile(mma, dtype, out_desc, offs_m, pid_n * BLOCK_N)


# %%
# Host-Side Launcher
# ==================
#
# The host side sets up TMA descriptors and launches the kernel.
# The critical part is configuring the ``TensorDescriptorIm2Col`` with the
# correct ``pixel_box``, ``element_strides``, and ``padding`` to match
# the convolution parameters.
#
# ``t7.select_mma_impl()`` automatically picks the right MMA backend:
# ``WGMMA`` on Hopper (SM 9.x) or ``MMAv5`` on Blackwell (SM 10.x).
def conv2d_tma_im2col(input_nhwc, weight, stride=1, padding=0, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4):
    """
    2D convolution using TMA im2col + MMA (Hopper / Blackwell).

    Args:
        input_nhwc: [N, H, W, Ci] in NHWC format, float16
        weight:     [Co, R, S, Ci] filter tensor, float16
        stride:     convolution stride (default: 1)
        padding:    convolution padding (default: 0)
    """
    MMAImpl = t7.select_mma_impl()
    assert MMAImpl is not None, "conv2d_tma_im2col requires a Hopper or Blackwell GPU"

    N, H, W, Ci = input_nhwc.shape
    Co, R, S, Ci_w = weight.shape
    assert Ci == Ci_w, "Channel mismatch"
    assert stride > 0, f"stride must be positive, got {stride}"
    assert padding >= 0, f"padding must be non-negative, got {padding}"
    # Ci may be any positive value; when not divisible by BLOCK_K, last tile is partial (TMA zero-fills OOB).

    out_h = (H + 2 * padding - R) // stride + 1
    out_w = (W + 2 * padding - S) // stride + 1
    assert out_h > 0 and out_w > 0, (f"Invalid convolution geometry: out_h={out_h}, out_w={out_w}, "
                                     f"H={H}, W={W}, R={R}, S={S}, stride={stride}, padding={padding}")
    M_GEMM = N * out_h * out_w
    N_GEMM = Co

    output = torch.empty(M_GEMM, Co, device=input_nhwc.device, dtype=input_nhwc.dtype)

    # ── TMA im2col descriptor for input ──────────────────────────────────
    # This configures TMA's hardware im2col mode on the [N, H, W, Ci]
    # input tensor. Key parameters:
    #
    #   block_shape  = [BLOCK_M, BLOCK_K]
    #       BLOCK_M output positions x BLOCK_K channels per async copy.
    #
    #   element_strides = [1, stride, stride, 1]
    #       TMA steps by `stride` in H/W between consecutive output
    #       positions, matching the convolution stride.
    #
    #   pixel_box_{lower,upper}_corner
    #       Defines the spatial window that TMA traverses per batch.
    #       The window must cover exactly out_h * out_w pixels:
    #         window_h = (out_h - 1) * stride + 1
    #         upper_h  = window_h - H - padding
    #       Lower corner is [-padding, -padding].
    #
    #   padding = "zero"
    #       Out-of-bounds reads (from conv padding) return 0 automatically.
    #
    upper_h = (out_h - 1) * stride + 1 - H - padding
    upper_w = (out_w - 1) * stride + 1 - W - padding

    input_block_shape = [BLOCK_M, BLOCK_K]
    input_layout = ttgl.NVMMASharedLayout.get_default_for(input_block_shape, ttgl.float16)
    in_desc = TensorDescriptorIm2Col.from_tensor(
        input_nhwc,
        input_block_shape,
        input_layout,
        padding="zero",
        element_strides=[1, stride, stride, 1],
        pixel_box_lower_corner=[-padding, -padding],
        pixel_box_upper_corner=[upper_h, upper_w],
    )

    # ── TMA descriptor for weight ────────────────────────────────────────
    # Reshape weight [Co, R, S, Ci] -> [Co, R*S*Ci] for standard 2D TMA
    weight_2d = weight.reshape(Co, R * S * Ci)
    weight_block_shape = [BLOCK_N, BLOCK_K]
    weight_layout = ttgl.NVMMASharedLayout.get_default_for(weight_block_shape, ttgl.float16)
    weight_desc = TensorDescriptor.from_tensor(weight_2d, weight_block_shape, weight_layout)

    # ── TMA descriptor for output ────────────────────────────────────────
    # Output is [M_GEMM, Co] (flattened spatial dims).
    out_block_shape = [BLOCK_M, BLOCK_N]
    out_layout = ttgl.NVMMASharedLayout.get_default_for(out_block_shape, ttgl.float16)
    out_desc = TensorDescriptor.from_tensor(output, out_block_shape, out_layout)

    # ── Launch kernel ────────────────────────────────────────────────────
    grid = (triton.cdiv(M_GEMM, BLOCK_M), triton.cdiv(N_GEMM, BLOCK_N))
    conv2d_im2col_kernel[grid](
        in_desc,
        weight_desc,
        out_desc,
        R,
        S,
        Ci,
        out_h,
        out_w,
        stride,
        stride,
        padding,
        padding,
        MMAImpl=MMAImpl,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return output.reshape(N, out_h, out_w, Co)


# %%
# Testing
# =======
#
# We compare our TMA im2col convolution against PyTorch's conv2d.


@pytest.mark.parametrize("N", [1, 4])
@pytest.mark.parametrize("H,W", [(16, 16)])
@pytest.mark.parametrize("Ci,Co", [(64, 64)])
@pytest.mark.parametrize("R,S", [(3, 3), (1, 1)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.skipif(not t7.is_hopper_or_newer(), reason="Requires Hopper or newer GPU (SM 9.x+)")
def test_conv2d_tma_im2col(N, H, W, Ci, Co, R, S, stride, padding):
    torch.manual_seed(0)
    x_nhwc = torch.randn(N, H, W, Ci, device="cuda", dtype=torch.float16)
    w_co_r_s_ci = torch.randn(Co, R, S, Ci, device="cuda", dtype=torch.float16)

    # Our kernel
    triton_out = conv2d_tma_im2col(x_nhwc, w_co_r_s_ci, stride=stride, padding=padding)

    # PyTorch reference (input NCHW, weight OIHW)
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    w_oihw = w_co_r_s_ci.permute(0, 3, 1, 2).contiguous()
    torch_out = torch.nn.functional.conv2d(x_nchw, w_oihw, stride=stride, padding=padding)
    torch_out_nhwc = torch_out.permute(0, 2, 3, 1)
    torch.testing.assert_close(triton_out, torch_out_nhwc, atol=1e-2, rtol=1e-2)


# %%
# Summary
# =======
#
# In this tutorial we learned:
#
# 1. **TMA im2col fundamentals**: How ``TensorDescriptorIm2Col`` parameters
#    (``pixel_box_lower_corner``, ``pixel_box_upper_corner``, ``element_strides``,
#    ``padding``) control access boundaries on NHWC tensors, including padded
#    regions, offset shifts, and multi-batch wrapping.
#
# 2. **Convolution as GEMM**: The im2col transformation converts convolution
#    into a matrix multiplication by rearranging input patches into rows of
#    a matrix.
#
# 3. **Implicit GEMM**: Instead of materializing the im2col matrix, we compute
#    input addresses on-the-fly during the GEMM K-loop, saving memory.
#
# 4. **TMA im2col**: TMA hardware natively supports im2col address
#    computation, eliminating register pressure from index arithmetic. We
#    configure a ``TensorDescriptorIm2Col`` with convolution parameters
#    (strides, padding, pixel box), and TMA handles the rest. This feature
#    is available on both Hopper and Blackwell GPUs.
#
# 5. **Unified kernel via MMA abstraction**: By using the ``WGMMA`` / ``MMAv5``
#    abstraction from ``07-persistence.py``, a single kernel implementation
#    works on both Hopper and Blackwell. The abstraction handles:
#
#    - Accumulator allocation (registers on Hopper, tensor memory on Blackwell)
#    - MMA barrier management
#    - Async MMA issue and wait
#
#    The kernel itself only deals with TMA loads and the unified MMA API:
#    ``initialize``, ``issue_async_mma``, ``wait_num_outstanding``, ``take_result``.
#
# For a production-quality implementation with warp specialization and
# pipelining, see ``examples/gluon/02-convolution.py``.
if __name__ == "__main__":
    # Conv2d demo
    torch.manual_seed(0)
    N, H, W, Ci, Co, R, S = 4, 16, 16, 64, 64, 3, 3
    stride, padding = 1, 1

    x_nhwc = torch.randn(N, H, W, Ci, device="cuda", dtype=torch.float16)
    w_co_r_s_ci = torch.randn(Co, R, S, Ci, device="cuda", dtype=torch.float16)

    triton_out = conv2d_tma_im2col(x_nhwc, w_co_r_s_ci, stride=stride, padding=padding)

    # Compare with PyTorch (input NCHW, weight OIHW)
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    w_oihw = w_co_r_s_ci.permute(0, 3, 1, 2).contiguous()
    torch_out = torch.nn.functional.conv2d(x_nchw, w_oihw, stride=stride, padding=padding)
    torch_out_nhwc = torch_out.permute(0, 2, 3, 1)

    max_err = (triton_out - torch_out_nhwc).abs().max().item()
    print(f"Conv2d: N={N}, H={H}, W={W}, Ci={Ci}, Co={Co}, R={R}, S={S}, "
          f"stride={stride}, pad={padding}")
    print(f"Output shape: {list(triton_out.shape)}")
    print(f"Max absolute error vs PyTorch: {max_err:.6f}")
    print("PASSED!" if max_err < 0.02 else "FAILED!")

# %%
# ```
# Conv2d: N=4, H=16, W=16, Ci=64, Co=64, R=3, S=3, stride=1, pad=1
# Output shape: [4, 16, 16, 64]
# Max absolute error vs PyTorch: 0.000000
# PASSED!
# ```
