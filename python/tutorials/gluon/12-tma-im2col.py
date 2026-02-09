"""
TMA im2col mode
================

TMA im2col mode enables efficient loading of tensor data and is described in
the PTX ISA "im2col mode" section:
https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-mode
This tutorial is a 4-D (NHWC) im2col example for convolution operations. It
loads a 2D block representing multiple pixels and their channels from a spatial
window in the input tensor.

Block Shape:
    block_shape = [pixelsPerColumn, channelsPerPixel]
    - pixelsPerColumn: number of pixels from spatial dims [N, H, W]
    - channelsPerPixel: number of channels per pixel from [C]

TensorDescriptor Parameters:
    - Input tensor: [N, H, W, C] in NHWC format
    - pixel_box_lower_corner: [H_lo, W_lo] offset from (0, 0)
    - pixel_box_upper_corner: [H_hi, W_hi] offset from (H, W)
    - element_strides: [1, 1, 1, 1] for contiguous element access
    - padding: "zero" or "nan" for out-of-bounds fills

Access Boundary:
    pixel_box_lower_corner, pixel_box_upper_corner, and offsets define the
    spatial access window:
    - Lower bound = pixel_box_lower_corner + offsets
    - Upper bound = [H, W] + pixel_box_upper_corner + offsets
    - Bounds are interpreted as half-open: [lower, upper)
    
    The input tensor in global memory is NOT padded. When accessing outside
    [0, 0] to [H-1, W-1], TMA fills with the padding value.

async_copy_global_to_shared_im2col:
    async_copy_global_to_shared_im2col(tensor_desc, coord, offsets, barrier, result)
    - coord: [batch_idx, start_h, start_w, channel_start] start coords
    - offsets: [h_offset, w_offset] spatial offsets (i16)
    
    Starting position (the first pixel): (batch_idx, start_h + h_offset, start_w + w_offset, channel_start)

"""

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor, TensorDescriptorIm2Col
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9

if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")

# %%
# Shared Kernel for All Examples
# ==============================
#
# All examples use the same kernel structure - only the TensorDescriptor
# configuration and load coordinates differ.


import triton.language as tl


@gluon.jit
def tma_im2col_kernel(in_desc, out_desc, coord_n: int, coord_h: int, coord_w: int, coord_c: int,
                      offset_h: tl.constexpr, offset_w: tl.constexpr):
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

    in_desc = TensorDescriptorIm2Col(
        base=inp, shape=list(inp.shape), strides=list(inp.stride()),
        block_shape=block_shape, layout=layout, padding="zero",
        element_strides=[1, 1, 1, 1],
        pixel_box_lower_corner=pixel_box_lower_corner,
        pixel_box_upper_corner=pixel_box_upper_corner,
    )
    out_desc = TensorDescriptor.from_tensor(out, block_shape, layout)

    tma_im2col_kernel[(1,)](in_desc, out_desc, *coord, *offsets, num_warps=1)

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
# - Access boundary: H in [0, 4), W in [0, 4) (original tensor)


def run_tma_im2col_simple():
    inp, out = run_tma_im2col(
        title="Example 1: Basic loading (no padding)",
        pixel_box_lower_corner=[0, 0],
        pixel_box_upper_corner=[0, 0],
        coord=[0, 0, 0, 0],
        offsets=[0, 0],
    )
    torch.testing.assert_close(out, inp.reshape(16, 32), atol=0, rtol=0)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper")
def test_tma_im2col_simple():
    run_tma_im2col_simple()


# %%
# Example 1 Explanation:
# ----------------------
# Configuration:
#   - pixel_box_lower_corner = [0, 0], pixel_box_upper_corner = [0, 0] (no padding)
#   - coord = [0, 0, 0, 0], offsets = [0, 0]
#   - Access boundary: H in [0, 4), W in [0, 4)
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


# %%
# Example 2: Loading from Padded Region
# =====================================
#
# With pixel_box_lower_corner=[-1,-1] and pixel_box_upper_corner=[-1,-1],
# access boundary is H in [-1, 3), W in [-1, 3).
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
    expected = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11],
                            device="cuda", dtype=torch.float32)
    torch.testing.assert_close(out[:, 0], expected, atol=0, rtol=0)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper")
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
#   - Upper = [H, W] + pixel_box_upper_corner + offsets = [4, 4] + [-1, -1] + [0, 0]
#   - Result: H in [-1, 3), W in [-1, 3)
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
#     Original 4x4 tensor:              Access boundary H in [-1,3), W in [-1,3):
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


# %%
# Example 3: Offset Shifts Access Boundary
# ========================================
#
# Same TensorDescriptor as Example 2, but with offsets=[1, 1].
# This shifts the access boundary from [-1, 3) to [0, 4), covering the original tensor.


def run_tma_im2col_offset():
    inp, out = run_tma_im2col(
        title="Example 3: Offset shifts access to original tensor",
        pixel_box_lower_corner=[-1, -1],
        pixel_box_upper_corner=[-1, -1],
        coord=[0, -1, -1, 0],
        offsets=[1, 1],
    )
    # Offset shifts access to exactly the original tensor [0,4) x [0,4)
    torch.testing.assert_close(out, inp.reshape(16, 32), atol=0, rtol=0)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper")
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
#   - Upper = [H, W] + pixel_box_upper_corner + offsets = [4, 4] + [-1, -1] + [1, 1]
#   - Result: H in [0, 4), W in [0, 4) -> exactly the original tensor
#
# Comparison (Example 2 vs Example 3):
# ::
#
#     Example 2 (offsets=[0,0]):           Example 3 (offsets=[1,1]):
#     Access H in [-1, 3), W in [-1, 3)    Access H in [0, 4), W in [0, 4)
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

    in_desc = TensorDescriptorIm2Col(
        base=inp, shape=list(inp.shape), strides=list(inp.stride()),
        block_shape=block_shape, layout=layout, padding="zero",
        element_strides=[1, 1, 1, 1],
        pixel_box_lower_corner=[0, 0],
        pixel_box_upper_corner=[0, 0],
    )
    out_desc = TensorDescriptor.from_tensor(out, block_shape, layout)

    # coord=[0, 1, 3, 0] - start from batch 0, H=1, W=3, C=0
    # This is pixel index 1*4 + 3 = 7 in batch 0
    tma_im2col_kernel[(1,)](in_desc, out_desc, 0, 1, 3, 0, 0, 0, num_warps=1)

    print("Example 4: Loading across multiple batches")
    print("Input shape: [2, 4, 4, 32] (2 batches of 4x4 images)")
    print("Starting at: batch=0, H=1, W=3 (pixel 8 in input)")
    print("\nLoaded data (pixel_id, value, source location):")
    for pixel_id in range(16):
        value = int(out[pixel_id, 0].item())
        print(f"  pixel {pixel_id:2d}: value {value:2d}")

    # Expected: values 8-16 from batch 0, then 17-23 from batch 1
    expected = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                            device="cuda", dtype=torch.float32)
    torch.testing.assert_close(out[:, 0], expected, atol=0, rtol=0)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper")
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
#     H=1 |  5 |  6 |  7 |[8]| <-start   H=1 |[21]|[22]|[23]| 24 |
#         +----+----+----+----+              +----+----+----+----+
#     H=2 |[9]|[10]|[11]|[12]|           H=2 | 25 | 26 | 27 | 28 |
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


# %%
# Example 5: Multi-Batch with Padded Access Boundary
# ==================================================
#
# This example combines multi-batch loading with padding.
# With pixel_box_lower_corner=[-1,-1] and pixel_box_upper_corner=[-1,-1],
# the access boundary is H in [-1, 3), W in [-1, 3) per batch.
# Starting at H=1, W=2 loads pixels that wrap across batches within
# this constrained boundary.


def run_tma_im2col_multi_batch_padded():
    # Input: [2, 4, 4, 32] - 2 batches of 4x4 images (32 pixels total)
    inp = torch.arange(1, 33, device="cuda", dtype=torch.float32).unsqueeze(1).repeat(1, 32).reshape(2, 4, 4, 32)
    out = torch.zeros(16, 32, device="cuda", dtype=torch.float32)

    block_shape = [16, 32]
    layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2)

    in_desc = TensorDescriptorIm2Col(
        base=inp, shape=list(inp.shape), strides=list(inp.stride()),
        block_shape=block_shape, layout=layout, padding="zero",
        element_strides=[1, 1, 1, 1],
        pixel_box_lower_corner=[-1, -1],
        pixel_box_upper_corner=[-1, -1],
    )
    out_desc = TensorDescriptor.from_tensor(out, block_shape, layout)

    # coord=[0, 1, 2, 0] - start from batch 0, H=1, W=2, C=0
    tma_im2col_kernel[(1,)](in_desc, out_desc, 0, 1, 2, 0, 0, 0, num_warps=1)

    print("Example 5: Multi-batch with padded access boundary")
    print("Input shape: [2, 4, 4, 32], pixel_box_lower_corner=[-1,-1], "
          "pixel_box_upper_corner=[-1,-1]")
    print("Access boundary per batch: H in [-1, 3), W in [-1, 3)")
    print("Starting at: batch=0, H=1, W=2")
    print("\nLoaded data (pixel_id, value):")
    for pixel_id in range(16):
        value = int(out[pixel_id, 0].item())
        print(f"  pixel {pixel_id:2d}: {value}")

    # Expected: 7, then padding + data pattern as described in documentation
    expected = torch.tensor([7, 0, 9, 10, 11, 0, 0, 0, 0, 0, 17, 18, 19, 0, 21, 22],
                            device="cuda", dtype=torch.float32)
    torch.testing.assert_close(out[:, 0], expected, atol=0, rtol=0)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper")
def test_tma_im2col_multi_batch_padded():
    run_tma_im2col_multi_batch_padded()


# %%
# Example 5 Explanation:
# ----------------------
# Configuration:
#   - Input: [2, 4, 4, 32] - 2 batches of 4x4 images
#   - pixel_box_lower_corner = [-1, -1], pixel_box_upper_corner = [-1, -1]
#   - Lower = pixel_box_lower_corner + offsets = [-1, -1] + [0, 0]
#   - Upper = [H, W] + pixel_box_upper_corner + offsets = [4, 4] + [-1, -1] + [0, 0]
#   - Access boundary per batch: H in [-1, 3), W in [-1, 3) (4x4 shifted region)
#   - coord = [0, 1, 2, 0] - start at batch 0, H=1, W=2
#
# Access boundary visualization (per batch):
# ::
#
#     Batch 0 - Original 4x4 tensor:     Batch 0 - Access boundary H in [-1,3), W in [-1,3):
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
#     Batch 1 - Original 4x4 tensor:     Batch 1 - Access boundary H in [-1,3), W in [-1,3):
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
# operations that need to handle image boundaries.


if __name__ == "__main__":
    run_tma_im2col_simple()
    print("\n" + "="*50 + "\n")
    run_tma_im2col_padded()
    print("\n" + "="*50 + "\n")
    run_tma_im2col_offset()
    print("\n" + "="*50 + "\n")
    run_tma_im2col_multi_batch()
    print("\n" + "="*50 + "\n")
    run_tma_im2col_multi_batch_padded()

