"""
Convolution via Implicit GEMM with TMA im2col
==============================================

This tutorial explains how to implement 2D convolution as a matrix multiplication
(GEMM) using the **im2col** transformation, and how NVIDIA's TMA (Tensor Memory
Accelerator) im2col hardware mode makes this highly efficient on Hopper+ GPUs.

We cover:
1. How 2D convolution works (the sliding window)
2. The im2col algorithm that reshapes convolution into GEMM
3. Explicit vs implicit im2col (the "Implicit GEMM" approach)
4. How TMA im2col hardware accelerates implicit GEMM
5. A convolution kernel using TMA im2col + MMA (works on both Hopper and Blackwell)

We re-use the MMA abstraction from ``07-persistence.py`` so that the same kernel
runs on Hopper (WGMMA) and Blackwell (tcgen05 MMA) without code duplication.

Prerequisites:
  - ``04-tma.py``: TMA basics (tensor descriptors, async copies)
  - ``05-wgmma.py``: Hopper warp-group MMA (WGMMA) basics
  - ``06-tcgen05.py``: tcgen05 MMA for matrix multiplication on Blackwell
  - ``07-persistence.py``: MMA abstraction (WGMMA / MMAv5) and persistent kernels
  - ``13-tma-im2col.py``: TMA im2col mode basics

Data layout convention:
  - Input:  NHWC  ``[N, H, W, Ci]``
  - Weight: NHWC  ``[Co, R, S, Ci]``  (output-channels first)
  - Output: NHWC  ``[N, out_h, out_w, Co]``
"""

import importlib

import pytest
import torch
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor, TensorDescriptorIm2Col
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
)

# Re-use MMA abstractions from the previous tutorial.
# t7.select_mma_impl() returns the correct MMA class (WGMMA on Hopper,
# MMAv5 on Blackwell) so we can write a single kernel for both architectures.
t7 = importlib.import_module("07-persistence")


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")

# %%
# What is 2D Convolution?
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
#     +---+---+---+                            +----+----+
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
#                        | w0 |
#          A  @  W^T  =  | w1 |  =  | y0 |
#          (4x4)(4x1)    | w2 |     | y1 |
#                        | w3 |     | y2 |
#                                   | y3 |
#                                   (4x1)
#
#     With Co output channels, W is (Co x K) and Output is (M x Co).

# %%
# Explicit vs Implicit im2col
# ============================
#
# **Explicit im2col** physically constructs the matrix A in memory. This is
# simple but wastes memory because overlapping patches duplicate input data.
# For an R x S filter, each input element may be copied up to R * S times!
#
# **Implicit im2col** (also called **Implicit GEMM**) avoids materializing A.
# Instead, during the GEMM computation, we compute the correct input address
# on-the-fly for each element of A. The GEMM loop becomes:
#
# .. code-block:: text
#
#     Pseudocode: Implicit GEMM Convolution
#     ======================================
#
#     # GEMM dimensions
#     M = N * out_h * out_w        # output spatial positions
#     N_gemm = Co                  # output channels
#     K = R * S * Ci               # reduction dimension
#
#     for each M-tile (block of output positions):
#         for each N-tile (block of output channels):
#             acc = zeros(BLOCK_M, BLOCK_N)
#
#             for r in range(R):          # filter height
#                 for s in range(S):      # filter width
#                     for ci_block in range(Ci // BLOCK_K):  # input channel blocks
#
#                         # Load input tile: for each output position in M-tile,
#                         # compute the input address:
#                         #   input[batch, out_y*stride + r - pad, out_x*stride + s - pad, ci_block*BLOCK_K : ...]
#                         A_tile = load_input_tile(...)   # shape: [BLOCK_M, BLOCK_K]
#
#                         # Load weight tile:
#                         #   weight[co_start:..., r, s, ci_block*BLOCK_K : ...]
#                         B_tile = load_weight_tile(...)  # shape: [BLOCK_N, BLOCK_K]
#
#                         # Matrix multiply-accumulate
#                         acc += A_tile @ B_tile^T        # [BLOCK_M, BLOCK_N]
#
#             store output tile: output[...] = acc
#
#     The key insight: we never build the full im2col matrix.
#     Instead, the address computation happens inside the K-loop.

# %%
# TMA im2col: Hardware-Accelerated Implicit GEMM
# ===============================================
#
# On Hopper and newer GPUs, NVIDIA's TMA (Tensor Memory Accelerator) has a special
# **im2col mode** that performs the address computation in hardware!
#
# Instead of manually computing input addresses for each output position
# (which requires expensive index arithmetic in registers), we configure a
# ``TensorDescriptorIm2Col`` once, and TMA handles the rest:
#
# .. code-block:: text
#
#     TMA im2col for Convolution
#     ==========================
#
#     TensorDescriptorIm2Col parameters:
#       - Input tensor:         [N, H, W, Ci] in NHWC format
#       - block_shape:          [BLOCK_M, BLOCK_K]
#                               BLOCK_M = pixels (output positions)
#                               BLOCK_K = channels per load
#       - element_strides:      [1, stride_h, stride_w, 1]
#                               TMA steps by stride between output positions
#       - pixel_box_lower:      [-pad_h_left, -pad_w_left]
#       - pixel_box_upper:      [H + pad_h_right, W + pad_w_right]
#       - padding:              "zero" (out-of-bounds reads return 0)
#
#     At each K-iteration for filter position (r, s):
#       async_copy_global_to_shared_im2col(
#           in_desc,
#           coord  = [batch_id, out_y*stride - pad, out_x*stride - pad, ci_start],
#           offset = [r, s],    # <-- the filter position
#           bar, smem
#       )
#
#     TMA automatically:
#       1. Computes the input address for each output position in the tile
#       2. Handles padding (returns 0 for out-of-bounds)
#       3. Handles strided access between output positions
#       4. Wraps across batch boundaries
#       5. Loads data directly into shared memory (async, off the main data path)
#
# This eliminates all the per-element address computation from the GPU cores,
# freeing them to focus on the matrix multiply computation (WGMMA on Hopper,
# tcgen05 on Blackwell).
#
# The mapping between convolution and GEMM is:
#
# .. code-block:: text
#
#     Convolution -> GEMM Mapping
#     ===========================
#
#     GEMM dimension    Size                 Meaning
#     ─────────────────────────────────────────────────────────
#     M                 N * out_h * out_w    Output spatial positions
#     N_gemm            Co                   Output channels
#     K                 R * S * Ci           Filter size x input channels
#
#     Operand           Shape                Source
#     ─────────────────────────────────────────────────────────
#     A (input)         [M, K]               TMA im2col (implicit)
#     B (weight)        [N_gemm, K]          TMA tiled load (explicit [Co, R*S*Ci])
#     C (output)        [M, N_gemm]          MMA accumulator

# %%
# Gluon Kernel: Convolution with TMA im2col
# ==========================================
#
# Now we implement a real GPU kernel. This kernel:
#
# - Loads **input tiles** using TMA im2col (hardware-accelerated implicit GEMM)
# - Loads **weight tiles** using standard TMA
# - Computes the matmul using the **MMA abstraction** from ``07-persistence.py``
# - Stores the output using **TMA**
#
# By accepting ``MMAImpl`` as a compile-time parameter, the same kernel works on
# both Hopper (WGMMA, accumulator in registers) and Blackwell (tcgen05 MMA,
# accumulator in tensor memory). The MMA abstraction handles all the differences:
# accumulator allocation, barrier management, and the async MMA issue/wait API.
#
# See ``07-persistence.py`` for the full ``WGMMA`` and ``MMAv5`` class
# definitions, and ``06-tcgen05.py`` / ``05-wgmma.py`` for the underlying
# tensor core operations.

# %%
# The kernel below implements a single-buffered implicit GEMM convolution.
# Each program computes one (BLOCK_M x BLOCK_N) tile of the output.


@gluon.jit
def conv2d_im2col_kernel(
    in_desc,
    weight_desc,
    out_desc,
    R: gl.constexpr,
    S: gl.constexpr,
    Ci: gl.constexpr,
    out_h: gl.constexpr,
    out_w: gl.constexpr,
    stride_h: gl.constexpr,
    stride_w: gl.constexpr,
    pad_h: gl.constexpr,
    pad_w: gl.constexpr,
    MMAImpl: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_warps: gl.constexpr,
):
    """
    Implicit GEMM convolution kernel using TMA im2col + MMA.

    Works on both Hopper (WGMMA) and Blackwell (tcgen05) via MMAImpl.

    GEMM: Output[M, N_gemm] = A[M, K] @ B[N_gemm, K]^T
      where M = N_batch * out_h * out_w, N_gemm = Co, K = R * S * Ci
    """
    dtype: gl.constexpr = in_desc.dtype

    # ── Tile indices ──────────────────────────────────────────────────────
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)

    # Decompose M-offset into (batch, out_y, out_x) for TMA im2col coords
    offs_m = pid_m * BLOCK_M
    batch_id = offs_m // (out_h * out_w)
    m_residual = offs_m % (out_h * out_w)
    out_y = m_residual // out_w
    out_x = m_residual % out_w

    # ── Allocate shared memory for A (input) and B (weight) tiles ─────────
    a_smem = gl.allocate_shared_memory(dtype, in_desc.block_shape, in_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, weight_desc.block_shape, weight_desc.layout)

    # ── Initialize MMA (handles accumulator setup for both architectures) ─
    # On Hopper: accumulator in distributed registers (WGMMA)
    # On Blackwell: accumulator in tensor memory (tcgen05 MMA)
    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)

    # ── TMA barrier ──────────────────────────────────────────────────────
    # MMA barriers are managed internally by MMAImpl.
    tma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar, count=1)
    phase = 0

    # ── K-loop: iterate over (r, s, ci_block) ────────────────────────────
    # We iterate in the natural order: r -> s -> ci_block, so that
    # weight loads are sequential along the K dimension.
    ci_num_blocks = Ci // BLOCK_K
    total_k_iters = R * S * ci_num_blocks

    for k_iter in range(total_k_iters):
        # Decompose k_iter into (r, s, ci_block)
        ci_block = k_iter % ci_num_blocks
        rs_idx = k_iter // ci_num_blocks
        r = rs_idx // S
        s = rs_idx % S

        # 1. Set up TMA barrier and expect bytes from both loads
        mbarrier.expect(tma_bar, in_desc.block_type.nbytes + weight_desc.block_type.nbytes)

        # 2. Load input tile via TMA im2col
        #    coord: starting position in [N, H, W, C]
        #    offset: filter position [r, s] shifts the spatial access window
        tma.async_copy_global_to_shared_im2col(
            in_desc,
            [batch_id, out_y * stride_h - pad_h, out_x * stride_w - pad_w, ci_block * BLOCK_K],
            [r.to(tl.int16), s.to(tl.int16)],
            tma_bar,
            a_smem,
        )

        # 3. Load weight tile via standard TMA
        #    Weight is stored as [Co, R*S*Ci], so K-offset is sequential
        k_offset = r * S * Ci + s * Ci + ci_block * BLOCK_K
        tma.async_copy_global_to_shared(
            weight_desc,
            [pid_n * BLOCK_N, k_offset],
            tma_bar,
            b_smem,
        )

        # 4. Wait for both TMA loads to complete
        mbarrier.wait(tma_bar, phase=phase)

        # 5. MMA: acc += A_tile @ B_tile^T
        #    The MMA abstraction handles async issue and barrier management
        #    for both WGMMA (Hopper) and tcgen05 (Blackwell).
        mma = mma.wait_num_outstanding(0)
        mma = mma.issue_async_mma(a_smem, b_smem.permute((1, 0)))

        phase ^= 1

    mbarrier.invalidate(tma_bar)

    # ── Wait for last MMA and extract accumulator ─────────────────────────
    mma = mma.wait_num_outstanding(0)
    acc, mma = mma.take_result()

    # ── Downcast and store output tile via TMA ────────────────────────────
    c_smem = gl.allocate_shared_memory(dtype, out_desc.block_shape, out_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(out_desc, [offs_m, pid_n * BLOCK_N], c_smem)
    tma.store_wait(pendings=0)


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
    assert Ci % BLOCK_K == 0, f"Ci={Ci} must be divisible by BLOCK_K={BLOCK_K}"

    out_h = (H + 2 * padding - R) // stride + 1
    out_w = (W + 2 * padding - S) // stride + 1
    M_GEMM = N * out_h * out_w
    N_GEMM = Co

    output = torch.empty(M_GEMM, Co, device=input_nhwc.device, dtype=input_nhwc.dtype)

    # ── TMA im2col descriptor for input ──────────────────────────────────
    # pixel_box bounds define the spatial access window per batch.
    # With element_strides = [1, stride, stride, 1], TMA steps by `stride`
    # in H/W between output positions. The window must contain exactly
    # out_h * out_w pixels per batch:
    #   pixels_h = floor((window_h - 1) / stride) + 1 = out_h
    #   => window_h = (out_h - 1) * stride + 1
    #   => upper_h = window_h - H - padding
    upper_h = (out_h - 1) * stride + 1 - H - padding
    upper_w = (out_w - 1) * stride + 1 - W - padding

    input_block_shape = [BLOCK_M, BLOCK_K]
    input_layout = gl.NVMMASharedLayout.get_default_for(input_block_shape, gl.float16)
    in_desc = TensorDescriptorIm2Col(
        base=input_nhwc,
        shape=list(input_nhwc.shape),
        strides=list(input_nhwc.stride()),
        block_shape=input_block_shape,
        layout=input_layout,
        padding="zero",
        element_strides=[1, stride, stride, 1],
        pixel_box_lower_corner=[-padding, -padding],
        pixel_box_upper_corner=[upper_h, upper_w],
    )

    # ── TMA descriptor for weight ────────────────────────────────────────
    # Reshape weight [Co, R, S, Ci] -> [Co, R*S*Ci] for standard 2D TMA
    weight_2d = weight.reshape(Co, R * S * Ci)
    weight_block_shape = [BLOCK_N, BLOCK_K]
    weight_layout = gl.NVMMASharedLayout.get_default_for(weight_block_shape, gl.float16)
    weight_desc = TensorDescriptor.from_tensor(weight_2d, weight_block_shape, weight_layout)

    # ── TMA descriptor for output ────────────────────────────────────────
    # Output is [M_GEMM, Co] (flattened spatial dims)
    out_block_shape = [BLOCK_M, BLOCK_N]
    out_layout = gl.NVMMASharedLayout.get_default_for(out_block_shape, gl.float16)
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
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer GPU (SM 9.x+)")
def test_conv2d_im2col(N, H, W, Ci, Co, R, S, stride, padding):
    torch.manual_seed(0)

    x_nhwc = torch.randn(N, H, W, Ci, device="cuda", dtype=torch.float16)
    w_nhwc = torch.randn(Co, R, S, Ci, device="cuda", dtype=torch.float16)

    # Our kernel
    triton_out = conv2d_tma_im2col(x_nhwc, w_nhwc, stride=stride, padding=padding)

    # PyTorch reference (uses NCHW internally)
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    w_nchw = w_nhwc.permute(0, 3, 1, 2).contiguous()
    torch_out = torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding)
    torch_out_nhwc = torch_out.permute(0, 2, 3, 1)

    torch.testing.assert_close(triton_out, torch_out_nhwc, atol=1e-2, rtol=1e-2)


# %%
# Summary
# =======
#
# In this tutorial we learned:
#
# 1. **Convolution as GEMM**: The im2col transformation converts convolution
#    into a matrix multiplication by rearranging input patches into rows of
#    a matrix.
#
# 2. **Implicit GEMM**: Instead of materializing the im2col matrix, we compute
#    input addresses on-the-fly during the GEMM K-loop, saving memory.
#
# 3. **TMA im2col**: TMA hardware natively supports im2col address
#    computation, eliminating register pressure from index arithmetic. We
#    configure a ``TensorDescriptorIm2Col`` with convolution parameters
#    (strides, padding, pixel box), and TMA handles the rest. This feature
#    is available on both Hopper and Blackwell GPUs.
#
# 4. **Unified kernel via MMA abstraction**: By using the ``WGMMA`` / ``MMAv5``
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
    torch.manual_seed(0)
    N, H, W, Ci, Co, R, S = 4, 16, 16, 64, 64, 3, 3
    stride, padding = 1, 1

    x_nhwc = torch.randn(N, H, W, Ci, device="cuda", dtype=torch.float16)
    w_nhwc = torch.randn(Co, R, S, Ci, device="cuda", dtype=torch.float16)

    triton_out = conv2d_tma_im2col(x_nhwc, w_nhwc, stride=stride, padding=padding)

    # Compare with PyTorch
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    w_nchw = w_nhwc.permute(0, 3, 1, 2).contiguous()
    torch_out = torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding)
    torch_out_nhwc = torch_out.permute(0, 2, 3, 1)

    max_err = (triton_out - torch_out_nhwc).abs().max().item()
    print(f"Conv2d: N={N}, H={H}, W={W}, Ci={Ci}, Co={Co}, R={R}, S={S}, "
          f"stride={stride}, pad={padding}")
    print(f"Output shape: {list(triton_out.shape)}")
    print(f"Max absolute error vs PyTorch: {max_err:.6f}")
    print("PASSED!" if max_err < 0.02 else "FAILED!")
