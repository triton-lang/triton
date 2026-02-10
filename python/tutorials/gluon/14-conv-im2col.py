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
5. A complete Blackwell convolution kernel using TMA im2col + tcgen05 MMA
6. A Hopper convolution kernel using TMA im2col + WGMMA

Prerequisites:
  - ``04-tma.py``: TMA basics (tensor descriptors, async copies)
  - ``05-wgmma.py``: Hopper warp-group MMA (WGMMA) basics
  - ``06-tcgen05.py``: tcgen05 MMA for matrix multiplication on Blackwell
  - ``13-tma-im2col.py``: TMA im2col mode basics

Data layout convention:
  - Input:  NHWC  ``[N, H, W, Ci]``
  - Weight: NHWC  ``[Co, R, S, Ci]``  (output-channels first)
  - Output: NHWC  ``[N, out_h, out_w, Co]``
"""

import pytest
import torch
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor, TensorDescriptorIm2Col
# TMA im2col, mbarrier, and fence live in the hopper module (shared by both architectures).
# The blackwell module's TMA does not expose async_copy_global_to_shared_im2col.
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
# tcgen05 MMA and tensor memory APIs are blackwell-specific.
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
)


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 9


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


# ── WGMMA layout helpers (used by the Hopper kernel) ────────────────────
# These determine the distributed register layout for the WGMMA accumulator.
# See ``05-wgmma.py`` for a detailed explanation.


@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    warps_per_cta = [4, 1]
    m = 16
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        if BLOCK_M > m * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    return warps_per_cta


@gluon.constexpr_function
def get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps):
    m = 16
    mReps = triton.cdiv(BLOCK_M, m)
    nReps = triton.cdiv(num_warps, mReps)
    maxN = max(BLOCK_N // nReps, 8)
    n = 256
    while n > maxN or BLOCK_N % n != 0:
        n -= 8
    assert n >= 8, "expected to find a valid n"
    return n


@gluon.constexpr_function
def pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    m = 16
    k = 256 // dtype.primitive_bitwidth
    n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
    warps_per_cta = get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")

# %%
# What is 2D Convolution?
# =======================
#
# A 2D convolution slides a small filter (kernel) across an input image and
# computes a weighted sum at every position. For a single-channel example
# with a 4x4 input and a 3x3 filter (stride=1, no padding), the output is 2x2:
#
# .. code-block:: text
#
#     Input (4x4):            Filter (3x3):           Output (2x2):
#     +---+---+---+---+       +----+----+----+         +----+----+
#     | a | b | c | d |       | w0 | w1 | w2 |         | y0 | y1 |
#     +---+---+---+---+       +----+----+----+         +----+----+
#     | e | f | g | h |       | w3 | w4 | w5 |         | y2 | y3 |
#     +---+---+---+---+       +----+----+----+         +----+----+
#     | i | j | k | l |       | w6 | w7 | w8 |
#     +---+---+---+---+
#     | m | n | o | p |
#     +---+---+---+---+
#
# Each output element is the dot product of the filter with a region of the input:
#
# .. code-block:: text
#
#     y0 = a*w0 + b*w1 + c*w2 + e*w3 + f*w4 + g*w5 + i*w6 + j*w7 + k*w8
#     y1 = b*w0 + c*w1 + d*w2 + f*w3 + g*w4 + h*w5 + j*w6 + k*w7 + l*w8
#     y2 = e*w0 + f*w1 + g*w2 + i*w3 + j*w4 + k*w5 + m*w6 + n*w7 + o*w8
#     y3 = f*w0 + g*w1 + h*w2 + j*w3 + k*w4 + l*w5 + n*w6 + o*w7 + p*w8
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
#     Example: 4x4 input, 3x3 filter, stride=1, no padding => 2x2 output
#
#     Step 1: Extract patches for each output position
#     -------------------------------------------------
#     y0 at (0,0): patch = [a, b, c, e, f, g, i, j, k]
#     y1 at (0,1): patch = [b, c, d, f, g, h, j, k, l]
#     y2 at (1,0): patch = [e, f, g, i, j, k, m, n, o]
#     y3 at (1,1): patch = [f, g, h, j, k, l, n, o, p]
#
#     Step 2: Stack patches into im2col matrix A (M=4, K=9)
#     -----------------------------------------------------
#                   K = R*S*Ci = 9
#              <-------------------------->
#          A = | a  b  c  e  f  g  i  j  k |  <- y0     ^
#              | b  c  d  f  g  h  j  k  l |  <- y1     | M = 4
#              | e  f  g  i  j  k  m  n  o |  <- y2     |
#              | f  g  h  j  k  l  n  o  p |  <- y3     v
#
#     Note: overlapping patches share input elements (e.g., 'f' appears in all 4 rows).
#
#     Step 3: Reshape filter into weight matrix W (Co=1, K=9)
#     -------------------------------------------------------
#          W = | w0 w1 w2 w3 w4 w5 w6 w7 w8 |
#
#     Step 4: Output = A @ W^T
#     ------------------------
#                        | w0 |
#                        | w1 |
#              Output =  | w2 |
#          A  @  W^T  =  | w3 |  =  | y0 |
#          (4x9)(9x1)    | w4 |     | y1 |
#                        | w5 |     | y2 |
#                        | w6 |     | y3 |
#                        | w7 |     (4x1)
#                        | w8 |
#
#     With Co output channels, W is (Co x K) and Output is (M x Co).

# %%
# Explicit vs Implicit im2col
# ============================
#
# **Explicit im2col** physically constructs the matrix A in memory. This is
# simple but wastes memory because overlapping patches duplicate input data.
# For a 3x3 filter, each input element may be copied up to 9 times!
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
#       - pixel_box_lower:      [-pad_h, -pad_w]
#       - pixel_box_upper:      [upper_h, upper_w]  (computed to give out_h * out_w pixels/batch)
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
#     C (output)        [M, N_gemm]          tcgen05 accumulator (in tensor memory)
#
#     K-loop iteration order:
#       for each (r, s) in filter:        # R*S iterations
#           for each ci_block:            # Ci // BLOCK_K iterations
#               load A tile via TMA im2col with offset=[r, s]
#               load B tile via TMA at weight[co_start, r*S*Ci + s*Ci + ci_block*BLOCK_K]
#               acc += A_tile @ B_tile^T  (tcgen05_mma)

# %%
# Reference: Explicit im2col in Python
# =====================================
#
# Before implementing the GPU kernel, let's verify our understanding with a
# simple Python reference that performs explicit im2col + matmul.


def conv2d_reference_im2col(input_nhwc, weight, stride=1, padding=0):
    """
    Reference convolution using explicit im2col + matmul.

    Args:
        input_nhwc: [N, H, W, Ci] input tensor in NHWC format
        weight: [Co, R, S, Ci] filter tensor
        stride: convolution stride
        padding: convolution padding

    Returns:
        output: [N, out_h, out_w, Co] in NHWC format
    """
    N, H, W, Ci = input_nhwc.shape
    Co, R, S, Ci_w = weight.shape
    assert Ci == Ci_w

    out_h = (H + 2 * padding - R) // stride + 1
    out_w = (W + 2 * padding - S) // stride + 1

    # Pad the input if needed: pad only H and W dimensions
    if padding > 0:
        padded = torch.nn.functional.pad(input_nhwc, (0, 0, padding, padding, padding, padding, 0, 0))
    else:
        padded = input_nhwc

    # Build the im2col matrix A of shape [M, K]
    # M = N * out_h * out_w, K = R * S * Ci
    M = N * out_h * out_w
    K = R * S * Ci
    A = torch.zeros(M, K, device=input_nhwc.device, dtype=input_nhwc.dtype)

    for n in range(N):
        for oh in range(out_h):
            for ow in range(out_w):
                # Row index in A
                m = n * out_h * out_w + oh * out_w + ow
                # Extract the R x S x Ci patch and flatten
                patch = padded[n, oh * stride:oh * stride + R, ow * stride:ow * stride + S, :]
                A[m, :] = patch.reshape(-1)

    # Weight matrix W of shape [Co, K]
    W_mat = weight.reshape(Co, K)

    # Output = A @ W^T -> [M, Co]
    output_flat = A @ W_mat.T

    return output_flat.reshape(N, out_h, out_w, Co)


# Quick verification against PyTorch:
def _verify_reference():
    torch.manual_seed(42)
    N, H, W, Ci, Co, R, S = 1, 6, 6, 3, 2, 3, 3
    x_nhwc = torch.randn(N, H, W, Ci)
    w_nhwc = torch.randn(Co, R, S, Ci)

    our_out = conv2d_reference_im2col(x_nhwc, w_nhwc, stride=1, padding=1)

    # Compare with PyTorch conv2d (NCHW format)
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    w_nchw = w_nhwc.permute(0, 3, 1, 2).contiguous()
    torch_out = torch.nn.functional.conv2d(x_nchw, w_nchw, stride=1, padding=1)
    torch_out_nhwc = torch_out.permute(0, 2, 3, 1)

    torch.testing.assert_close(our_out, torch_out_nhwc, atol=1e-5, rtol=1e-5)
    print("Reference im2col matches PyTorch conv2d!")


if __name__ == "__main__":
    _verify_reference()

# %%
# Gluon Kernel: Convolution with TMA im2col + tcgen05 MMA
# ========================================================
#
# Now we implement a real GPU kernel. This kernel:
#
# - Loads **input tiles** using TMA im2col (hardware-accelerated implicit GEMM)
# - Loads **weight tiles** using standard TMA
# - Computes the matmul using **tcgen05 MMA** (Blackwell 5th-gen Tensor Core)
# - Stores the output using **TMA**
#
# The accumulator lives in **tensor memory** (TMEM), a fast on-chip memory
# space introduced on Blackwell that is directly accessed by the tensor cores.
# The structure follows a standard blocked GEMM with a K-loop over
# ``(r, s, ci_block)`` iterations.
#
# See ``06-tcgen05.py`` for a detailed explanation of tensor memory, tcgen05_mma,
# and tcgen05_commit.

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
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_warps: gl.constexpr,
):
    """
    Implicit GEMM convolution kernel using TMA im2col + tcgen05 MMA.

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

    # ── Accumulator in tensor memory (TMEM) ──────────────────────────────
    # Tensor memory is a fast on-chip memory directly accessed by Blackwell
    # tensor cores.  tcgen05_mma reads/writes the accumulator from TMEM.
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    # ── Barriers ─────────────────────────────────────────────────────────
    # Two barriers: one for TMA loads, one for tcgen05 MMA completion.
    tma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar, count=1)
    mbarrier.init(mma_bar, count=1)
    phase = 0

    # ── K-loop: iterate over (r, s, ci_block) ────────────────────────────
    # We iterate in the natural order: r -> s -> ci_block, so that
    # weight loads are sequential along the K dimension.
    ci_num_blocks = Ci // BLOCK_K
    total_k_iters = R * S * ci_num_blocks

    # use_acc=False on the first iteration zero-initializes the accumulator.
    use_acc = False
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

        # 5. tcgen05 MMA: acc += A_tile @ B_tile^T
        #    A_tile is [BLOCK_M, BLOCK_K] (input pixels x channels)
        #    B_tile is [BLOCK_N, BLOCK_K] (output channels x channels)
        #    Transpose B to get [BLOCK_K, BLOCK_N]
        #    use_acc=False on first iteration zeros the accumulator.
        tcgen05_mma(a_smem, b_smem.permute((1, 0)), acc_tmem, use_acc=use_acc)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase=phase)
        use_acc = True

        phase ^= 1

    mbarrier.invalidate(tma_bar)
    mbarrier.invalidate(mma_bar)

    # ── Load accumulator from TMEM to registers ──────────────────────────
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32,
        (BLOCK_M, BLOCK_N),
        tmem_layout,
        num_warps,
    )
    acc = acc_tmem.load(acc_reg_layout)

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


def conv2d_tma_im2col(input_nhwc, weight, stride=1, padding=0, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4):
    """
    2D convolution using TMA im2col + tcgen05 MMA.

    Args:
        input_nhwc: [N, H, W, Ci] in NHWC format, float16
        weight:     [Co, R, S, Ci] filter tensor, float16
        stride:     convolution stride (default: 1)
        padding:    convolution padding (default: 0)
    """
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
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU (SM 10.x)")
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
# Gluon Kernel: Convolution with TMA im2col + WGMMA (Hopper)
# ===========================================================
#
# Hopper GPUs also support TMA im2col mode, so we can use the same
# hardware-accelerated implicit GEMM approach with WGMMA (warp-group MMA)
# instead of tcgen05 MMA.
#
# Key differences from the Blackwell kernel above:
#
# - **Accumulator in registers**: WGMMA accumulates into distributed
#   registers (``NVMMADistributedLayout``) rather than tensor memory (TMEM).
#   This means no ``allocate_tensor_memory`` or ``get_tmem_reg_layout``.
# - **WGMMA API**: We use ``warpgroup_mma`` / ``warpgroup_mma_wait`` instead
#   of ``tcgen05_mma`` / ``tcgen05_commit``. WGMMA is asynchronous and the
#   accumulator is threaded through the API as a dependency.
# - **Same TMA im2col**: The input loading path is identical -- TMA im2col
#   hardware handles the address computation for both architectures.
#
# See ``05-wgmma.py`` for a detailed explanation of WGMMA layouts and
# the accumulator threading pattern.


@gluon.jit
def conv2d_im2col_wgmma_kernel(
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
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_warps: gl.constexpr,
):
    """
    Implicit GEMM convolution kernel using TMA im2col + WGMMA (Hopper).

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

    # ── Accumulator in registers ──────────────────────────────────────────
    # On Hopper, WGMMA accumulates into distributed registers rather than
    # tensor memory.  pick_wgmma_layout determines the register layout
    # based on the MMA instruction shape and warp configuration.
    mma_layout: gl.constexpr = pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)

    # ── Barrier ───────────────────────────────────────────────────────────
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    # ── K-loop: iterate over (r, s, ci_block) ────────────────────────────
    ci_num_blocks = Ci // BLOCK_K
    total_k_iters = R * S * ci_num_blocks

    for k_iter in range(total_k_iters):
        # Decompose k_iter into (r, s, ci_block)
        ci_block = k_iter % ci_num_blocks
        rs_idx = k_iter // ci_num_blocks
        r = rs_idx // S
        s = rs_idx % S

        # 1. Set up TMA barrier and expect bytes from both loads
        mbarrier.expect(bar, in_desc.block_type.nbytes + weight_desc.block_type.nbytes)

        # 2. Load input tile via TMA im2col (same as Blackwell kernel)
        tma.async_copy_global_to_shared_im2col(
            in_desc,
            [batch_id, out_y * stride_h - pad_h, out_x * stride_w - pad_w, ci_block * BLOCK_K],
            [r.to(tl.int16), s.to(tl.int16)],
            bar,
            a_smem,
        )

        # 3. Load weight tile via standard TMA
        k_offset = r * S * Ci + s * Ci + ci_block * BLOCK_K
        tma.async_copy_global_to_shared(
            weight_desc,
            [pid_n * BLOCK_N, k_offset],
            bar,
            b_smem,
        )

        # 4. Wait for both TMA loads to complete
        mbarrier.wait(bar, phase=phase)

        # 5. WGMMA: acc += A_tile @ B_tile^T
        #    Transpose B via permute, then issue async WGMMA and wait.
        acc = warpgroup_mma(a_smem, b_smem.permute((1, 0)), acc, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc, ))

        phase ^= 1

    mbarrier.invalidate(bar)

    # ── Downcast and store output tile via TMA ────────────────────────────
    c_smem = gl.allocate_shared_memory(dtype, out_desc.block_shape, out_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(out_desc, [offs_m, pid_n * BLOCK_N], c_smem)
    tma.store_wait(pendings=0)


# %%
# Host-Side Launcher (WGMMA)
# ===========================
#
# The host side is nearly identical to the Blackwell launcher.  The only
# difference is that we call the WGMMA kernel instead of the tcgen05 kernel.
# TMA descriptor setup (including im2col) is the same for both architectures.


def conv2d_tma_im2col_wgmma(input_nhwc, weight, stride=1, padding=0, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4):
    """
    2D convolution using TMA im2col + WGMMA (Hopper).

    Args:
        input_nhwc: [N, H, W, Ci] in NHWC format, float16
        weight:     [Co, R, S, Ci] filter tensor, float16
        stride:     convolution stride (default: 1)
        padding:    convolution padding (default: 0)
    """
    N, H, W, Ci = input_nhwc.shape
    Co, R, S, Ci_w = weight.shape
    assert Ci == Ci_w, "Channel mismatch"
    assert Ci % BLOCK_K == 0, f"Ci={Ci} must be divisible by BLOCK_K={BLOCK_K}"

    out_h = (H + 2 * padding - R) // stride + 1
    out_w = (W + 2 * padding - S) // stride + 1
    M_GEMM = N * out_h * out_w
    N_GEMM = Co

    output = torch.empty(M_GEMM, Co, device=input_nhwc.device, dtype=input_nhwc.dtype)

    # ── TMA im2col descriptor for input (same as Blackwell) ───────────────
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
    weight_2d = weight.reshape(Co, R * S * Ci)
    weight_block_shape = [BLOCK_N, BLOCK_K]
    weight_layout = gl.NVMMASharedLayout.get_default_for(weight_block_shape, gl.float16)
    weight_desc = TensorDescriptor.from_tensor(weight_2d, weight_block_shape, weight_layout)

    # ── TMA descriptor for output ────────────────────────────────────────
    out_block_shape = [BLOCK_M, BLOCK_N]
    out_layout = gl.NVMMASharedLayout.get_default_for(out_block_shape, gl.float16)
    out_desc = TensorDescriptor.from_tensor(output, out_block_shape, out_layout)

    # ── Launch kernel ────────────────────────────────────────────────────
    grid = (triton.cdiv(M_GEMM, BLOCK_M), triton.cdiv(N_GEMM, BLOCK_N))

    conv2d_im2col_wgmma_kernel[grid](
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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )

    return output.reshape(N, out_h, out_w, Co)


# %%
# Testing (WGMMA / Hopper)
# =========================


@pytest.mark.parametrize("N", [1, 4])
@pytest.mark.parametrize("H,W", [(16, 16)])
@pytest.mark.parametrize("Ci,Co", [(64, 64)])
@pytest.mark.parametrize("R,S", [(3, 3), (1, 1)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper GPU (SM 9.x)")
def test_conv2d_im2col_wgmma(N, H, W, Ci, Co, R, S, stride, padding):
    torch.manual_seed(0)

    x_nhwc = torch.randn(N, H, W, Ci, device="cuda", dtype=torch.float16)
    w_nhwc = torch.randn(Co, R, S, Ci, device="cuda", dtype=torch.float16)

    # Our WGMMA kernel
    triton_out = conv2d_tma_im2col_wgmma(x_nhwc, w_nhwc, stride=stride, padding=padding)

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
# 4. **Blackwell kernel (tcgen05)**: The kernel follows a standard blocked
#    GEMM pattern with the accumulator in tensor memory (TMEM):
#
#    - Each program computes one (BLOCK_M x BLOCK_N) output tile
#    - The K-loop iterates over filter positions (r, s) and channel blocks
#    - TMA im2col loads input, standard TMA loads weights
#    - tcgen05_mma accumulates into tensor memory (TMEM)
#    - After the K-loop, results are loaded from TMEM and stored via TMA
#
# 5. **Hopper kernel (WGMMA)**: The same TMA im2col approach works on Hopper
#    with WGMMA as the compute engine:
#
#    - The accumulator lives in distributed registers (not TMEM)
#    - ``warpgroup_mma`` / ``warpgroup_mma_wait`` replace tcgen05 APIs
#    - TMA im2col input loading is identical to the Blackwell kernel
#    - No tensor memory allocation is needed
#
# For a production-quality implementation with warp specialization and
# pipelining, see ``examples/gluon/02-convolution.py``.

if __name__ == "__main__":
    torch.manual_seed(0)
    N, H, W, Ci, Co, R, S = 4, 16, 16, 64, 64, 3, 3
    stride, padding = 1, 1

    x_nhwc = torch.randn(N, H, W, Ci, device="cuda", dtype=torch.float16)
    w_nhwc = torch.randn(Co, R, S, Ci, device="cuda", dtype=torch.float16)

    # PyTorch reference
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    w_nchw = w_nhwc.permute(0, 3, 1, 2).contiguous()
    torch_out = torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding)
    torch_out_nhwc = torch_out.permute(0, 2, 3, 1)

    if is_blackwell():
        print("\nRunning TMA im2col + tcgen05 convolution (Blackwell)...")
        triton_out = conv2d_tma_im2col(x_nhwc, w_nhwc, stride=stride, padding=padding)

        max_err = (triton_out - torch_out_nhwc).abs().max().item()
        print(f"Conv2d: N={N}, H={H}, W={W}, Ci={Ci}, Co={Co}, R={R}, S={S}, "
              f"stride={stride}, pad={padding}")
        print(f"Output shape: {list(triton_out.shape)}")
        print(f"Max absolute error vs PyTorch: {max_err:.6f}")
        print("PASSED!" if max_err < 0.02 else "FAILED!")

    if is_hopper():
        print("\nRunning TMA im2col + WGMMA convolution (Hopper)...")
        triton_out_wgmma = conv2d_tma_im2col_wgmma(x_nhwc, w_nhwc, stride=stride, padding=padding)

        max_err = (triton_out_wgmma - torch_out_nhwc).abs().max().item()
        print(f"Conv2d: N={N}, H={H}, W={W}, Ci={Ci}, Co={Co}, R={R}, S={S}, "
              f"stride={stride}, pad={padding}")
        print(f"Output shape: {list(triton_out_wgmma.shape)}")
        print(f"Max absolute error vs PyTorch: {max_err:.6f}")
        print("PASSED!" if max_err < 0.02 else "FAILED!")
