"""
Convolution via Implicit GEMM with TMA im2col
==============================================

This tutorial explains how to implement 2D convolution as a matrix multiplication
(GEMM) using the **im2col** transformation, and how NVIDIA's TMA (Tensor Memory
Accelerator) im2col hardware mode makes this highly efficient on Hopper+ GPUs.

We cover:
1. How 2D convolution works (the sliding window)
2. The im2col algorithm that reshapes convolution into GEMM
3. A convolution kernel using TMA im2col + MMA (works on both Hopper and Blackwell)

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
# Implicit GEMM with TMA im2col
# ==============================
#
# **Explicit im2col** physically constructs the matrix A in memory. This is
# simple but wastes memory because overlapping patches duplicate input data.
# For an R x S filter, each input element may be copied up to R * S times!
#
# **Implicit GEMM** avoids materializing A.  Instead, during the GEMM
# K-loop we compute input addresses on-the-fly for each filter position
# ``(r, s)`` and channel block.  On Hopper+ GPUs, TMA has a dedicated
# **im2col mode** that performs this address computation entirely in
# hardware.  We configure a ``TensorDescriptorIm2Col`` once (see the
# launcher), and each TMA load takes a filter offset ``[r, s]`` that
# tells the hardware which spatial tap to gather from.  TMA automatically
# handles striding between output positions, zero-padding for
# out-of-bounds accesses, and wrapping across batch boundaries — all
# without any register-level index arithmetic.

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


@gluon.jit
def decompose_m_offset(pid_m, BLOCK_M: gl.constexpr, out_h: gl.constexpr, out_w: gl.constexpr):
    """Map M-tile index to (offs_m, batch, out_y, out_x) for TMA im2col coordinates."""
    offs_m = pid_m * BLOCK_M
    batch_id = offs_m // (out_h * out_w)
    m_residual = offs_m % (out_h * out_w)
    out_y = m_residual // out_w
    out_x = m_residual % out_w
    return offs_m, batch_id, out_y, out_x


@gluon.jit
def init_accumulator(in_desc, weight_desc, MMAImpl, dtype, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr):
    """Allocate shared-memory tiles, MMA accumulator, and TMA barrier."""
    a_smem = gl.allocate_shared_memory(dtype, in_desc.block_shape, in_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, weight_desc.block_shape, weight_desc.layout)
    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    tma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar, count=1)
    return a_smem, b_smem, mma, tma_bar


@gluon.jit
def store_output_tile(mma, dtype, out_desc, offs_m, offs_n):
    """Wait for the final MMA, downcast, and write the output tile via TMA."""
    mma = mma.wait_num_outstanding(0)
    acc, mma = mma.take_result()
    c_smem = gl.allocate_shared_memory(dtype, out_desc.block_shape, out_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(out_desc, [offs_m, offs_n], c_smem)
    tma.store_wait(pendings=0)


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
    """
    dtype: gl.constexpr = in_desc.dtype

    # ── Implicit GEMM algorithm ──────────────────────────────────────────
    #
    #   Recall:  y[n, oh, ow, co] = SUM over (r, s, ci) of
    #       input[n, oh*stride+r-pad, ow*stride+s-pad, ci] * weight[co, r, s, ci]
    #   where:
    #     oh = output height index (output row), mapped to out_y below
    #     ow = output width index  (output col), mapped to out_x below
    #
    #   We tile this as a GEMM with dimensions:
    #     M      = N * out_h * out_w   (output spatial positions)
    #     N_gemm = Co                  (output channels)
    #     K      = R * S * Ci          (reduction dimension)
    #
    #   for each M-tile:                          ← pid_m
    #       for each N-tile:                      ← pid_n
    #           acc = zeros(BLOCK_M, BLOCK_N)
    #           for r in range(R):
    #               for s in range(S):
    #                   for ci_blk in range(Ci // BLOCK_K):
    #                       # BLOCK_M output positions × BLOCK_K input channels
    #                       A[BLOCK_M, BLOCK_K] = load_input[batch, oh*stride+r-pad, ow*stride+s-pad, ci_blk*BLOCK_K:…]
    #                       # BLOCK_N output channels × BLOCK_K input channels
    #                       B[BLOCK_N, BLOCK_K] = load_weight[co_start:…, r, s, ci_blk*BLOCK_K:…]
    #                       acc += A @ B^T                  # [BLOCK_M, BLOCK_N]
    #           store output[…] = acc
    #
    # ─────────────────────────────────────────────────────────────────────
    # Gluon implementation (follows the pseudocode above line-by-line):

    # ┌ for each M-tile / N-tile:
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    offs_m, batch_id, out_y, out_x = decompose_m_offset(pid_m, BLOCK_M, out_h, out_w)

    # │ acc = zeros(BLOCK_M, BLOCK_N)
    a_smem, b_smem, mma, tma_bar = init_accumulator(
        in_desc, weight_desc, MMAImpl, dtype, BLOCK_M, BLOCK_N, num_warps)
    phase = 0

    # │ for r in range(R): for s in range(S): for ci_blk in range(Ci//BLOCK_K):
    ci_num_blocks = Ci // BLOCK_K
    total_k_iters = R * S * ci_num_blocks
    for k_iter in range(total_k_iters):
        ci_block = k_iter % ci_num_blocks
        rs_idx = k_iter // ci_num_blocks
        r = rs_idx // S
        s = rs_idx % S

        # │   A = load_input[batch, oh*stride+r-pad, ow*stride+s-pad, ci_blk*BLOCK_K:…]
        k_offset = r * S * Ci + s * Ci + ci_block * BLOCK_K
        mbarrier.expect(tma_bar, in_desc.block_type.nbytes + weight_desc.block_type.nbytes)
        tma.async_copy_global_to_shared_im2col(
            in_desc,
            [batch_id, out_y * stride_h - pad_h, out_x * stride_w - pad_w, ci_block * BLOCK_K],
            [r.to(tl.int16), s.to(tl.int16)],
            tma_bar, a_smem,
        )
        # │   B = load_weight[co_start:…, r, s, ci_blk*BLOCK_K:…]
        tma.async_copy_global_to_shared(
            weight_desc, [pid_n * BLOCK_N, k_offset], tma_bar, b_smem)
        mbarrier.wait(tma_bar, phase=phase)

        # │   acc += A @ B^T
        mma = mma.wait_num_outstanding(0)
        mma = mma.issue_async_mma(a_smem, b_smem.permute((1, 0)))

        phase ^= 1

    mbarrier.invalidate(tma_bar)

    # └ store output[…] = acc
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
    assert Ci % BLOCK_K == 0, f"Ci={Ci} must be divisible by BLOCK_K={BLOCK_K}"

    out_h = (H + 2 * padding - R) // stride + 1
    out_w = (W + 2 * padding - S) // stride + 1
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
