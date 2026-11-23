"""
Scaled MFMA and WMMA
====================

Block scaling is a way to quantize a tensor into low-precision values and one
scale factor per block, as in the OCP microscaling (MX) formats. CDNA4 and
CDNA5 GPUs support matrix multiplication for MX formats, with scaled MFMA and
scaled WMMA instructions.

In this tutorial, we will show to use them in Gluon. And use an MLP-like kernel,
where we multiply fp8 activations by mxfp4 weights, as an example.
"""

import importlib
import pytest
import sys
import torch
import triton
from pathlib import Path
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4, cdna5
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
t1 = importlib.import_module("amd.01-buffer-ops")
t4 = importlib.import_module("amd.04-mfma-wmma")


def is_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx950"


def is_cdna5():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx1250"


if __name__ == "__main__" and not (is_cdna4() or is_cdna5()):
    raise RuntimeError("This tutorial requires an AMD CDNA4 or CDNA5 GPU")

# A warp has 64 lanes on CDNA4 GPUs and 32 lanes on CDNA5 GPUs.
warp_size = gl.constexpr(64 if is_cdna4() else 32)


@gluon.jit
def offs_2d(rows: gl.constexpr, cols: gl.constexpr, stride, layout: gl.constexpr):
    r = gl.arange(0, rows, gl.SliceLayout(1, layout))
    c = gl.arange(0, cols, gl.SliceLayout(0, layout))
    return r[:, None] * stride + c[None, :]


def random_fp8(M, K):
    return torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)


def random_mxfp4(N, K):
    values = MXFP4Tensor(size=(N, K), device="cuda").random()
    scales = MXScaleTensor(size=(N, K // 32), device="cuda").random(low=1 / 128, high=2.0)
    ref = values.to(torch.float32) * scales.to(torch.float32).repeat_interleave(32, dim=1)
    return values.to_packed_tensor(dim=1), scales.data, ref


def random_operand(rows, K, kind):
    if kind == "mxfp4":
        values = MXFP4Tensor(size=(rows, K), device="cuda").random()
        data, ref = values.to_packed_tensor(dim=1), values.to(torch.float32)
    else:
        data = random_fp8(rows, K)
        ref = data.float()
    if kind == "fp8":
        return data, None, ref
    scales = MXScaleTensor(size=(rows, K // 32), device="cuda").random(low=1 / 128, high=2.0)
    return data, scales.data, ref * scales.to(torch.float32).repeat_interleave(32, dim=1)


# %%
# CDNA4 Scaled MFMA
# -----------------
#
# The scaled MFMA instruction we will use is
# `v_mfma_scale_f32_16x16x128_f8f6f4`. It multiplies a 16x128 tile of A by a
# 128x16 tile of B, applies the scales, and accumulates a 16x16 fp32 tile. Gluon
# exposes it as `mfma_scaled`, which takes each operand with its scales. The 2
# operands can be fp8, fp6, or fp4 in any combination. We can also pass `None`
# to use uniform scales (all ones).
#
# The accumulator uses `AMDMFMALayout` with `instr_shape=[16, 16, 128]`. Both
# operands use `k_width=16`, so each lane holds 16 consecutive bytes along K. A
# packed fp4 operand uses the same parent layout as an fp8 operand.
#
# The scales also need their own layout. But there is no dedicated layout class
# for them, so the kernel uses the generic `DistributedLinearLayout` to describe
# them. Writing them manually can be tedious, so we use `get_mfma_scale_layout`
# to compute them from the operand layout.
#
# Before diving into the MFMA, let's look at how fp4 operands are stored.
# Memory is addressed in bytes, so we pack 2 fp4 values into one byte along K,
# with the first value in the low 4 bits. An [N, K] fp4 tensor is stored as an
# [N, K // 2] uint8 tensor. Triton provides some utilities in
# `triton.tools.mxfp` to pack fp4 values, such as `to_packed_tensor`. For MFMA,
# a packed operand has K // 2 bytes along K.

if __name__ == "__main__" and is_cdna4():
    values = MXFP4Tensor(data=torch.tensor([[1.0, 2.0, 0.5, -6.0]], device="cuda"))
    print([hex(x) for x in values.to_packed_tensor(dim=1)[0].tolist()])


@gluon.jit
def scaled_mfma_kernel(a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                       A_FORMAT: gl.constexpr, B_FORMAT: gl.constexpr):
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 1])
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=16)
    b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=16)
    A_K: gl.constexpr = K // 2 if A_FORMAT == "e2m1" else K
    B_K: gl.constexpr = K // 2 if B_FORMAT == "e2m1" else K

    a = gl.load(a_ptr + offs_2d(M, A_K, A_K, a_layout))
    bk = gl.arange(0, B_K, gl.SliceLayout(1, b_layout))
    bn = gl.arange(0, N, gl.SliceLayout(0, b_layout))
    b = gl.load(b_ptr + bn[None, :] * B_K + bk[:, None])
    a_scale = None
    if a_scale_ptr is not None:
        a_scale_layout: gl.constexpr = cdna4.get_mfma_scale_layout(a_layout, [M, K // 32])
        a_scale = gl.load(a_scale_ptr + offs_2d(M, K // 32, K // 32, a_scale_layout))
    b_scale = None
    if b_scale_ptr is not None:
        b_scale_layout: gl.constexpr = cdna4.get_mfma_scale_layout(b_layout, [N, K // 32])
        b_scale = gl.load(b_scale_ptr + offs_2d(N, K // 32, K // 32, b_scale_layout))

    acc = gl.zeros([M, N], gl.float32, mfma_layout)
    acc = cdna4.mfma_scaled(a, a_scale, A_FORMAT, b, b_scale, B_FORMAT, acc)
    gl.store(c_ptr + offs_2d(M, N, N, mfma_layout), acc)


@pytest.mark.parametrize("M, N, K", [(32, 32, 128), (64, 128, 256)])
@pytest.mark.parametrize("a_kind", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("b_kind", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_scaled_mfma(M, N, K, a_kind, b_kind):
    A, A_scale, A_ref = random_operand(M, K, a_kind)
    B, B_scale, B_ref = random_operand(N, K, b_kind)
    a_format = "e2m1" if a_kind == "mxfp4" else "e4m3"
    b_format = "e2m1" if b_kind == "mxfp4" else "e4m3"
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    scaled_mfma_kernel[(1, )](A, A_scale, B, B_scale, C, M, N, K, a_format, b_format, num_warps=1)
    torch.testing.assert_close(A_ref @ B_ref.T, C, atol=1e-2, rtol=1e-2)


# %%
# The scales are a small tensor, with one byte for every 32 values, so we need
# to pay some attention to how they are consumed.
# `v_mfma_scale_f32_16x16x128_f8f6f4` takes the scales of A and B as two 32-bit
# registers, although one MFMA only needs one byte of each. The instruction
# carries a 2-bit index for each scale register, with the low bit `op_sel` and
# the high bit `op_sel_hi`. The index picks which byte of the 32-bit register
# to use. So one 32-bit register can hold 4 scales, and 4 MFMAs can each pick a
# different byte of it. Then one load, from global or shared memory, can bring 4
# scales into a register, as long as a lane has 4 scales and they sit next to
# each other in memory.
#
# To let each warp work on more than one MFMA tile, we use the `tiles_per_warp`
# argument of `AMDMFMALayout`. For example, `tiles_per_warp=[2, 2]` means each
# warp computes 2 neighboring tiles along M and 2 along N. A lane then holds 2
# scales for each scaled operand.
#
# We have now filled half of each 32-bit scale register, and the repeated MFMAs
# along the K dimension can fill the other half. A block of 256 values along the
# K dimension takes 2 MFMAs, given the K dimension of the MFMA instruction is
# 128. Each MFMA has its own scale.

if __name__ == "__main__" and is_cdna4():
    mfma_layout = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 1], tiles_per_warp=[2, 2])
    b_layout = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=16)
    # Operand B has shape [256, 32], and its scales have shape [32, 8].
    print(cdna4.get_mfma_scale_layout(b_layout, [32, 8]).format_tensor_view([32, 8]))

# %%
# ```
# [[ T0:0, T16:0, T32:0, T48:0,  T0:1, T16:1, T32:1, T48:1]
# [  T1:0, T17:0, T33:0, T49:0,  T1:1, T17:1, T33:1, T49:1]
# ...
# [ T15:0, T31:0, T47:0, T63:0, T15:1, T31:1, T47:1, T63:1]
# [  T0:2, T16:2, T32:2, T48:2,  T0:3, T16:3, T32:3, T48:3]
# ...
# [ T15:2, T31:2, T47:2, T63:2, T15:3, T31:3, T47:3, T63:3]]
# ```
#
# The printout shows the scales of B for one warp and one block of 256 values
# along K. Each lane holds the scales of 2 rows, from its 2 tiles along N, and 2
# columns, from its 2 MFMAs along K. The scales of A look the same along M. Now
# we can fill them into a single 32-bit register. But we can also see that
# these 4 values are not next to each other. This means we have to use 4 loads
# and pack them together.
#
# To load them more efficiently, we use a technique called "scale preshuffle".
# On the host, it reorders every group of 32 rows, so that the 4 scales of each
# lane sit next to each other, and the scales of consecutive lanes follow each
# other. A lane then reads its 4 scales with one 4-byte read, already packed,
# and a warp reads one contiguous piece of memory, which also helps when the
# scales are loaded straight from global memory. In the kernel, we will undo the
# preshuffle to restore the original order of the scales, as the MFMA requires.


def preshuffle_scales_cdna4(scales):
    N, K_SCALE = scales.shape
    scales = torch.nn.functional.pad(scales, (0, 0, 0, -N % 32))
    # Split each group of 32 rows into [tile, row], and each block of 8 scales
    # along K into [MFMA, K offset]. A lane holds one row and one K offset.
    scales = scales.view(-1, 2, 16, K_SCALE // 8, 2, 4)
    # Order each group as [block, K offset, row, tile, MFMA]. The 4 scales of a
    # lane then sit next to each other, and the lanes follow each other.
    scales = scales.permute(0, 3, 5, 2, 1, 4)
    return scales.reshape(-1, K_SCALE * 32)


@gluon.jit
def unshuffle_scales_cdna4(s_buf):
    ROWS: gl.constexpr = s_buf.shape[0] * 32
    K_SCALE: gl.constexpr = s_buf.shape[1] // 32
    # Split each row of a group into [block, K offset, row, tile, MFMA], as
    # written by `preshuffle_scales_cdna4`, and move the dimensions back to
    # [tile, row] and [block, MFMA, K offset].
    s_buf = s_buf.reshape([ROWS // 32, K_SCALE // 8, 4, 16, 2, 2]).permute([0, 4, 3, 1, 5, 2])
    return s_buf.reshape([ROWS, K_SCALE])


@gluon.jit
def load_scales_cdna4(scale_ptr, ROWS: gl.constexpr, K_SCALE: gl.constexpr, layout: gl.constexpr,
                      PRESHUFFLE: gl.constexpr):
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    if PRESHUFFLE:
        gmem_layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])
        s = gl.load(scale_ptr + offs_2d(ROWS // 32, K_SCALE * 32, K_SCALE * 32, gmem_layout))
        s_smem = gl.allocate_shared_memory(gl.uint8, [ROWS // 32, K_SCALE * 32], smem_layout, s)
        s_view = unshuffle_scales_cdna4(s_smem)
    else:
        gmem_layout: gl.constexpr = gl.BlockedLayout([1, 4], [32, 2], [4, 1], [1, 0])
        s = gl.load(scale_ptr + offs_2d(ROWS, K_SCALE, K_SCALE, gmem_layout))
        s_view = gl.allocate_shared_memory(gl.uint8, [ROWS, K_SCALE], smem_layout, s)
    return s_view.load(layout)


@gluon.jit
def scaled_mfma_smem_kernel(a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr,
                            K: gl.constexpr, A_FORMAT: gl.constexpr, B_FORMAT: gl.constexpr, PRESHUFFLE: gl.constexpr):
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[2, 2], tiles_per_warp=[2, 2])
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=16)
    b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=16)
    A_K: gl.constexpr = K // 2 if A_FORMAT == "e2m1" else K
    B_K: gl.constexpr = K // 2 if B_FORMAT == "e2m1" else K

    a = gl.load(a_ptr + offs_2d(M, A_K, A_K, a_layout))
    bk = gl.arange(0, B_K, gl.SliceLayout(1, b_layout))
    bn = gl.arange(0, N, gl.SliceLayout(0, b_layout))
    b = gl.load(b_ptr + bn[None, :] * B_K + bk[:, None])
    a_scale = None
    if a_scale_ptr is not None:
        a_scale_layout: gl.constexpr = cdna4.get_mfma_scale_layout(a_layout, [M, K // 32])
        a_scale = load_scales_cdna4(a_scale_ptr, M, K // 32, a_scale_layout, PRESHUFFLE)
    b_scale = None
    if b_scale_ptr is not None:
        b_scale_layout: gl.constexpr = cdna4.get_mfma_scale_layout(b_layout, [N, K // 32])
        b_scale = load_scales_cdna4(b_scale_ptr, N, K // 32, b_scale_layout, PRESHUFFLE)

    acc = gl.zeros([M, N], gl.float32, mfma_layout)
    acc = cdna4.mfma_scaled(a, a_scale, A_FORMAT, b, b_scale, B_FORMAT, acc)
    gl.store(c_ptr + offs_2d(M, N, N, mfma_layout), acc)


@pytest.mark.parametrize("M, N, K", [(64, 64, 256), (128, 128, 512)])
@pytest.mark.parametrize("a_kind", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("b_kind", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("preshuffle", [False, True])
@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_scaled_mfma_smem(M, N, K, a_kind, b_kind, preshuffle):
    A, A_scale, A_ref = random_operand(M, K, a_kind)
    B, B_scale, B_ref = random_operand(N, K, b_kind)
    if preshuffle:
        A_scale = None if A_scale is None else preshuffle_scales_cdna4(A_scale)
        B_scale = None if B_scale is None else preshuffle_scales_cdna4(B_scale)
    a_format = "e2m1" if a_kind == "mxfp4" else "e4m3"
    b_format = "e2m1" if b_kind == "mxfp4" else "e4m3"
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    scaled_mfma_smem_kernel[(1, )](A, A_scale, B, B_scale, C, M, N, K, a_format, b_format, preshuffle)
    torch.testing.assert_close(A_ref @ B_ref.T, C, atol=1e-2, rtol=1e-2)


if __name__ == "__main__" and is_cdna4():
    A = random_fp8(64, 256)
    B, B_scale, _ = random_mxfp4(64, 256)
    C = torch.empty(64, 64, device="cuda", dtype=torch.float32)
    for preshuffle in [False, True]:
        scales = preshuffle_scales_cdna4(B_scale) if preshuffle else B_scale
        kernel = scaled_mfma_smem_kernel[(1, )](A, None, B, scales, C, 64, 64, 256, "e4m3", "e2m1", preshuffle)
        print(f"preshuffle={preshuffle}:")
        for instr in t1.get_instructions(kernel):
            if instr.startswith("ds_read"):
                print(f"  {instr}")
            elif instr.startswith("v_mfma_scale"):
                # Only show the scale operands and the byte selection.
                print("  v_mfma_scale ...," + instr.split(",", 4)[4])

# %%
# ```
# preshuffle=False:
#   ds_read_u8 v1, v0 offset:132
#   ds_read_u8 v2, v0 offset:128
#   ds_read_u8 v3, v0
#   ds_read_u8 v0, v0 offset:4
#   v_mfma_scale ..., v20, v19 op_sel_hi:[0,0,0] cbsz:4
#   v_mfma_scale ..., v20, v19 op_sel_hi:[1,0,0] cbsz:4
#   ...
# preshuffle=True:
#   ds_read_b32 v19, v0
#   v_mfma_scale ..., v19, v20 op_sel_hi:[0,0,0] cbsz:4
#   v_mfma_scale ..., v19, v20 op_sel_hi:[1,0,0] cbsz:4
#   ...
#   v_mfma_scale ..., v19, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4
#   v_mfma_scale ..., v19, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4
#   ...
# ```
#
# Here A is fp8 without scales, and B is mxfp4. With plain scales, a lane reads
# its 4 scales of B with 4 `ds_read_u8`, and packs them into `v20`. With
# preshuffled scales, one `ds_read_b32` reads them into `v19` at once. In both
# cases, the 8 MFMAs take B's scales from the first scale register, and the
# first elements of `op_sel` and `op_sel_hi` form the index of the byte.

# %%
# Now let's build a matmul using MX data types. This example mimics an MLP-style
# matmul, where we multiply activations A by weights W. We use fp8 for the
# activations and mxfp4 for the weights. In practice, the weights are quantized
# beforehand, and also preshuffled before we serve the model.


@gluon.aggregate
class MatmulConfig:
    # Tensor shapes
    M: gl.tensor | gl.constexpr
    N: gl.tensor | gl.constexpr
    K: gl.constexpr
    # Operand formats
    A_FORMAT: gl.constexpr
    W_FORMAT: gl.constexpr
    # Tile size
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    # Kernel config
    NUM_BUFFERS: gl.constexpr
    PRESHUFFLE: gl.constexpr
    PRESHUFFLE_FACTOR: gl.constexpr


@gluon.aggregate
class MfmaLayouts:
    # Operand A's global memory, shared memory, and register layouts
    a_gmem_layout: gl.constexpr
    a_smem_layout: gl.constexpr
    a_reg_layout: gl.constexpr
    # Operand W's global memory, shared memory, and register layouts
    w_gmem_layout: gl.constexpr
    w_smem_layout: gl.constexpr
    w_reg_layout: gl.constexpr
    # The scales' global memory, shared memory, and register layouts
    ws_gmem_layout: gl.constexpr
    ws_smem_layout: gl.constexpr
    ws_reg_layout: gl.constexpr
    # Accumulator's register layout
    acc_layout: gl.constexpr


@gluon.constexpr_function
def get_mx_mfma_layouts(preshuffle):
    # Assume the 128x256x256 block size and 4 warps per CTA
    # Operand A has shape [128, 256] in fp8
    # Operand W has shape [256, 128] in packed fp4, so transposed
    acc_layout = gl.amd.AMDMFMALayout(  #
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[2, 2], tiles_per_warp=[2, 2])
    a_reg_layout = gl.DotOperandLayout(0, acc_layout, 16)
    w_reg_layout = gl.DotOperandLayout(1, acc_layout, 16)

    a_gmem_layout = t4.get_gmem_layout_for_async_copies([128, 256], gl.float8e4nv)
    a_smem_layout = t4.get_smem_layout_for_async_copies([128, 256], gl.float8e4nv)
    w_gmem_layout = t4.get_gmem_layout_for_async_copies([256, 128], gl.uint8)
    w_smem_layout = t4.get_smem_layout_for_async_copies([256, 128], gl.uint8)

    if preshuffle:
        ws_gmem_layout = gl.BlockedLayout([1, 4], [1, warp_size], [4, 1], [1, 0])
    else:
        ws_gmem_layout = gl.BlockedLayout([1, 4], [warp_size // 2, 2], [4, 1], [1, 0])
    ws_smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    ws_reg_layout = cdna4.get_mfma_scale_layout(w_reg_layout, [256, 8])
    return MfmaLayouts(a_gmem_layout, a_smem_layout, a_reg_layout,  #
                       w_gmem_layout, w_smem_layout, w_reg_layout,  #
                       ws_gmem_layout, ws_smem_layout, ws_reg_layout,  #
                       acc_layout)


@gluon.jit
def matmul_cdna4_ld_gmem(cfg, k, a_desc, w_desc, ws_desc, a_offs, w_offs, ws_offs, a_smem, w_smem, ws_smem):
    i = k % cfg.NUM_BUFFERS
    cdna4.buffer.load_to_shared(a_smem.index(i), a_desc, a_offs, soffset=k * cfg.BLOCK_K)
    cdna4.buffer.load_to_shared(w_smem.index(i), w_desc, w_offs, soffset=k * cfg.BLOCK_K // 2)
    cdna4.buffer.load_to_shared(ws_smem.index(i), ws_desc, ws_offs, soffset=k * ws_smem.shape[2])
    cdna4.async_copy.commit_group()


@gluon.jit
def matmul_cdna4_ld_smem(cfg, k, a_smem, w_smem, ws_smem, layouts):
    a = a_smem.index(k % cfg.NUM_BUFFERS).load(layouts.a_reg_layout)
    w = w_smem.index(k % cfg.NUM_BUFFERS).permute([1, 0]).load(layouts.w_reg_layout)
    ws_buf = ws_smem.index(k % cfg.NUM_BUFFERS)
    if cfg.PRESHUFFLE:
        ws_buf = unshuffle_scales_cdna4(ws_buf)
    w_scale = ws_buf.load(layouts.ws_reg_layout)
    return a, w, w_scale


@gluon.jit
def matmul_cdna4_kernel(a_ptr, w_ptr, w_scale_ptr, y_ptr, M, N, K: gl.constexpr,  #
                        PRESHUFFLE: gl.constexpr, NUM_BUFFERS: gl.constexpr = 2):
    cfg = MatmulConfig(M, N, K, A_FORMAT="e4m3", W_FORMAT="e2m1",  #
                       BLOCK_M=128, BLOCK_N=256, BLOCK_K=256,  #
                       NUM_BUFFERS=NUM_BUFFERS, PRESHUFFLE=PRESHUFFLE, PRESHUFFLE_FACTOR=32 if PRESHUFFLE else 1)
    layouts = get_mx_mfma_layouts(cfg.PRESHUFFLE)

    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(cfg.N, cfg.BLOCK_N)
    off_m = pid // num_pid_n * cfg.BLOCK_M
    off_n = pid % num_pid_n * cfg.BLOCK_N

    a_desc = t1.make_buffer_descriptor(a_ptr, off_m, cfg.M, cfg.K, cfg.BLOCK_M)
    w_desc = t1.make_buffer_descriptor(w_ptr, off_n, cfg.N, cfg.K // 2, cfg.BLOCK_N)
    ws_desc = t1.make_buffer_descriptor(w_scale_ptr, off_n // cfg.PRESHUFFLE_FACTOR,
                                        gl.cdiv(cfg.N, cfg.PRESHUFFLE_FACTOR), cfg.K // 32 * cfg.PRESHUFFLE_FACTOR,
                                        cfg.BLOCK_N // cfg.PRESHUFFLE_FACTOR)
    a_offs = offs_2d(cfg.BLOCK_M, cfg.BLOCK_K, cfg.K, layouts.a_gmem_layout)
    w_offs = offs_2d(cfg.BLOCK_N, cfg.BLOCK_K // 2, cfg.K // 2, layouts.w_gmem_layout)
    ws_offs = offs_2d(cfg.BLOCK_N // cfg.PRESHUFFLE_FACTOR, cfg.BLOCK_K // 32 * cfg.PRESHUFFLE_FACTOR,
                      cfg.K // 32 * cfg.PRESHUFFLE_FACTOR, layouts.ws_gmem_layout)
    a_smem = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [cfg.NUM_BUFFERS, cfg.BLOCK_M, cfg.BLOCK_K],
                                       layouts.a_smem_layout)
    w_smem = gl.allocate_shared_memory(w_ptr.dtype.element_ty, [cfg.NUM_BUFFERS, cfg.BLOCK_N, cfg.BLOCK_K // 2],
                                       layouts.w_smem_layout)
    ws_smem = gl.allocate_shared_memory(
        w_scale_ptr.dtype.element_ty,
        [cfg.NUM_BUFFERS, cfg.BLOCK_N // cfg.PRESHUFFLE_FACTOR, cfg.BLOCK_K // 32 * cfg.PRESHUFFLE_FACTOR],
        layouts.ws_smem_layout)

    num_k = gl.cdiv(cfg.K, cfg.BLOCK_K)
    for i in gl.static_range(cfg.NUM_BUFFERS - 1):
        matmul_cdna4_ld_gmem(cfg, i, a_desc, w_desc, ws_desc, a_offs, w_offs, ws_offs, a_smem, w_smem, ws_smem)

    acc = gl.zeros([cfg.BLOCK_M, cfg.BLOCK_N], gl.float32, layouts.acc_layout)
    for k in range(num_k - (cfg.NUM_BUFFERS - 1)):
        matmul_cdna4_ld_gmem(cfg, k + cfg.NUM_BUFFERS - 1, a_desc, w_desc, ws_desc, a_offs, w_offs, ws_offs, a_smem,
                             w_smem, ws_smem)
        cdna4.async_copy.wait_group(cfg.NUM_BUFFERS - 1)
        a, w, w_scale = matmul_cdna4_ld_smem(cfg, k, a_smem, w_smem, ws_smem, layouts)
        acc = cdna4.mfma_scaled(a, None, cfg.A_FORMAT, w, w_scale, cfg.W_FORMAT, acc)

    for i in gl.static_range(cfg.NUM_BUFFERS - 1):
        cdna4.async_copy.wait_group(cfg.NUM_BUFFERS - 2 - i)
        a, w, w_scale = matmul_cdna4_ld_smem(cfg, num_k - (cfg.NUM_BUFFERS - 1) + i, a_smem, w_smem, ws_smem, layouts)
        acc = cdna4.mfma_scaled(a, None, cfg.A_FORMAT, w, w_scale, cfg.W_FORMAT, acc)

    y_desc = t1.make_buffer_descriptor(y_ptr + off_n, off_m, cfg.M, cfg.N, cfg.BLOCK_M)
    rows = gl.arange(0, cfg.BLOCK_M, gl.SliceLayout(1, layouts.acc_layout))
    cols = gl.arange(0, cfg.BLOCK_N, gl.SliceLayout(0, layouts.acc_layout))
    mask = (off_n + cols < cfg.N)[None, :]
    cdna4.buffer.store(acc.to(y_ptr.dtype.element_ty), y_desc, rows[:, None] * cfg.N + cols[None, :], mask)


def matmul_cdna4(A, W, W_scale, preshuffle=True):
    M, K = A.shape
    N = W.shape[0]
    Y = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256), )
    matmul_cdna4_kernel[grid](A, W, W_scale, Y, M, N, K, preshuffle)
    return Y


@pytest.mark.parametrize("M, N, K", [(256, 256, 512), (500, 300, 512), (1024, 512, 1024)])
@pytest.mark.parametrize("preshuffle", [False, True])
@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_matmul_cdna4(M, N, K, preshuffle):
    A = random_fp8(M, K)
    W, W_scale, W_ref = random_mxfp4(N, K)
    if preshuffle:
        W_scale = preshuffle_scales_cdna4(W_scale)
    Y = matmul_cdna4(A, W, W_scale, preshuffle)
    torch.testing.assert_close(A.float() @ W_ref.T, Y.float(), atol=1e-1, rtol=1e-2)


if __name__ == "__main__" and is_cdna4():
    print("Benchmarking matmul")
    print("===================")
    M, N, K = 8192, 8192, 8192
    A = random_fp8(M, K)
    W, W_scale, _ = random_mxfp4(N, K)
    A_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    def bench(label, fn):
        ms = triton.testing.do_bench(fn, warmup=200, rep=1000)
        print(f"{label}: {2 * M * N * K / ms * 1e-9:.0f} TFLOPS")

    bench("bf16", lambda: t4.matmul_cdna4(A_bf16, W_bf16, t4.matmul_cdna4_subtiled_kernel))
    bench("fp8 x mxfp4", lambda: matmul_cdna4(A, W, W_scale, preshuffle=False))
    W_scale = preshuffle_scales_cdna4(W_scale)
    bench("fp8 x mxfp4, preshuffled scales", lambda: matmul_cdna4(A, W, W_scale))

# %%
# ```
# bf16: 1068 TFLOPS
# fp8 x mxfp4: 1835 TFLOPS
# fp8 x mxfp4, preshuffled scales: 1900 TFLOPS
# ```

# %%
# CDNA5 Scaled WMMA
# -----------------
#
# On CDNA5, the scaled WMMA instruction is
# `v_wmma_scale_f32_16x16x128_f8f6f4`. It has the same 16x16x128 shape as the
# scaled MFMA instruction, and its Gluon operation `wmma_scaled` takes the same
# arguments as `mfma_scaled`.
#
# Most of the layouts carry over from MFMA. The accumulator uses `AMDWMMALayout`
# with `instr_shape=[16, 16, 128]`, and both operands use `k_width=16`. The
# difference is a packed fp4 operand. In one WMMA, its tile covers only 64 bytes
# along K, so unlike MFMA, it needs its own parent layout, with
# `instr_shape=[16, 16, 64]` and the same warp bases. The fp4 values are packed
# in the same way as on CDNA4. The scales get their layouts from
# `get_wmma_scale_layout`, the CDNA5 counterpart of `get_mfma_scale_layout`.


@gluon.jit
def scaled_wmma_kernel(a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                       A_FORMAT: gl.constexpr, B_FORMAT: gl.constexpr):
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(  #
        version=3, transposed=True, warp_bases=[], reg_bases=[], instr_shape=[16, 16, 128])
    wmma_packed_layout: gl.constexpr = gl.amd.AMDWMMALayout(  #
        version=3, transposed=True, warp_bases=[], reg_bases=[], instr_shape=[16, 16, 64])
    if A_FORMAT == "e2m1":
        A_K: gl.constexpr = K // 2
        a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma_packed_layout, k_width=16)
    else:
        A_K: gl.constexpr = K
        a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=16)
    if B_FORMAT == "e2m1":
        B_K: gl.constexpr = K // 2
        b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma_packed_layout, k_width=16)
    else:
        B_K: gl.constexpr = K
        b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=16)

    a = gl.load(a_ptr + offs_2d(M, A_K, A_K, a_layout))
    bk = gl.arange(0, B_K, gl.SliceLayout(1, b_layout))
    bn = gl.arange(0, N, gl.SliceLayout(0, b_layout))
    b = gl.load(b_ptr + bn[None, :] * B_K + bk[:, None])
    a_scale = None
    if a_scale_ptr is not None:
        a_scale_layout: gl.constexpr = cdna5.get_wmma_scale_layout(a_layout, [M, K // 32])
        a_scale = gl.load(a_scale_ptr + offs_2d(M, K // 32, K // 32, a_scale_layout))
    b_scale = None
    if b_scale_ptr is not None:
        b_scale_layout: gl.constexpr = cdna5.get_wmma_scale_layout(b_layout, [N, K // 32])
        b_scale = gl.load(b_scale_ptr + offs_2d(N, K // 32, K // 32, b_scale_layout))

    acc = gl.zeros([M, N], gl.float32, wmma_layout)
    acc = cdna5.wmma_scaled(a, a_scale, A_FORMAT, b, b_scale, B_FORMAT, acc)
    gl.store(c_ptr + offs_2d(M, N, N, wmma_layout), acc)


@pytest.mark.parametrize("M, N, K", [(32, 32, 128), (64, 128, 256)])
@pytest.mark.parametrize("a_kind", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("b_kind", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_scaled_wmma(M, N, K, a_kind, b_kind):
    A, A_scale, A_ref = random_operand(M, K, a_kind)
    B, B_scale, B_ref = random_operand(N, K, b_kind)
    a_format = "e2m1" if a_kind == "mxfp4" else "e4m3"
    b_format = "e2m1" if b_kind == "mxfp4" else "e4m3"
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    scaled_wmma_kernel[(1, )](A, A_scale, B, B_scale, C, M, N, K, a_format, b_format, num_warps=1)
    torch.testing.assert_close(A_ref @ B_ref.T, C, atol=1e-2, rtol=1e-2)


# %%
# The idea of scale preshuffle also applies to WMMA. Only the scale layout of
# WMMA differs, so the preshuffle reorders groups of 128 rows instead of 32, and
# again puts the scales of each lane next to each other.


def preshuffle_scales_cdna5(scales):
    N, K_SCALE = scales.shape
    scales = torch.nn.functional.pad(scales, (0, 0, 0, -N % 128))
    scales = scales.view(-1, 4, 32, K_SCALE // 4, 4).permute(0, 3, 2, 1, 4)
    return scales.reshape(-1, K_SCALE * 128)


@gluon.jit
def unshuffle_scales_cdna5(s_buf):
    ROWS: gl.constexpr = s_buf.shape[0] * 128
    K_SCALE: gl.constexpr = s_buf.shape[1] // 128
    s_buf = s_buf.reshape([ROWS // 128, K_SCALE // 4, 32, 4, 4]).permute([0, 3, 2, 1, 4])
    return s_buf.reshape([ROWS, K_SCALE])


@gluon.jit
def load_scales_cdna5(scale_ptr, ROWS: gl.constexpr, K_SCALE: gl.constexpr, layout: gl.constexpr,
                      PRESHUFFLE: gl.constexpr):
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    if PRESHUFFLE:
        gmem_layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
        s = gl.load(scale_ptr + offs_2d(ROWS // 128, K_SCALE * 128, K_SCALE * 128, gmem_layout))
        s_smem = gl.allocate_shared_memory(gl.uint8, [ROWS // 128, K_SCALE * 128], smem_layout, s)
        s_view = unshuffle_scales_cdna5(s_smem)
    else:
        gmem_layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 2], [4, 1], [1, 0])
        s = gl.load(scale_ptr + offs_2d(ROWS, K_SCALE, K_SCALE, gmem_layout))
        s_view = gl.allocate_shared_memory(gl.uint8, [ROWS, K_SCALE], smem_layout, s)
    return s_view.load(layout)


@gluon.constexpr_function
def get_partitioned_wmma_layout(M, N, K, packed):
    a_smem_layout = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [M, K], [1, 0])
    b_smem_layout = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [N, K // 2], [1, 0])
    _, _, acc_layout = cdna5.make_partitioned_dot_layouts(  #
        M, N, a_smem_layout, b_smem_layout, 4, [16, 16, 128], b_transposed=True)
    if not packed:
        return acc_layout
    return gl.amd.AMDWMMALayout(  #
        version=3, transposed=True, warp_bases=acc_layout.warp_bases, reg_bases=acc_layout.reg_bases,
        instr_shape=[16, 16, 64])


@gluon.jit
def scaled_wmma_smem_kernel(a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr,
                            K: gl.constexpr, A_FORMAT: gl.constexpr, B_FORMAT: gl.constexpr, PRESHUFFLE: gl.constexpr):
    acc_layout: gl.constexpr = get_partitioned_wmma_layout(M, N, K, False)
    a_parent: gl.constexpr = get_partitioned_wmma_layout(M, N, K, A_FORMAT == "e2m1")
    b_parent: gl.constexpr = get_partitioned_wmma_layout(M, N, K, B_FORMAT == "e2m1")
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=a_parent, k_width=16)
    b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=b_parent, k_width=16)
    A_K: gl.constexpr = K // 2 if A_FORMAT == "e2m1" else K
    B_K: gl.constexpr = K // 2 if B_FORMAT == "e2m1" else K

    a = gl.load(a_ptr + offs_2d(M, A_K, A_K, a_layout))
    bk = gl.arange(0, B_K, gl.SliceLayout(1, b_layout))
    bn = gl.arange(0, N, gl.SliceLayout(0, b_layout))
    b = gl.load(b_ptr + bn[None, :] * B_K + bk[:, None])
    a_scale = None
    if a_scale_ptr is not None:
        a_scale_layout: gl.constexpr = cdna5.get_wmma_scale_layout(a_layout, [M, K // 32])
        a_scale = load_scales_cdna5(a_scale_ptr, M, K // 32, a_scale_layout, PRESHUFFLE)
    b_scale = None
    if b_scale_ptr is not None:
        b_scale_layout: gl.constexpr = cdna5.get_wmma_scale_layout(b_layout, [N, K // 32])
        b_scale = load_scales_cdna5(b_scale_ptr, N, K // 32, b_scale_layout, PRESHUFFLE)

    acc = gl.zeros([M, N], gl.float32, acc_layout)
    acc = cdna5.wmma_scaled(a, a_scale, A_FORMAT, b, b_scale, B_FORMAT, acc)
    gl.store(c_ptr + offs_2d(M, N, N, acc_layout), acc)


@pytest.mark.parametrize("M, N, K", [(128, 256, 256), (256, 256, 512)])
@pytest.mark.parametrize("a_kind", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("b_kind", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("preshuffle", [False, True])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_scaled_wmma_smem(M, N, K, a_kind, b_kind, preshuffle):
    A, A_scale, A_ref = random_operand(M, K, a_kind)
    B, B_scale, B_ref = random_operand(N, K, b_kind)
    if preshuffle:
        A_scale = None if A_scale is None else preshuffle_scales_cdna5(A_scale)
        B_scale = None if B_scale is None else preshuffle_scales_cdna5(B_scale)
    a_format = "e2m1" if a_kind == "mxfp4" else "e4m3"
    b_format = "e2m1" if b_kind == "mxfp4" else "e4m3"
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    scaled_wmma_smem_kernel[(1, )](A, A_scale, B, B_scale, C, M, N, K, a_format, b_format, preshuffle)
    torch.testing.assert_close(A_ref @ B_ref.T, C, atol=1e-2, rtol=1e-2)


# %%
# Similarly, here is an end-to-end matmul example. It uses the partitioned
# layouts of the best WMMA kernel of `04-mfma-wmma.py`, and TDM also copies the
# preshuffled scales along with the tiles.


@gluon.aggregate
class WmmaLayouts:
    # Operand A's shared memory and register layouts
    a_smem_layout: gl.constexpr
    a_reg_layout: gl.constexpr
    # Operand W's shared memory and register layouts
    w_smem_layout: gl.constexpr
    w_reg_layout: gl.constexpr
    # The scales' shared memory and register layouts
    ws_smem_layout: gl.constexpr
    ws_reg_layout: gl.constexpr
    # Accumulator's register layout
    acc_layout: gl.constexpr


@gluon.constexpr_function
def get_mx_wmma_layouts(BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    # Operand A has shape [BLOCK_M, BLOCK_K] in fp8
    # Operand W has shape [BLOCK_N, BLOCK_K // 2] in packed fp4
    a_smem_layout = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [BLOCK_M, BLOCK_K], [1, 0])
    w_smem_layout = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [BLOCK_N, BLOCK_K // 2], [1, 0])
    a_smem_layout, w_smem_layout, acc_layout = cdna5.make_partitioned_dot_layouts(  #
        BLOCK_M, BLOCK_N, a_smem_layout, w_smem_layout, num_warps, [16, 16, 128], b_transposed=True)
    # The packed fp4 operand W needs its own parent layout.
    w_parent_layout = gl.amd.AMDWMMALayout(  #
        version=3, transposed=True, warp_bases=acc_layout.warp_bases, reg_bases=acc_layout.reg_bases,
        instr_shape=[16, 16, 64])
    a_reg_layout = gl.DotOperandLayout(0, acc_layout, 16)
    w_reg_layout = gl.DotOperandLayout(1, w_parent_layout, 16)

    ws_smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    ws_reg_layout = cdna5.get_wmma_scale_layout(w_reg_layout, [BLOCK_N, BLOCK_K // 32])
    return WmmaLayouts(a_smem_layout, a_reg_layout,  #
                       w_smem_layout, w_reg_layout,  #
                       ws_smem_layout, ws_reg_layout,  #
                       acc_layout)


@gluon.jit
def matmul_cdna5_ld_gmem(cfg, k, pred, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem):
    i = k % cfg.NUM_BUFFERS
    a_desc = cdna5.tdm.update_tensor_descriptor(a_desc, add_offsets=[off_m, k * cfg.BLOCK_K], pred=pred,
                                                clamp_bounds=True)
    w_desc = cdna5.tdm.update_tensor_descriptor(w_desc, add_offsets=[off_n, k * cfg.BLOCK_K // 2], pred=pred,
                                                clamp_bounds=True)
    ws_desc = cdna5.tdm.update_tensor_descriptor(
        ws_desc, add_offsets=[off_n // cfg.PRESHUFFLE_FACTOR, k * cfg.BLOCK_K // 32 * cfg.PRESHUFFLE_FACTOR], pred=pred,
        clamp_bounds=True)
    cdna5.tdm.async_load(a_desc, dest=a_smem.index(i))
    cdna5.tdm.async_load(w_desc, dest=w_smem.index(i))
    cdna5.tdm.async_load(ws_desc, dest=ws_smem.index(i))


@gluon.jit
def matmul_cdna5_ld_smem(cfg, k, a_smem, w_smem, ws_smem, layouts):
    a = a_smem.index(k % cfg.NUM_BUFFERS).load(layouts.a_reg_layout)
    w = w_smem.index(k % cfg.NUM_BUFFERS).permute([1, 0]).load(layouts.w_reg_layout)
    ws_buf = ws_smem.index(k % cfg.NUM_BUFFERS)
    if cfg.PRESHUFFLE:
        ws_buf = unshuffle_scales_cdna5(ws_buf)
    w_scale = ws_buf.load(layouts.ws_reg_layout)
    return a, w, w_scale


@gluon.jit
def matmul_cdna5_kernel(a_ptr, w_ptr, w_scale_ptr, y_ptr, M, N, K: gl.constexpr,  #
                        PRESHUFFLE: gl.constexpr, NUM_BUFFERS: gl.constexpr = 3):
    cfg = MatmulConfig(M, N, K, A_FORMAT="e4m3", W_FORMAT="e2m1",  #
                       BLOCK_M=256, BLOCK_N=256, BLOCK_K=256,  #
                       NUM_BUFFERS=NUM_BUFFERS, PRESHUFFLE=PRESHUFFLE, PRESHUFFLE_FACTOR=128 if PRESHUFFLE else 1)
    layouts: gl.constexpr = get_mx_wmma_layouts(cfg.BLOCK_M, cfg.BLOCK_N, cfg.BLOCK_K, gl.num_warps())

    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(cfg.N, cfg.BLOCK_N)
    off_m = pid // num_pid_n * cfg.BLOCK_M
    off_n = pid % num_pid_n * cfg.BLOCK_N

    a_desc = cdna5.tdm.make_tensor_descriptor(a_ptr, [cfg.M, cfg.K], [cfg.K, 1], [cfg.BLOCK_M, cfg.BLOCK_K],
                                              layouts.a_smem_layout)
    w_desc = cdna5.tdm.make_tensor_descriptor(w_ptr, [cfg.N, cfg.K // 2], [cfg.K // 2, 1],
                                              [cfg.BLOCK_N, cfg.BLOCK_K // 2], layouts.w_smem_layout)
    ws_desc = cdna5.tdm.make_tensor_descriptor(
        w_scale_ptr, [gl.cdiv(cfg.N, cfg.PRESHUFFLE_FACTOR), cfg.K // 32 * cfg.PRESHUFFLE_FACTOR],
        [cfg.K // 32 * cfg.PRESHUFFLE_FACTOR, 1],
        [cfg.BLOCK_N // cfg.PRESHUFFLE_FACTOR, cfg.BLOCK_K // 32 * cfg.PRESHUFFLE_FACTOR], layouts.ws_smem_layout)
    a_smem = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [cfg.NUM_BUFFERS, cfg.BLOCK_M, cfg.BLOCK_K],
                                       layouts.a_smem_layout)
    w_smem = gl.allocate_shared_memory(w_ptr.dtype.element_ty, [cfg.NUM_BUFFERS, cfg.BLOCK_N, cfg.BLOCK_K // 2],
                                       layouts.w_smem_layout)
    ws_smem = gl.allocate_shared_memory(
        gl.uint8, [cfg.NUM_BUFFERS, cfg.BLOCK_N // cfg.PRESHUFFLE_FACTOR, cfg.BLOCK_K // 32 * cfg.PRESHUFFLE_FACTOR],
        layouts.ws_smem_layout)

    num_k = gl.cdiv(cfg.K, cfg.BLOCK_K)
    for k in gl.static_range(cfg.NUM_BUFFERS - 1):
        matmul_cdna5_ld_gmem(cfg, k, k < num_k, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem, ws_smem)

    acc = gl.zeros([cfg.BLOCK_M, cfg.BLOCK_N], gl.float32, layouts.acc_layout)
    for k in range(num_k):
        next_k = k + cfg.NUM_BUFFERS - 1
        matmul_cdna5_ld_gmem(cfg, next_k, next_k < num_k, off_m, off_n, a_desc, w_desc, ws_desc, a_smem, w_smem,
                             ws_smem)
        cdna5.tdm.async_wait((cfg.NUM_BUFFERS - 1) * 3)
        a, w, w_scale = matmul_cdna5_ld_smem(cfg, k, a_smem, w_smem, ws_smem, layouts)
        acc = cdna5.wmma_scaled(a, None, cfg.A_FORMAT, w, w_scale, cfg.W_FORMAT, acc)

    y_smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[cfg.BLOCK_N, 8]],
                                                                          [cfg.BLOCK_M, cfg.BLOCK_N], [1, 0])
    y_smem = gl.allocate_shared_memory(y_ptr.dtype.element_ty, [cfg.BLOCK_M, cfg.BLOCK_N], y_smem_layout)
    y_smem.store(acc.to(y_ptr.dtype.element_ty))
    y_desc = cdna5.tdm.make_tensor_descriptor(y_ptr, [cfg.M, cfg.N], [cfg.N, 1], [cfg.BLOCK_M, cfg.BLOCK_N],
                                              y_smem_layout)
    y_desc = cdna5.tdm.update_tensor_descriptor(y_desc, add_offsets=[off_m, off_n], clamp_bounds=True)
    cdna5.tdm.async_store(y_desc, src=y_smem)
    cdna5.tdm.async_wait(0)


def matmul_cdna5(A, W, W_scale, preshuffle=True):
    M, K = A.shape
    N = W.shape[0]
    Y = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 256) * triton.cdiv(N, 256), )
    matmul_cdna5_kernel[grid](A, W, W_scale, Y, M, N, K, preshuffle)
    return Y


@pytest.mark.parametrize("M, N, K", [(256, 256, 512), (512, 768, 1024), (300, 500, 768)])
@pytest.mark.parametrize("preshuffle", [False, True])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_matmul_cdna5(M, N, K, preshuffle):
    A = random_fp8(M, K)
    W, W_scale, W_ref = random_mxfp4(N, K)
    if preshuffle:
        W_scale = preshuffle_scales_cdna5(W_scale)
    Y = matmul_cdna5(A, W, W_scale, preshuffle)
    torch.testing.assert_close(A.float() @ W_ref.T, Y.float(), atol=1e-1, rtol=1e-2)


if __name__ == "__main__" and is_cdna5():
    print("Benchmarking matmul")
    print("===================")
    M, N, K = 8192, 8192, 8192
    A = random_fp8(M, K)
    W, W_scale, _ = random_mxfp4(N, K)
    A_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    def bench(label, fn):
        ms = triton.testing.do_bench(fn, warmup=200, rep=1000)
        print(f"{label}: {2 * M * N * K / ms * 1e-9:.0f} TFLOPS")

    bench("bf16", lambda: t4.matmul_cdna5(A_bf16, W_bf16, partitioned=True))
    bench("fp8 x mxfp4", lambda: matmul_cdna5(A, W, W_scale, preshuffle=False))
    W_scale = preshuffle_scales_cdna5(W_scale)
    bench("fp8 x mxfp4, preshuffled scales", lambda: matmul_cdna5(A, W, W_scale))

# %%
# ```
# bf16: 2268 TFLOPS
# fp8 x mxfp4: 7726 TFLOPS
# fp8 x mxfp4, preshuffled scales: 7811 TFLOPS
# ```
