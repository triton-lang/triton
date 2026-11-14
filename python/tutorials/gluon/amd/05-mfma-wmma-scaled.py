"""
Scaled MFMA and WMMA
====================

Block scaling quantizes a tensor into low-precision values and one scale
factor per block of values, as in the OCP microscaling (MX) formats. CDNA4 and
CDNA5 GPUs support MX formats in hardware, with scaled MFMA and scaled WMMA
instructions.

In this tutorial, we will demonstrate scaled matrix multiplications using an
MLP-style kernel, where we multiply fp8 (e4m3) activations by mxfp4 weights.
The weights use fp4 (e2m1) values with one e8m0 scale per 32 values along K.

```
C = A @ (B * B_scale).T

A.shape = (M, K)            # fp8
B.shape = (N, K // 2)       # fp4, 2 values packed per byte
B_scale.shape = (N, K // 32)
```
"""

import importlib
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4
from triton.experimental.gluon.language.amd.cdna4 import async_copy, buffer
from triton.experimental.gluon.language.amd.cdna5 import tdm, wmma_scaled
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

t1 = importlib.import_module("01-buffer-ops")
t4 = importlib.import_module("04-mfma-wmma")


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
    # The utilities in `triton.tools.mxfp` create random MX values and scales.
    # The last value is the dequantized tensor, for the reference result.
    values = MXFP4Tensor(size=(N, K), device="cuda").random()
    scales = MXScaleTensor(size=(N, K // 32), device="cuda").random(low=1 / 128, high=2.0)
    ref = values.to(torch.float32) * scales.to(torch.float32).repeat_interleave(32, dim=1)
    return values.to_packed_tensor(dim=1), scales.data, ref


def random_operand(rows, K, kind):
    # Create an fp8, mxfp8, or mxfp4 operand of shape [rows, K]. Return the
    # data, the scales or None, and the dequantized tensor for the reference.
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
# CDNA4 MFMA
# ----------
#
# The scaled MFMA instruction we will use is
# `v_mfma_scale_f32_16x16x128_f8f6f4`. It multiplies a 16x128 tile of A by a
# 128x16 tile of B, applies the scales, and accumulates a 16x16 fp32 tile. Gluon
# exposes it as `mfma_scaled`, which takes each operand with its scales. The
# scaled MFMA takes fp8, fp6, and fp4 operands in any combination. An fp8
# operand can come with or without scales, and we pass `None` for an operand
# without scales.
#
# The accumulator uses `AMDMFMALayout` with `instr_shape=[16, 16, 128]`. Both
# operands use `k_width=16`, so each lane holds 16 consecutive bytes along K. A
# packed fp4 operand uses the same parent layout as an fp8 operand.
#
# The scales also need their own layout. But there is no dedicated layout class
# for them, so the kernel uses the generic `DistributedLinearLayout` to describe
# them. Writing them manually can be tedious, so we use `get_mfma_scale_layout`
# to compute them from the operand layout.

# %%
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
    # A operand [32, 256] has the scale [32, 8]
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
    # Copy the scales to shared memory, 4 bytes per lane, and read them with
    # `layout`.
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
class MXMfmaLayouts:
    # Operand A's global memory, shared memory, and register layouts
    a_gmem_layout: gl.constexpr
    a_smem_layout: gl.constexpr
    a_reg_layout: gl.constexpr
    # Operand W's global memory, shared memory, and register layouts
    w_gmem_layout: gl.constexpr
    w_smem_layout: gl.constexpr
    w_reg_layout: gl.constexpr
    # The scales' global memory, shared memory, and register layouts
    s_gmem_layout: gl.constexpr
    s_smem_layout: gl.constexpr
    s_reg_layout: gl.constexpr
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

    # Both operands use the padded layout and the copy layout of
    # `04-mfma-wmma.py`.
    a_gmem_layout = t4.get_gmem_layout_for_async_copies([128, 256], gl.float8e4nv)
    a_smem_layout = t4.get_smem_layout_for_async_copies([128, 256], gl.float8e4nv)
    w_gmem_layout = t4.get_gmem_layout_for_async_copies([256, 128], gl.uint8)
    w_smem_layout = t4.get_smem_layout_for_async_copies([256, 128], gl.uint8)

    # The scales are [256, 8] per K block, or [8, 256] if preshuffled. Each lane
    # copies 4 bytes.
    if preshuffle:
        s_gmem_layout = gl.BlockedLayout([1, 4], [1, warp_size], [4, 1], [1, 0])
    else:
        s_gmem_layout = gl.BlockedLayout([1, 4], [warp_size // 2, 2], [4, 1], [1, 0])
    s_smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    s_reg_layout = cdna4.get_mfma_scale_layout(w_reg_layout, [256, 8])
    return MXMfmaLayouts(a_gmem_layout, a_smem_layout, a_reg_layout,  #
                         w_gmem_layout, w_smem_layout, w_reg_layout,  #
                         s_gmem_layout, s_smem_layout, s_reg_layout,  #
                         acc_layout)


@gluon.jit
def issue_mx_copy(k, a_desc, w_desc, s_desc, a_offs, w_offs, s_offs, a_smem, w_smem, s_smem, BLOCK_K: gl.constexpr,
                  NUM_BUFFERS: gl.constexpr):
    buffer.load_to_shared(a_smem.index(k % NUM_BUFFERS), a_desc, a_offs, soffset=k * BLOCK_K)
    buffer.load_to_shared(w_smem.index(k % NUM_BUFFERS), w_desc, w_offs, soffset=k * BLOCK_K // 2)
    buffer.load_to_shared(s_smem.index(k % NUM_BUFFERS), s_desc, s_offs, soffset=k * s_smem.shape[2])
    async_copy.commit_group()


@gluon.jit
def load_mx_operands(k, a_smem, w_smem, s_smem, layouts, NUM_BUFFERS: gl.constexpr, PRESHUFFLE: gl.constexpr):
    a = a_smem.index(k % NUM_BUFFERS).load(layouts.a_reg_layout)
    w = w_smem.index(k % NUM_BUFFERS).permute([1, 0]).load(layouts.w_reg_layout)
    s_buf = s_smem.index(k % NUM_BUFFERS)
    if PRESHUFFLE:
        s_buf = unshuffle_scales_cdna4(s_buf)
    w_scale = s_buf.load(layouts.s_reg_layout)
    return a, w, w_scale


@gluon.jit
def mx_mfma_matmul_kernel(a_ptr, w_ptr, w_scale_ptr, y_ptr, M, N, K,  #
                          stride_am, stride_wn, stride_scale, stride_ym,  #
                          PRESHUFFLE: gl.constexpr, NUM_BUFFERS: gl.constexpr = 2):
    BLOCK_M: gl.constexpr = 128
    BLOCK_N: gl.constexpr = 256
    BLOCK_K: gl.constexpr = 256
    # A row of the preshuffled scales holds a group of 32 rows.
    GROUP: gl.constexpr = 32 if PRESHUFFLE else 1
    layouts = get_mx_mfma_layouts(PRESHUFFLE)

    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    off_m = pid // num_pid_n * BLOCK_M
    off_n = pid % num_pid_n * BLOCK_N

    a_desc = t1.make_buffer_descriptor(a_ptr, off_m, M, stride_am, BLOCK_M)
    w_desc = t1.make_buffer_descriptor(w_ptr, off_n, N, stride_wn, BLOCK_N)
    s_desc = t1.make_buffer_descriptor(w_scale_ptr, off_n // GROUP, gl.cdiv(N, GROUP), stride_scale, BLOCK_N // GROUP)
    a_offs = offs_2d(BLOCK_M, BLOCK_K, stride_am, layouts.a_gmem_layout)
    w_offs = offs_2d(BLOCK_N, BLOCK_K // 2, stride_wn, layouts.w_gmem_layout)
    s_offs = offs_2d(BLOCK_N // GROUP, BLOCK_K // 32 * GROUP, stride_scale, layouts.s_gmem_layout)
    a_smem = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], layouts.a_smem_layout)
    w_smem = gl.allocate_shared_memory(w_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_N, BLOCK_K // 2],
                                       layouts.w_smem_layout)
    s_smem = gl.allocate_shared_memory(w_scale_ptr.dtype.element_ty,
                                       [NUM_BUFFERS, BLOCK_N // GROUP, BLOCK_K // 32 * GROUP], layouts.s_smem_layout)

    num_k = gl.cdiv(K, BLOCK_K)
    for i in gl.static_range(NUM_BUFFERS - 1):
        issue_mx_copy(i, a_desc, w_desc, s_desc, a_offs, w_offs, s_offs, a_smem, w_smem, s_smem, BLOCK_K, NUM_BUFFERS)

    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layouts.acc_layout)
    for k in range(num_k - (NUM_BUFFERS - 1)):
        issue_mx_copy(k + NUM_BUFFERS - 1, a_desc, w_desc, s_desc, a_offs, w_offs, s_offs, a_smem, w_smem, s_smem,
                      BLOCK_K, NUM_BUFFERS)
        async_copy.wait_group(NUM_BUFFERS - 1)
        a, w, w_scale = load_mx_operands(k, a_smem, w_smem, s_smem, layouts, NUM_BUFFERS, PRESHUFFLE)
        acc = cdna4.mfma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)

    for i in gl.static_range(NUM_BUFFERS - 1):
        async_copy.wait_group(NUM_BUFFERS - 2 - i)
        a, w, w_scale = load_mx_operands(num_k - (NUM_BUFFERS - 1) + i, a_smem, w_smem, s_smem, layouts, NUM_BUFFERS,
                                         PRESHUFFLE)
        acc = cdna4.mfma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)

    y_desc = t1.make_buffer_descriptor(y_ptr + off_n, off_m, M, stride_ym, BLOCK_M)
    rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, layouts.acc_layout))
    cols = gl.arange(0, BLOCK_N, gl.SliceLayout(0, layouts.acc_layout))
    mask = (off_n + cols < N)[None, :]
    buffer.store(acc.to(y_ptr.dtype.element_ty), y_desc, rows[:, None] * stride_ym + cols[None, :], mask)


def mx_mfma_matmul(A, W, W_scale, preshuffle=True):
    M, K = A.shape
    N = W.shape[0]
    Y = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256), )
    mx_mfma_matmul_kernel[grid](A, W, W_scale, Y, M, N, K, A.stride(0), W.stride(0), W_scale.stride(0), Y.stride(0),
                                preshuffle)
    return Y


@pytest.mark.parametrize("M, N, K", [(256, 256, 512), (500, 300, 512), (1024, 512, 1024)])
@pytest.mark.parametrize("preshuffle", [False, True])
@pytest.mark.skipif(not is_cdna4(), reason="Requires CDNA4")
def test_mx_mfma_matmul(M, N, K, preshuffle):
    A = random_fp8(M, K)
    W, W_scale, W_ref = random_mxfp4(N, K)
    if preshuffle:
        W_scale = preshuffle_scales_cdna4(W_scale)
    Y = mx_mfma_matmul(A, W, W_scale, preshuffle)
    torch.testing.assert_close(A.float() @ W_ref.T, Y.float(), atol=1e-1, rtol=1e-2)


# %%
# We compare with the fastest bf16 MFMA kernel from `04-mfma-wmma.py` on an
# MI355X GPU with M = N = K = 8192, with and without preshuffled scales.

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

    bench("bf16", lambda: t4.mfma_matmul(A_bf16, W_bf16, t4.mfma_matmul_subtiled_kernel))
    bench("fp8 x mxfp4", lambda: mx_mfma_matmul(A, W, W_scale, preshuffle=False))
    W_scale = preshuffle_scales_cdna4(W_scale)
    bench("fp8 x mxfp4, preshuffled scales", lambda: mx_mfma_matmul(A, W, W_scale))

# %%
# ```
# bf16: 1202 TFLOPS
# fp8 x mxfp4: 1992 TFLOPS
# fp8 x mxfp4, preshuffled scales: 2132 TFLOPS
# ```

# %%
# CDNA5 WMMA
# ----------
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
        a_scale_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(a_layout, [M, K // 32])
        a_scale = gl.load(a_scale_ptr + offs_2d(M, K // 32, K // 32, a_scale_layout))
    b_scale = None
    if b_scale_ptr is not None:
        b_scale_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(b_layout, [N, K // 32])
        b_scale = gl.load(b_scale_ptr + offs_2d(N, K // 32, K // 32, b_scale_layout))

    acc = gl.zeros([M, N], gl.float32, wmma_layout)
    acc = wmma_scaled(a, a_scale, A_FORMAT, b, b_scale, B_FORMAT, acc)
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
    # Copy the scales to shared memory, 4 bytes per lane, and read them with
    # `layout`.
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
    # The WMMA layout of `make_partitioned_dot_layouts` for an [M, N] tile. A
    # packed fp4 operand uses the same warps with `instr_shape=[16, 16, 64]`.
    a_smem_layout = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [M, K], [1, 0])
    b_smem_layout = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [N, K // 2], [1, 0])
    _, _, acc_layout = gl.amd.cdna5.make_partitioned_dot_layouts(  #
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
        a_scale_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(a_layout, [M, K // 32])
        a_scale = load_scales_cdna5(a_scale_ptr, M, K // 32, a_scale_layout, PRESHUFFLE)
    b_scale = None
    if b_scale_ptr is not None:
        b_scale_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(b_layout, [N, K // 32])
        b_scale = load_scales_cdna5(b_scale_ptr, N, K // 32, b_scale_layout, PRESHUFFLE)

    acc = gl.zeros([M, N], gl.float32, acc_layout)
    acc = wmma_scaled(a, a_scale, A_FORMAT, b, b_scale, B_FORMAT, acc)
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
# Similarly, here is an end-to-end matmul example. It follows the best WMMA
# kernel of `04-mfma-wmma.py`, and TDM also copies the preshuffled scales along
# with the tiles.


@gluon.aggregate
class MXWmmaLayouts:
    # Operand A's shared memory and register layouts
    a_smem_layout: gl.constexpr
    a_reg_layout: gl.constexpr
    # Operand W's shared memory and register layouts
    w_smem_layout: gl.constexpr
    w_reg_layout: gl.constexpr
    # The scales' shared memory and register layouts
    s_smem_layout: gl.constexpr
    s_reg_layout: gl.constexpr
    # Accumulator's register layout
    acc_layout: gl.constexpr


@gluon.constexpr_function
def get_mx_wmma_layouts(BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    # Operand A has shape [BLOCK_M, BLOCK_K] in fp8
    # Operand W has shape [BLOCK_N, BLOCK_K // 2] in packed fp4, so transposed
    a_smem_layout = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [BLOCK_M, BLOCK_K], [1, 0])
    w_smem_layout = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [BLOCK_N, BLOCK_K // 2], [1, 0])
    a_smem_layout, w_smem_layout, acc_layout = gl.amd.cdna5.make_partitioned_dot_layouts(  #
        BLOCK_M, BLOCK_N, a_smem_layout, w_smem_layout, num_warps, [16, 16, 128], b_transposed=True)
    # The packed fp4 operand W needs its own parent layout.
    b_parent_layout = gl.amd.AMDWMMALayout(  #
        version=3, transposed=True, warp_bases=acc_layout.warp_bases, reg_bases=acc_layout.reg_bases,
        instr_shape=[16, 16, 64])
    a_reg_layout = gl.DotOperandLayout(0, acc_layout, 16)
    w_reg_layout = gl.DotOperandLayout(1, b_parent_layout, 16)

    s_smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    s_reg_layout = gl.amd.cdna5.get_wmma_scale_layout(w_reg_layout, [BLOCK_N, BLOCK_K // 32])
    return MXWmmaLayouts(a_smem_layout, a_reg_layout,  #
                         w_smem_layout, w_reg_layout,  #
                         s_smem_layout, s_reg_layout,  #
                         acc_layout)


@gluon.jit
def issue_loads(k, pred, off_m, off_n, a_desc, w_desc, s_desc, a_smem, w_smem, s_smem, BLOCK_K: gl.constexpr,
                NUM_BUFFERS: gl.constexpr, GROUP: gl.constexpr):
    i = k % NUM_BUFFERS
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[off_m, k * BLOCK_K], pred=pred, clamp_bounds=True)
    w_desc = tdm.update_tensor_descriptor(w_desc, add_offsets=[off_n, k * BLOCK_K // 2], pred=pred, clamp_bounds=True)
    s_desc = tdm.update_tensor_descriptor(s_desc, add_offsets=[off_n // GROUP, k * BLOCK_K // 32 * GROUP], pred=pred,
                                          clamp_bounds=True)
    tdm.async_load(a_desc, dest=a_smem.index(i))
    tdm.async_load(w_desc, dest=w_smem.index(i))
    tdm.async_load(s_desc, dest=s_smem.index(i))


@gluon.jit
def mx_matmul_kernel(a_ptr, w_ptr, w_scale_ptr, y_ptr, M, N, K,  #
                     stride_am, stride_wn, stride_scale, stride_ym,  #
                     BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, NUM_BUFFERS: gl.constexpr,
                     PRESHUFFLE: gl.constexpr):
    # A row of the preshuffled scales holds a group of 128 rows.
    GROUP: gl.constexpr = 128 if PRESHUFFLE else 1
    layouts: gl.constexpr = get_mx_wmma_layouts(BLOCK_M, BLOCK_N, BLOCK_K, gl.num_warps())

    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    off_m = pid // num_pid_n * BLOCK_M
    off_n = pid % num_pid_n * BLOCK_N

    a_desc = tdm.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BLOCK_M, BLOCK_K], layouts.a_smem_layout)
    w_desc = tdm.make_tensor_descriptor(w_ptr, [N, K // 2], [stride_wn, 1], [BLOCK_N, BLOCK_K // 2],
                                        layouts.w_smem_layout)
    s_desc = tdm.make_tensor_descriptor(w_scale_ptr, [gl.cdiv(N, GROUP), K // 32 * GROUP], [stride_scale, 1],
                                        [BLOCK_N // GROUP, BLOCK_K // 32 * GROUP], layouts.s_smem_layout)
    a_smem = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], layouts.a_smem_layout)
    w_smem = gl.allocate_shared_memory(w_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_N, BLOCK_K // 2],
                                       layouts.w_smem_layout)
    s_smem = gl.allocate_shared_memory(gl.uint8, [NUM_BUFFERS, BLOCK_N // GROUP, BLOCK_K // 32 * GROUP],
                                       layouts.s_smem_layout)

    num_k = gl.cdiv(K, BLOCK_K)
    for k in gl.static_range(NUM_BUFFERS - 1):
        issue_loads(k, True, off_m, off_n, a_desc, w_desc, s_desc, a_smem, w_smem, s_smem, BLOCK_K, NUM_BUFFERS, GROUP)

    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layouts.acc_layout)
    for k in range(num_k):
        # Load the tiles of iteration k + NUM_BUFFERS - 1. The last iterations
        # disable the loads with a predicate.
        next_k = k + NUM_BUFFERS - 1
        issue_loads(next_k, next_k < num_k, off_m, off_n, a_desc, w_desc, s_desc, a_smem, w_smem, s_smem, BLOCK_K,
                    NUM_BUFFERS, GROUP)
        tdm.async_wait((NUM_BUFFERS - 1) * 3)
        a = a_smem.index(k % NUM_BUFFERS).load(layouts.a_reg_layout)
        w = w_smem.index(k % NUM_BUFFERS).permute([1, 0]).load(layouts.w_reg_layout)
        s_buf = s_smem.index(k % NUM_BUFFERS)
        if PRESHUFFLE:
            s_buf = unshuffle_scales_cdna5(s_buf)
        w_scale = s_buf.load(layouts.s_reg_layout)
        acc = wmma_scaled(a, None, "e4m3", w, w_scale, "e2m1", acc)

    y_smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0])
    y_smem = gl.allocate_shared_memory(y_ptr.dtype.element_ty, [BLOCK_M, BLOCK_N], y_smem_layout)
    y_smem.store(acc.to(y_ptr.dtype.element_ty))
    y_desc = tdm.make_tensor_descriptor(y_ptr, [M, N], [stride_ym, 1], [BLOCK_M, BLOCK_N], y_smem_layout)
    y_desc = tdm.update_tensor_descriptor(y_desc, add_offsets=[off_m, off_n], clamp_bounds=True)
    tdm.async_store(y_desc, src=y_smem)
    tdm.async_wait(0)


def mx_matmul(A, W, W_scale, preshuffle=True, BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, NUM_BUFFERS=3):
    M, K = A.shape
    N = W.shape[0]
    Y = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    mx_matmul_kernel[grid](A, W, W_scale, Y, M, N, K, A.stride(0), W.stride(0), W_scale.stride(0), Y.stride(0), BLOCK_M,
                           BLOCK_N, BLOCK_K, NUM_BUFFERS, preshuffle)
    return Y


@pytest.mark.parametrize("M, N, K", [(256, 256, 512), (512, 768, 1024), (300, 500, 768)])
@pytest.mark.parametrize("preshuffle", [False, True])
@pytest.mark.skipif(not is_cdna5(), reason="Requires CDNA5")
def test_mx_matmul(M, N, K, preshuffle):
    A = random_fp8(M, K)
    W, W_scale, W_ref = random_mxfp4(N, K)
    if preshuffle:
        W_scale = preshuffle_scales_cdna5(W_scale)
    Y = mx_matmul(A, W, W_scale, preshuffle)
    torch.testing.assert_close(A.float() @ W_ref.T, Y.float(), atol=1e-1, rtol=1e-2)


# %%
# On an MI455X GPU, we run the same comparison against the fastest bf16 WMMA
# kernel from `04-mfma-wmma.py`, which uses the partitioned layouts.

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

    bench("bf16", lambda: t4.wmma_matmul(A_bf16, W_bf16, partitioned=True))
    bench("fp8 x mxfp4", lambda: mx_matmul(A, W, W_scale, preshuffle=False))
    W_scale = preshuffle_scales_cdna5(W_scale)
    bench("fp8 x mxfp4, preshuffled scales", lambda: mx_matmul(A, W, W_scale))

# %%
# ```
# bf16: 2299 TFLOPS
# fp8 x mxfp4: 7750 TFLOPS
# fp8 x mxfp4, preshuffled scales: 7866 TFLOPS
# ```
