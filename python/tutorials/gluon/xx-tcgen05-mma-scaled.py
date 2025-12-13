"""
Block-Scaled MMA with tcgen05_mma_scaled
========================================

This tutorial ports the Triton block-scaled matmul example to Gluon and shows
how to stage scale factors in shared memory and move them into Tensor Memory
with ``tcgen05_copy`` before issuing ``tcgen05_mma_scaled``.

Why the extra dance for scales?
-------------------------------
- The scale factors for FP4/FP8 operands must be rearranged into a layout that
  ``tcgen05_mma_scaled`` can stream from Tensor Memory (TMEM).
- ``tcgen05_copy`` can only write to the special TMEM scale layout, so scales
  need to arrive in shared memory in the chunked layout described in
  ``UseShmemForScales`` (see ``OptimizeDotOperands.cpp``) and the
  ``TCGen5MMAScaleSharedToTmemConversion`` lowering.
- The SMEM layout that works with ``tcgen05_copy`` is 4-D:
  ``(rep_m, rep_k, 32, 16)`` where
    * ``rep_m = BLOCK_M / 128`` (or ``BLOCK_N / 128`` for RHS scales),
    * ``rep_k = BLOCK_K / VEC_SIZE / 4``,
    * the innermost 16 bytes correspond to the 4x4 scale tiles in the Triton
      tutorial. ``tcgen05_copy`` duplicates each 32x128b chunk over four warps
      into the TMEM scale layout.

This example sticks to a single CTA-sized matmul (128x256x256) so the layouts
stay clear. Scaling up to multiple CTAs just replicates the same per-block
logic.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
)
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma_scaled,
)


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires a Blackwell GPU (compute capability 10.x)")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def expand_scales(scale, vec_size):
    """Broadcast packed scales (shape: rows x K/VEC_SIZE) along K."""
    return scale.repeat_interleave(vec_size, dim=1)


def chunk_to_2d(scale_chunk, block_rows, vec_size, block_k):
    """Convert chunked (rep_m, rep_k, 32, 4, 4) into logical 2D (block_rows, block_k/vec_size)."""
    rep_k = block_k // vec_size // 4
    chunk = scale_chunk.view(block_rows // 128, rep_k, 32, 4, 4)
    return chunk.permute(0, 3, 2, 1, 4).reshape(block_rows, block_k // vec_size)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@gluon.jit
def block_scaled_matmul_kernel(  #
    a_desc,
    b_desc,
    a_scale_desc,
    b_scale_desc,
    c_desc,
    K: gl.constexpr,
    VEC_SIZE: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_warps: gl.constexpr,
    num_buffers: gl.constexpr,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    gl.static_assert(K == BLOCK_K, "tutorial assumes a single K block for scales")
    scale_cols: gl.constexpr = BLOCK_K // VEC_SIZE
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Double-buffer operands and scales in SMEM
    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    a_scale_2d_layout: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=0, transposed=False, element_bitwidth=8,
                                                           rank=2)
    b_scale_2d_layout: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=0, transposed=False, element_bitwidth=8,
                                                           rank=2)
    a_scale_2d_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [num_buffers, BLOCK_M, scale_cols],
                                                a_scale_2d_layout)
    b_scale_2d_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [num_buffers, BLOCK_N, scale_cols],
                                                b_scale_2d_layout)

    load_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(load_bar, count=1)
    mbarrier.init(mma_bar, count=1)

    # Accumulator in TMEM
    acc_tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), acc_tmem_layout, num_warps)
    acc_init = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=acc_reg_layout)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], acc_tmem_layout, acc_init)

    # TMEM layouts for scales (2D)
    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale_desc.dtype, [BLOCK_M, scale_cols], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale_desc.dtype, [BLOCK_N, scale_cols], scale_layout)
    bytes_per_stage: gl.constexpr = a_desc.block_type.nbytes + b_desc.block_type.nbytes + \
        a_scale_desc.block_type.nbytes + b_scale_desc.block_type.nbytes

    k_block = 0
    phase = 0
    use_acc = False

    # Preload the first set of tiles (A/B + scales)
    mbarrier.expect(load_bar, bytes_per_stage)
    tma.async_copy_global_to_shared(a_desc, [off_m, k_block], load_bar, a_bufs.index(phase))
    tma.async_copy_global_to_shared(b_desc, [k_block, off_n], load_bar, b_bufs.index(phase))
    tma.async_copy_global_to_shared(a_scale_desc, [off_m, 0], load_bar, a_scale_2d_bufs.index(phase))
    tma.async_copy_global_to_shared(b_scale_desc, [off_n, 0], load_bar, b_scale_2d_bufs.index(phase))

    while k_block < K:
        # Consume current stage
        mbarrier.wait(load_bar, phase)
        fence_async_shared()
        tcgen05_copy(a_scale_2d_bufs.index(phase), a_scale_tmem)
        tcgen05_copy(b_scale_2d_bufs.index(phase), b_scale_tmem)
        tcgen05_mma_scaled(a_bufs.index(phase), b_bufs.index(phase), acc_tmem, a_scale_tmem, b_scale_tmem, "e4m3",
                           "e4m3", use_acc=use_acc)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase)
        use_acc = True
        k_block += BLOCK_K

    mbarrier.invalidate(load_bar)
    mbarrier.invalidate(mma_bar)

    # Store accumulator back to GMEM
    acc = acc_tmem.load(acc_reg_layout)
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(c_desc.dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


# ---------------------------------------------------------------------------
# Host driver
# ---------------------------------------------------------------------------


def make_block_scaled_inputs(M=128, N=256, K=256, VEC_SIZE=32, BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, num_buffers=2):
    """Create input tensors and TensorDescriptors for one CTA tile."""
    assert M == BLOCK_M and N == BLOCK_N, "This tutorial keeps one CTA for clarity"
    assert K % BLOCK_K == 0 and K >= BLOCK_K
    dtype_in = gl.float8e4nv

    # Operands
    # Generate in fp16 then cast to fp8 to avoid unsupported random kernels in fp8
    a = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    # Match tcgen05 MMA shared layouts: swizzle 128B, B is column-major (transposed=True)
    a_layout = gl.NVMMASharedLayout(swizzle_byte_width=128, transposed=False,
                                    element_bitwidth=dtype_in.primitive_bitwidth, rank=2)
    b_layout = gl.NVMMASharedLayout(swizzle_byte_width=128, transposed=False,
                                    element_bitwidth=dtype_in.primitive_bitwidth, rank=2)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N], b_layout)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N], c_layout)

    # Scales stored chunked (rep_m, rep_k, 32, 4, 4) to mirror UseShmemForScales pattern.
    rep_k = BLOCK_K // VEC_SIZE // 4
    scale_cols = BLOCK_K // VEC_SIZE
    a_scale_chunk = torch.randint(1, 120, (BLOCK_M // 128, rep_k, 32, 4, 4), dtype=torch.int8, device="cuda")
    b_scale_chunk = torch.randint(1, 120, (BLOCK_N // 128, rep_k, 32, 4, 4), dtype=torch.int8, device="cuda")
    a_scale_2d = chunk_to_2d(a_scale_chunk, BLOCK_M, VEC_SIZE, BLOCK_K)
    b_scale_2d = chunk_to_2d(b_scale_chunk, BLOCK_N, VEC_SIZE, BLOCK_K)
    # TMA descriptors require 16-byte aligned strides; pad the leading stride.
    scale_stride = 16
    a_scale_storage = torch.zeros((BLOCK_M, scale_stride), dtype=torch.int8, device="cuda")
    b_scale_storage = torch.zeros((BLOCK_N, scale_stride), dtype=torch.int8, device="cuda")
    a_scale = a_scale_storage.as_strided((BLOCK_M, scale_cols), (scale_stride, 1))
    b_scale = b_scale_storage.as_strided((BLOCK_N, scale_cols), (scale_stride, 1))
    a_scale.copy_(a_scale_2d)
    b_scale.copy_(b_scale_2d)
    a_scale_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, scale_cols], gl.int8)
    b_scale_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_N, scale_cols], gl.int8)
    a_scale_desc = TensorDescriptor.from_tensor(a_scale, [BLOCK_M, scale_cols], a_scale_layout)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale, [BLOCK_N, scale_cols], b_scale_layout)

    return (
        a,
        b,
        a_desc,
        b_desc,
        c,
        c_desc,
        a_scale_chunk,
        b_scale_chunk,
        a_scale_desc,
        b_scale_desc,
        BLOCK_K,
        VEC_SIZE,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_buffers,
    )


def block_scaled_matmul(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, num_buffers=2):
    (
        a,
        b,
        a_desc,
        b_desc,
        c,
        c_desc,
        a_scale_chunk,
        b_scale_chunk,
        a_scale_desc,
        b_scale_desc,
        BLOCK_K,
        VEC_SIZE,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_buffers,
    ) = make_block_scaled_inputs(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_buffers=num_buffers)

    grid = (1, 1)  # single CTA for this tutorial
    block_scaled_matmul_kernel[grid](
        a_desc, b_desc, a_scale_desc, b_scale_desc, c_desc,  #
        a.shape[1], VEC_SIZE, BLOCK_K, num_warps=4, num_buffers=num_buffers)
    return a, b, a_scale_chunk, b_scale_chunk, c


def validate_block_scaled():
    a, b, a_scale_chunk, b_scale_chunk, c = block_scaled_matmul()
    VEC_SIZE = 32

    # Reconstruct logical scale matrices
    a_scale_2d = chunk_to_2d(a_scale_chunk, a.shape[0], VEC_SIZE, a.shape[1])
    b_scale_2d = chunk_to_2d(b_scale_chunk, b.shape[0], VEC_SIZE, b.shape[1])
    a_scale_full = expand_scales(a_scale_2d, VEC_SIZE)  # (128, K)
    b_scale_full = expand_scales(b_scale_2d, VEC_SIZE).T  # (K, 256)

    ref = (a.float() * a_scale_full) @ (b.float() * b_scale_full)
    torch.testing.assert_close(ref, c, atol=1e-2, rtol=1e-2)
    print("✅ block-scaled matmul (single CTA) passed")


def demo_tcgen05_copy_layout():
    """Show how tcgen05_copy maps a scale chunk from SMEM to TMEM."""
    smem_h = 64
    smem_w = 16
    num_rows = 128
    num_cols = smem_h * smem_w // 32

    @gluon.jit
    def kernel(in_ptr, out_ptr, smem_h: gl.constexpr, smem_w: gl.constexpr, num_rows: gl.constexpr,
               num_cols: gl.constexpr):
        in_ptrs = in_ptr + gl.arange(0, smem_h)[:, None] * smem_w + gl.arange(0, smem_w)[None, :]
        out_ptrs = out_ptr + gl.arange(0, num_rows)[:, None] * num_cols + gl.arange(0, num_cols)[None, :]

        blocked: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
        value = gl.load(gl.set_auto_layout(in_ptrs, blocked))

        smem_layout: gl.constexpr = gl.SharedLinearLayout(
            offset_bases=[[0, 1], [0, 2], [32, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]])
        tmem_layout: gl.constexpr = TensorMemoryScalesLayout()
        smem = gl.allocate_shared_memory(gl.int8, (smem_h, smem_w), layout=smem_layout)
        tmem = allocate_tensor_memory(gl.int8, (smem_h, smem_w), layout=tmem_layout)

        barrier = gl.allocate_shared_memory(gl.int64, [1], gl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(barrier, count=1)

        smem.store(value)
        fence_async_shared()
        tcgen05_copy(smem, tmem)
        tcgen05_commit(barrier)
        mbarrier.wait(barrier, phase=0)
        tmem_alias: gl.constexpr = TensorMemoryLayout((num_rows, num_cols), col_stride=1)
        tmem = tmem._reinterpret(gl.int8, (num_rows, num_cols), tmem_alias)
        value = tmem.load(blocked)
        gl.store(gl.set_auto_layout(out_ptrs, blocked), value)

    x = torch.randint(size=(smem_h, smem_w), low=-50, high=50, dtype=torch.int8, device="cuda")
    y = torch.zeros(size=(num_rows, num_cols), dtype=torch.int8, device="cuda")
    kernel[(1, )](x, y, smem_h, smem_w, num_rows, num_cols)
    return x, y


if __name__ == "__main__":
    torch.manual_seed(0)
    validate_block_scaled()
    x, y = demo_tcgen05_copy_layout()
    print("tcgen05_copy demo shapes:", x.shape, "->", y.shape)
