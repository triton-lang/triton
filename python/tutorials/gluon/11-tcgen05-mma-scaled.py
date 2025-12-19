"""
Blocked-Scaled Matrix Multiplication
====================================

Block scaling is a quantization technique whereby a floating point tensor `X` is
quantized into: a tensor `Q` of the same shape, but with a lower-precision dtype;
and a scale tensor `S`. Tensor `X` is quantized into `Q` by dividing it into
equally-sized blocks, where each block is associated with a single scale factor.

When performing matrix multiplication on block-scaled tensors, we load both
quantized operands and their scales from global memory on to the SMs,
where they are dequantized by multiplying each block of quantized values by their
respective scale factors. The MMA itself is then performed in a higher precision.

We can accelerate the MMA of the dequantized operands using tensor core
instructions like `tcgen05_mma`. But NVIDIA Blackwell GPUs support hardware
acceleration for block-scaled MMAs, in the form of the `tcgen05_mma_scaled`
instructions which fuse the operand dequantization and MMA into a single
instruction.

`tcgen05_mma_scaled` supports specific block-scaled quantization schemes:
- nvfp4: NVIDIA-specific fp4 quantization scheme using VEC_SIZE=16 and
  float8_e4m3fn scales
- mxfp4/mxfp6/mxfp6: Open Compute Project (OCP) microscaling format (MX) for
  fp4/fp6/fp8, using VEC_SIZE=32 and fp8e8m0 scales

mxfp6 is not supported by Gluon because Gluon does not expose fp6 dtypes.
MX scales are e8m0, meaning 0 mantissa bits and 8 exponent bits. In other words,
they are exponents of 2 from 2**-127 to 2**127, where 255 represents NaN.

The nvfp4, mxfp4, and mxfp8 quantization schemes use a 1D block of size `VEC_SIZE`,
and quantize the original tensors along the MMA reduction dimension
(i.e. the K dimension). For example, in the block-scale MMA in the form:

```
C = (A * A_scale) @ (B * B_scale)
```

The tensors will have the following shapes:

```
A.shape = (M, K)
B.shape = (N, K)
A_scale.shape = (M, K // VEC_SIZE)
B_scale.shape = (N, K // VEC_SIZE)
```

Each scale factor is broadcasted and multiplied across a vector of `VEC_SIZE`
elements from the A and B tensors along the K dimension.

Gluon currently only supports transposed B operands for `tcgen05_mma_scaled`,
meaning it expects the B tile to have the shape `[BLOCK_N, BLOCK_K]` to be fed
into `tcgen05_mma_scaled` as a transposed shared memory descriptor.

In this tutorial, we will demonstrate how to use `tcgen05_mma_scaled` to perform
hardware-accelerated block-scaled MMAs. Then, we will introduce using `tcgen05_copy`
to efficiently copy the scales into tensor memory. We will also cover how to pick
an efficient scale layout in global memory. Finally, we will show how to write
pipelined and warp-specialized block-scaled MMAs.
"""

import itertools
import importlib
import pytest
import triton
import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from dataclasses import replace
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from triton.language.core import _aggregate as aggregate
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    fence_async_shared,
    tensor_memory_descriptor,
    tcgen05_copy,
    tcgen05_commit,
    tcgen05_mma_scaled,
    mbarrier,
    tma,
)


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")

# Re-use utilities from the previous tutorials.
t7 = importlib.import_module("07-persistence")
t8 = importlib.import_module("08-warp-specialization")

# %%
# Let's write a simple blocked-scaled matmul kernel. First, we will assume that
# the scale factors take the same layout as their corresponding blocks.
# Specifically, our A, B, A_scale, and B_scale tensors will have the following shapes:
#
# ```
# A.shape = (M, K)
# B.shape = (N, K)
# A_scale.shape = (M, K // VEC_SIZE)
# B_scale.shape = (N, K // VEC_SIZE)
# ```
#
# Note that Gluon represents fp4 dtypes by packing 2 fp4 elements into a uint8
# element. Typically, we pack the fp4 elements along the reduction dimension,
# i.e. the K dimension. For example, if A and B were fp4e2m1 tensors packed
# along K into uint8 elements, they would have the shapes:
#
# ```
# A.shape = (M, K // 2)
# B.shape = (N, K // 2)
# A_scale.shape = (M, K // VEC_SIZE)
# B_scale.shape = (N, K // VEC_SIZE)
# ```


@gluon.jit
def simple_mma_scaled_kernel(a_desc, b_desc, c_desc, a_scale_ptr, a_scale_stride_m, a_scale_stride_k, b_scale_ptr,
                             b_scale_stride_n, b_scale_stride_k, VEC_SIZE: gl.constexpr):
    # If the operand dtype is fp4, they will be packed into uint8.
    A_IS_FP4: gl.constexpr = a_desc.dtype == gl.uint8
    B_IS_FP4: gl.constexpr = b_desc.dtype == gl.uint8
    # fp4 is a sub-byte dtype, so we need to account for this when loading the
    # operands from a uint8 tensor descriptor.
    A_ELEM_PER_BYTE: gl.constexpr = 2 if A_IS_FP4 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if B_IS_FP4 else 1

    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    # BLOCK_K represents the number of actual elements along K.
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    # Allocate shared memory for the operands.
    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)

    # Allocate tensor memory for the scales. The scales must have the layout
    # `TensorMemoryScalesLayout`. Note that the B scales are always passed to
    # `tcgen05_mma_scaled` as [BLOCK_N, BLOCK_K // VEC_SIZE].
    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale_ptr.dtype.element_ty, [BLOCK_N, BLOCK_K // VEC_SIZE], scale_layout)

    # Allocate tensor memory for the accumulator.
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    use_acc = False

    # Allocate a barrier to track the operand loads and MMA.
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)
    phase = 0

    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    for k in range(0, K, BLOCK_K):
        # BLOCK_K is the number of logical elements along K to load in a tile.
        # For sub-byte dtypes like fp4, translate them into uint8 offset.
        off_k_a = k // A_ELEM_PER_BYTE
        off_k_b = k // B_ELEM_PER_BYTE

        # When issuing a TMA transaction to TMA tensor descriptors with fp4 padded operands, we need to multiply
        # the offset along the contiguous dimension by 2 to account for the padding. This applies to async TMA
        # loads, stores, gather, and scatter. Failing to do this can result in illegal instruction errors. If you
        # catch the illegal instruction error inside `cuda-gdb`, it may point to the TMA instruction or the
        # `mbarrier.wait` on the instruction completion barrier. When breaking on the illegal instruction error,
        # you can use `x/i $pc` to print the instruction at the faulting address, and for example use `x/-50i $pc`
        # to print the previous 50 instructions.
        if a_desc.layout.fp4_padded:
            off_k_a *= 2
        if b_desc.layout.fp4_padded:
            off_k_b *= 2

        # Load the A and B tiles.
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, off_k_a], bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [off_n, off_k_b], bar, b_smem)
        mbarrier.wait(bar, phase)

        # Load the scales. We must always feed `b_scales` into `tcgen05_mma_scaled`
        # as [BLOCK_N, BLOCK_K // VEC_SIZE].
        coalesced_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])

        # Compute the right offsets by dividing the offset along K by VEC_SIZE.
        a_scale_offs_m = off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, coalesced_2d_layout))
        a_scale_offs_k = k // VEC_SIZE + gl.arange(0, BLOCK_K // VEC_SIZE, layout=gl.SliceLayout(
            0, coalesced_2d_layout))
        a_scale = gl.load(a_scale_ptr + a_scale_offs_m[:, None] * a_scale_stride_m +
                          a_scale_offs_k[None, :] * a_scale_stride_k)

        b_scale_offs_n = off_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, coalesced_2d_layout))
        b_scale_offs_k = k // VEC_SIZE + gl.arange(0, BLOCK_K // VEC_SIZE, layout=gl.SliceLayout(
            0, coalesced_2d_layout))
        b_scale = gl.load(b_scale_ptr + b_scale_offs_n[:, None] * b_scale_stride_n +
                          b_scale_offs_k[None, :] * b_scale_stride_k)

        # We have to write the scales to tensor memory. Convert them into a the right
        # layout so we can write into tensor memory with layout `TensorMemoryScalesLayout`.
        a_scale_layout: gl.constexpr = get_tmem_reg_layout(a_scale.dtype, a_scale.type.shape, scale_layout,
                                                           gl.num_warps())
        b_scale_layout: gl.constexpr = get_tmem_reg_layout(b_scale.dtype, b_scale.type.shape, scale_layout,
                                                           gl.num_warps())
        a_scale = gl.convert_layout(a_scale, a_scale_layout)
        b_scale = gl.convert_layout(b_scale, b_scale_layout)
        a_scale_tmem.store(a_scale)
        b_scale_tmem.store(b_scale)

        # Pass the operand and scale tensors to `tcgen05_mma_scaled` along with the right
        # operand format strings.
        a_format: gl.constexpr = "e2m1" if A_IS_FP4 else "e4m3"
        b_format: gl.constexpr = "e2m1" if B_IS_FP4 else "e4m3"
        # Pass the operand and scale tensors to `tcgen05_mma_scaled` along with the right
        # operand format strings. Accumulate in-place with `use_acc`, which is set to False
        # on the first iteration to zero-initialize the accumulator. The B operand must be
        # transposed in shared memory.
        tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                           use_acc=use_acc)
        # Commit the MMA and wait for it to complete.
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase)
        use_acc = True
        phase ^= 1

    # Make sure to invalidate the barriers after we are done with them to avoid
    # race conditions and memory corruption errors. This is especially important
    # because a few lines below we are allocating shared memory for the async TMA
    # store of the accumulator. Re-using mbarrier shared memory without calling
    # `invalidate` is undefined behaviour.
    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)

    # Load the accumulator tile from tensor memory and convert it to the output dtype.
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, gl.num_warps())
    acc = acc_tmem.load(acc_reg_layout)
    acc = acc.to(c_desc.dtype)

    # Write the accumulator via TMA store.
    acc_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
    acc_smem.store(acc)
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], acc_smem)
    tma.store_wait(0)


def make_operand_descriptor(value: torch.Tensor, BLOCK_MN: int, BLOCK_K: int, MIXED_PREC: bool):
    # If the operand dtype is fp4, they will be packed into uint8.
    IS_FP4 = value.dtype == torch.uint8
    ELEM_PER_BYTE = 2 if IS_FP4 else 1

    # When performing a mixed-precision `tcgen05_mma_scaled`, where one operand
    # is mxfp8 and the other is mxfp4, the fp4 operand is padded in shared memory.
    IS_MIXED_PREC_FP4 = MIXED_PREC and IS_FP4
    layout = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_MN, BLOCK_K // ELEM_PER_BYTE],
        gl.uint8 if IS_FP4 else gl.float8e4nv,
        fp4_padded=IS_MIXED_PREC_FP4,
    )
    return TensorDescriptor.from_tensor(value, [BLOCK_MN, BLOCK_K // ELEM_PER_BYTE], layout)


def make_output_descriptor(M: int, N: int, dtype: torch.dtype, BLOCK_M: int, BLOCK_N: int):
    C = torch.empty(M, N, device="cuda", dtype=dtype)
    C_dtype = getattr(gl, str(dtype).split('.')[1])
    C_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], C_dtype)
    return TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], C_desc_layout)


def simple_mma_scaled(A, B, A_scale, B_scale, VEC_SIZE, out_dtype=torch.float16, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128):
    M, N = A.shape[0], B.shape[0]

    is_nvfp4 = A_scale.dtype == torch.float8_e4m3fn
    assert not is_nvfp4 or B_scale.dtype == A_scale.dtype, "tcgen05_mma_scaled does not support mixing nvfp4 with other microscaled formats"
    # Our MMA block size must be at least the size of the scale vector.
    assert BLOCK_K >= VEC_SIZE, f"{BLOCK_K=} must be at least the size of the scale vector {VEC_SIZE=}"
    # TensorMemoryScalesLayout requires at least 32 rows when writing to tensor
    # memory. The A scales will have 128 rows because BLOCK_M must be 128 to use
    # `tcgen05_mma_scaled`, but BLOCK_N will cannot be less than 32.
    assert BLOCK_N >= 32, f"{BLOCK_N=} must be at least 32"
    assert BLOCK_M == 128, f"{BLOCK_M=} must be 128"

    # Mixed precision is when one operand is mxfp4 and the other is mxfp8.
    MIXED_PREC = A.dtype != B.dtype

    # TMA tensor descriptors require the swizzling byte width to be 128 for fp4
    # padded operands. In practice this means the TMA tensor descriptor block
    # shape along the contiguous dimension must be at least 64.
    #
    # In other words, if we have mixed precision, BLOCK_K must be at least 128
    # for the fp4 TMA descriptor's inner dimension to be at least 64.
    assert not MIXED_PREC or BLOCK_K >= 128, f"{BLOCK_K=} must be at least 128 for mixed precision fp4 operands"

    A_desc = make_operand_descriptor(A, BLOCK_M, BLOCK_K, MIXED_PREC)
    B_desc = make_operand_descriptor(B, BLOCK_N, BLOCK_K, MIXED_PREC)
    C_desc = make_output_descriptor(M, N, out_dtype, BLOCK_M, BLOCK_N)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    simple_mma_scaled_kernel[grid](A_desc, B_desc, C_desc, A_scale, *A_scale.stride(), B_scale, *B_scale.stride(),
                                   VEC_SIZE)
    return C_desc.base


# %%
# We can use the generic utilities in `triton.tools.mxfp` to manage quantized
# tensors. MXFP4Tensor wraps a tensor of sub-byte fp4 elements, and MXScaleTensor
# wraps a uint8 tensor of e8m0 MX scale factors.


def random_quantized_tensor(MN, K, format):
    assert format in ["mxfp4", "mxfp8", "nvfp4"]
    VEC_SIZE = 16 if format == "nvfp4" else 32

    # Generate a random quantized tensor and its scale factors, assuming we are
    # scaling along the K dimension.
    base = MXFP4Tensor(size=(MN, K), device="cuda").random()
    scale = MXScaleTensor(size=(MN, K // VEC_SIZE), device="cuda").random(low=1 / 128, high=2.0)

    # Compute the dequantized tensor to use for testing.
    ref = base.to(torch.float32)
    scale_ref = scale.to(torch.float32)
    value = ref * scale_ref.repeat_interleave(VEC_SIZE, dim=1)

    if format == "mxfp8":
        # For mxfp8, convert the tensor to a regular float8 torch tensor.
        return ref.to(torch.float8_e4m3fn), scale.data, value
    elif format == "mxfp4":
        # For mxfp4, pack the elements along the K dimension.
        return base.to_packed_tensor(dim=1), scale.data, value
    else:
        # For nvfp4, pack the elements along the K dimension, and convert the
        # scale factors to float8_e4m3fn.
        return base.to_packed_tensor(dim=1), scale_ref.to(torch.float8_e4m3fn), value


@pytest.mark.parametrize("M, N, K", [(2048, 2048, 4096)])
@pytest.mark.parametrize("BLOCK_N", [32, 64, 128, 256])
@pytest.mark.parametrize("BLOCK_K", [32, 64, 128, 256, 512])
@pytest.mark.parametrize("a_format, b_format",
                         list(itertools.product(["mxfp8", "mxfp4"], repeat=2)) + [("nvfp4", "nvfp4")])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_simple_mma_scaled(M, N, K, a_format, b_format, BLOCK_N, BLOCK_K):
    if a_format in ["mxfp4", "nvfp4"] and b_format in ["mxfp4", "nvfp4"] and BLOCK_K <= 32:
        pytest.skip("BLOCK_K must be greater than 32 for fp4 formats")
    if a_format != b_format and BLOCK_K < 128:
        pytest.skip("BLOCK_K must be at least 128 for mixed precision operands")
    torch.manual_seed(0)
    A, A_scale, A_ref = random_quantized_tensor(M, K, a_format)
    B, B_scale, B_ref = random_quantized_tensor(N, K, b_format)
    C_ref = A_ref @ B_ref.T
    C = simple_mma_scaled(A, B, A_scale, B_scale, VEC_SIZE=16 if a_format == "nvfp4" else 32, BLOCK_N=BLOCK_N,
                          BLOCK_K=BLOCK_K)
    torch.testing.assert_close(C_ref, C.to(torch.float32), atol=1e-3, rtol=1e-3)


# %%
# We know we can improve the performance of our simple blocked-scaled matmul
# kernel with software pipelining and/or warp-specialization. However, before we
# do that, there are a few other ways we can optimize the block-scaled matmul.
# Specifically, we want to optimize the way we handle the MMA scales.
#
# The scales are contiguous along the inner dimension, which is the K dimension.
# However, because we load the scales with block shape [BLOCK_M, BLOCK_K // VEC_SIZE],
# even for large BLOCK_K, the size of the load along the contiguous dimension will
# be less than the cache line size (128 bytes). For example, for BLOCK_K=256 and
# MX scaling (VEC_SIZE=32), the size of the load along the contiguous dimension will
# be 8 bytes. This creates inefficient global load coalescing, vectorizing, and L2
# cache utilization.

if __name__ == "__main__":
    M, N, K = 8192, 8192, 8192
    BLOCK_N = 256
    formats = [("mxfp8", "mxfp8"), ("mxfp4", "mxfp4"), ("mxfp8", "mxfp4"), ("nvfp4", "nvfp4")]
    print(f"Benchmarking simple_mma_scaled ({M=}, {N=}, {K=})")
    print("===============================================================")
    print("|    format     |   tflops/s   |")
    print("|---------------|--------------|")
    for a_format, b_format in formats:
        A, A_scale, A_ref = random_quantized_tensor(M, K, a_format)
        B, B_scale, B_ref = random_quantized_tensor(N, K, b_format)
        # Use BLOCK_K=256 when both operands are fp4, otherwise use BLOCK_K=128.
        BLOCK_K = 256 if "fp4" in a_format and "fp4" in b_format else 128
        VEC_SIZE = 16 if a_format == "nvfp4" else 32

        ms = triton.testing.do_bench_cudagraph(
            lambda: simple_mma_scaled(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K))
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"| {a_format} x {b_format} | {tflops_per_sec:>8.2f}     |")
    print()

# %%
# |    format     |   tflops/s   |
# |---------------|--------------|
# | mxfp8 x mxfp8 |    33.41     |
# | mxfp4 x mxfp4 |    67.02     |
# | mxfp8 x mxfp4 |    34.60     |
# | nvfp4 x nvfp4 |    70.84     |
#
# Performance is abysmal. However, it is unclear how much of the performance issues
# are due to the scales. If you microbenchmark the mxfp8 x mxfp8c case with
# `ncu --set full --kernel-name simple_mma_scaled_kernel`, you will see in the output:
#
# ```
# Section: Memory Workload Analysis Tables
# OPT   Est. Speedup: 15.72%
#       The memory access pattern for global loads from L1TEX might not be optimal. On average, only 4.0 of the 32
#       bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between
#       threads. Check the Source Counters section for uncoalesced global loads.
# ----- --------------------------------------------------------------------------------------------------------------
# OPT   Est. Speedup: 17.41%
#       The memory access pattern for local loads from L1TEX might not be optimal. On average, only 1.0 of the 32
#       bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between
#       threads. Check the Source Counters section for uncoalesced local loads.
# ----- --------------------------------------------------------------------------------------------------------------
# OPT   Est. Speedup: 17.41%
#       The memory access pattern for local stores to L1TEX might not be optimal. On average, only 1.0 of the 32
#       bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between
#       threads. Check the Source Counters section for uncoalesced local stores.
# ```
#
# This shows what we suspect: our scale loads from global memory are inefficient.
# We can fix the issue by changing the layout of the scales in global memory such
# that each [BLOCK_M, BLOCK_K // VEC_SIZE] block is contiguous in global memory.
#
# One naive way to do that is layout the scale tensor as
# [M // BLOCK_M, K // BLOCK_K, BLOCK_M, BLOCK_K // VEC_SIZE]
# with order=[?, ?, 1, 0], i.e. contiguous along the dim=3 and then dim=2.
#
# The first two dimensions correspond to the grid index along the M and K dimensions
# respectively, and the last two are the scales for a single program.
#
# We achieve this by dividing the block shape into the original shape by reshaping the tensor into
# [M // BLOCK_M, BLOCK_M, (K // BLOCK_K) // (BLOCK_K // VEC_SIZE), BLOCK_K // VEC_SIZE]
# and then permuting the block dimensions to the end with order (0, 2, 1, 3).


def relayout_scales_contiguous(scales: torch.Tensor, BLOCK_MN: int, BLOCK_K: int, VEC_SIZE: int):
    MN, SCALE_K = scales.shape[0], scales.shape[1]
    SCALES_BLOCK_K = BLOCK_K // VEC_SIZE
    scales = scales.reshape(MN // BLOCK_MN, BLOCK_MN, SCALE_K // SCALES_BLOCK_K, SCALES_BLOCK_K)
    scales = scales.permute(0, 2, 1, 3)
    return scales.contiguous()


# %%
# Now let's reimplement the kernel to account for the new scale layout. This
# kernel is the same as `simple_mma_scaled_kernel` except for the way it loads
# the scales.


@gluon.jit
def mma_scaled_contig_kernel(a_desc, b_desc, c_desc, a_scale_ptr, b_scale_ptr, VEC_SIZE: gl.constexpr):
    # ======= Begin unchanged code from `simple_mma_scaled_kernel` =======
    A_IS_FP4: gl.constexpr = a_desc.dtype == gl.uint8
    B_IS_FP4: gl.constexpr = b_desc.dtype == gl.uint8
    A_ELEM_PER_BYTE: gl.constexpr = 2 if A_IS_FP4 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if B_IS_FP4 else 1
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)

    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale_ptr.dtype.element_ty, [BLOCK_N, BLOCK_K // VEC_SIZE], scale_layout)
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    use_acc = False

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)
    phase = 0
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    for k in range(0, K, BLOCK_K):
        off_k_a = k // A_ELEM_PER_BYTE
        off_k_b = k // B_ELEM_PER_BYTE
        if a_desc.layout.fp4_padded:
            off_k_a *= 2
        if b_desc.layout.fp4_padded:
            off_k_b *= 2

        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, off_k_a], bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [off_n, off_k_b], bar, b_smem)
        mbarrier.wait(bar, phase)

        # ======= End unchanged code from `simple_mma_scaled_kernel` =======

        SCALE_K = K // VEC_SIZE
        SCALE_BLOCK_K: gl.constexpr = BLOCK_K // VEC_SIZE
        # We know the global memory tensor `a_scale` is contiguous with shape
        # [M // BLOCK_M, SCALE_K // SCALE_BLOCK_K, BLOCK_M, SCALE_BLOCK_K]. Each inner
        # loop tile will load `a_scale[pid_m, k // BLOCK_K, :, :]`.
        a_stride_k: gl.constexpr = BLOCK_M * SCALE_BLOCK_K
        a_stride_m = SCALE_K // SCALE_BLOCK_K * a_stride_k
        b_stride_k: gl.constexpr = BLOCK_N * SCALE_BLOCK_K
        b_stride_n = SCALE_K // SCALE_BLOCK_K * b_stride_k

        # Load `a_scale[pid_m, k // BLOCK_K, :, :]`. Since we know the inner two
        # dimensions are contiguous, we can use a 1D load for simplicity.
        coalesced_1d: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])

        a_scale_base = a_scale_ptr + pid_m * a_stride_m + k // BLOCK_K * a_stride_k
        b_scale_base = b_scale_ptr + pid_n * b_stride_n + k // BLOCK_K * b_stride_k
        a_scale = gl.load(a_scale_base + gl.arange(0, BLOCK_M * SCALE_BLOCK_K, coalesced_1d))
        b_scale = gl.load(b_scale_base + gl.arange(0, BLOCK_N * SCALE_BLOCK_K, coalesced_1d))
        a_scale = a_scale.reshape(BLOCK_M, SCALE_BLOCK_K)
        b_scale = b_scale.reshape(BLOCK_N, SCALE_BLOCK_K)

        # ======= Begin unchanged code from `simple_mma_scaled_kernel` =======
        a_scale_layout: gl.constexpr = get_tmem_reg_layout(a_scale.dtype, a_scale.type.shape, scale_layout,
                                                           gl.num_warps())
        b_scale_layout: gl.constexpr = get_tmem_reg_layout(b_scale.dtype, b_scale.type.shape, scale_layout,
                                                           gl.num_warps())
        a_scale = gl.convert_layout(a_scale, a_scale_layout)
        b_scale = gl.convert_layout(b_scale, b_scale_layout)
        a_scale_tmem.store(a_scale)
        b_scale_tmem.store(b_scale)

        a_format: gl.constexpr = "e2m1" if A_IS_FP4 else "e4m3"
        b_format: gl.constexpr = "e2m1" if B_IS_FP4 else "e4m3"
        tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                           use_acc=use_acc)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase)
        use_acc = True
        phase ^= 1

    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, gl.num_warps())
    acc = acc_tmem.load(acc_reg_layout)
    acc = acc.to(c_desc.dtype)
    acc_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
    acc_smem.store(acc)
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], acc_smem)
    tma.store_wait(0)
    # ======= End unchanged code from `simple_mma_scaled_kernel` =======


def mma_scaled_contig(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M, BLOCK_N, BLOCK_K, out_dtype=torch.float16):
    M, N = A.shape[0], B.shape[0]
    MIXED_PREC = A.dtype != B.dtype
    A_desc = make_operand_descriptor(A, BLOCK_M, BLOCK_K, MIXED_PREC)
    B_desc = make_operand_descriptor(B, BLOCK_N, BLOCK_K, MIXED_PREC)
    C_desc = make_output_descriptor(M, N, out_dtype, BLOCK_M, BLOCK_N)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mma_scaled_contig_kernel[grid](A_desc, B_desc, C_desc, A_scale, B_scale, VEC_SIZE)
    return C_desc.base


@pytest.mark.parametrize("M, N, K", [(2048, 2048, 4096)])
@pytest.mark.parametrize("BLOCK_N", [32, 64, 128, 256])
@pytest.mark.parametrize("BLOCK_K", [32, 64, 128, 256, 512])
@pytest.mark.parametrize("a_format, b_format",
                         list(itertools.product(["mxfp8", "mxfp4"], repeat=2)) + [("nvfp4", "nvfp4")])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_mma_scaled_contig(M, N, K, a_format, b_format, BLOCK_N, BLOCK_K):
    if a_format in ["mxfp4", "nvfp4"] and b_format in ["mxfp4", "nvfp4"] and BLOCK_K <= 32:
        pytest.skip("BLOCK_K must be greater than 32 for fp4 formats")
    if a_format != b_format and BLOCK_K < 128:
        pytest.skip("BLOCK_K must be at least 128 for mixed precision operands")
    torch.manual_seed(0)
    A, A_scale, A_ref = random_quantized_tensor(M, K, a_format)
    B, B_scale, B_ref = random_quantized_tensor(N, K, b_format)
    BLOCK_M = 128
    VEC_SIZE = 16 if a_format == "nvfp4" else 32
    A_scale = relayout_scales_contiguous(A_scale, BLOCK_M, BLOCK_K, VEC_SIZE)
    B_scale = relayout_scales_contiguous(B_scale, BLOCK_N, BLOCK_K, VEC_SIZE)
    C_ref = A_ref @ B_ref.T
    C = mma_scaled_contig(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M, BLOCK_N, BLOCK_K)
    torch.testing.assert_close(C_ref, C.to(torch.float32), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    M, N, K = 8192, 8192, 8192
    BLOCK_M, BLOCK_N = 128, 256
    print(f"Benchmarking mma_scaled_contiguous ({M=}, {N=}, {K=})")
    print("===============================================================")
    print("|    format     |   tflops/s   |")
    print("|---------------|--------------|")
    for a_format, b_format in formats:
        A, A_scale, A_ref = random_quantized_tensor(M, K, a_format)
        B, B_scale, B_ref = random_quantized_tensor(N, K, b_format)
        # Use BLOCK_K=256 when both operands are fp4, otherwise use BLOCK_K=128.
        BLOCK_K = 256 if "fp4" in a_format and "fp4" in b_format else 128
        VEC_SIZE = 16 if a_format == "nvfp4" else 32

        A_scale = relayout_scales_contiguous(A_scale, BLOCK_M, BLOCK_K, VEC_SIZE)
        B_scale = relayout_scales_contiguous(B_scale, BLOCK_N, BLOCK_K, VEC_SIZE)

        ms = triton.testing.do_bench_cudagraph(
            lambda: mma_scaled_contig(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M, BLOCK_N, BLOCK_K))
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"| {a_format} x {b_format} | {tflops_per_sec:>8.2f}     |")
    print()

# %%
# |    format     |   tflops/s   |
# |---------------|--------------|
# | mxfp8 x mxfp8 |   663.28     |
# | mxfp4 x mxfp4 |  1435.05     |
# | mxfp8 x mxfp4 |   741.82     |
# | nvfp4 x nvfp4 |  1303.69     |
#
# That's a huge speedup! By changing how the scales are laid out in global memory
# so that the inner loop of the kernel can load them more efficiently, we improved
# the performance of our kernel by 20x.
#
# The reason the performance of `simple_mma_scaled` is so much worse is because
# the inefficient scale loads were thrashing the L2 caches.
#
# The next thing we can consider is to use TMAs to load the scales. We will pick
# a 5D global memory layout for the scales called a "packed block" layout. For
# the A matrix, the layout is
#
# ```
# [M // (32 * 4), K // (VEC_SIZE * 4), 32, 4, 4]
# ```
#
# This way, each tensor core MMA in the matmul inner loop over the K blocks can
# achieve contiguous access of a block of 128 rows of scale factors along the M
# axis, for each [BLOCK_M, BLOCK_K] subtile of the A tensor.
#
# Later, on the GPU, we will logically permute and reshape the scales back into
# the 2D layout expected by `tcgen05_mma_scaled`.


def align_to(a, b):
    # Return next multiple of `b` greater than or equal to `a`.
    return triton.cdiv(a, b) * b


def swizzle_scales_packed_block(scales: torch.Tensor, VEC_SIZE: int):
    # When the scale tensor is not an even multiple of [128, 4], we need to pad
    # the scale tensor so it can use the packed block format.
    PAD_MN = align_to(scales.shape[0], 128) - scales.shape[0]
    PAD_K = align_to(scales.shape[1], 4) - scales.shape[1]
    scales = torch.nn.functional.pad(scales, (0, PAD_K, 0, PAD_MN))

    MN, SCALE_K = scales.shape[0], scales.shape[1]
    REP_MN = MN // 128
    REP_K = SCALE_K // 4
    scales = scales.reshape(REP_MN, 4, 32, REP_K, 4)
    scales = scales.permute(0, 3, 2, 1, 4)
    return scales.contiguous()


def make_scales_descriptor(scales: torch.Tensor, BLOCK_MN: int, BLOCK_K: int, VEC_SIZE: int):
    # Note that this 5D swizzling scheme has minimum block size requirements
    # of BLOCK_N >= 128 and BLOCK_K >= VEC_SIZE * 4 (64 for nvfp4 and 128 for MX).
    REP_MN = BLOCK_MN // 128
    REP_K = BLOCK_K // (VEC_SIZE * 4)
    # Use a 5D TMA descriptor with block shape [1, rep_m, rep_k, 2, 256] of uint8
    # elements. With 256 bytes along the inner dimension, we better utilize the
    # L2 cache and don't require the TMA engine to emit many small messages (16B)
    # as it would with 32x16xu8.
    block_shape = [1, REP_MN, REP_K, 2, 256]
    scales = scales.reshape(1, scales.shape[0], scales.shape[1], 2, 256)
    IS_NVFP4 = scales.dtype == torch.float8_e4m3fn
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float8e4nv if IS_NVFP4 else gl.uint8)
    return TensorDescriptor.from_tensor(scales, block_shape, layout)


@gluon.jit
def unswizzle_scales_packed_block(scales, BLOCK_MN: gl.constexpr, BLOCK_K: gl.constexpr, VEC_SIZE: gl.constexpr):
    # Unswizzle the scales subtile from its packed block layout.
    scales = scales.reshape(scales.shape[1], scales.shape[2], 32, 4, 4)
    scales = scales.permute(0, 3, 2, 1, 4)
    return scales.reshape(BLOCK_MN, BLOCK_K // VEC_SIZE)


@gluon.jit
def mma_scaled_packed_block_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, VEC_SIZE: gl.constexpr):
    # ======= Begin unchanged code from `simple_mma_scaled_kernel` =======
    A_IS_FP4: gl.constexpr = a_desc.dtype == gl.uint8
    B_IS_FP4: gl.constexpr = b_desc.dtype == gl.uint8
    A_ELEM_PER_BYTE: gl.constexpr = 2 if A_IS_FP4 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if B_IS_FP4 else 1
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)

    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale_desc.dtype, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale_desc.dtype, [BLOCK_N, BLOCK_K // VEC_SIZE], scale_layout)
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    use_acc = False

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)
    phase = 0
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # ======= End unchanged code from `simple_mma_scaled_kernel` =======

    # Allocate shared memory to TMA load the scales.
    a_scale_smem = gl.allocate_shared_memory(a_scale_desc.dtype, a_scale_desc.block_type.shape, a_scale_desc.layout)
    b_scale_smem = gl.allocate_shared_memory(b_scale_desc.dtype, b_scale_desc.block_type.shape, b_scale_desc.layout)
    REP_M: gl.constexpr = a_scale_desc.block_type.shape[1]
    REP_N: gl.constexpr = b_scale_desc.block_type.shape[1]
    A_REP_K: gl.constexpr = a_scale_desc.block_type.shape[2]
    B_REP_K: gl.constexpr = b_scale_desc.block_type.shape[2]
    # Index the M and N subtiles along REP_M.
    off_m_a_scale = pid_m * REP_M
    off_n_b_scale = pid_n * REP_N

    for k in range(0, K, BLOCK_K):
        off_k_a = k // A_ELEM_PER_BYTE
        off_k_b = k // B_ELEM_PER_BYTE
        if a_desc.layout.fp4_padded:
            off_k_a *= 2
        if b_desc.layout.fp4_padded:
            off_k_b *= 2
        # Index the K subtile along REP_K for each scale.
        off_k_a_scale = k // BLOCK_K * A_REP_K
        off_k_b_scale = k // BLOCK_K * B_REP_K

        mbarrier.expect(
            bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + a_scale_desc.block_type.nbytes +
            b_scale_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, off_k_a], bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [off_n, off_k_b], bar, b_smem)
        tma.async_copy_global_to_shared(a_scale_desc, [0, off_m_a_scale, off_k_a_scale, 0, 0], bar, a_scale_smem)
        tma.async_copy_global_to_shared(b_scale_desc, [0, off_n_b_scale, off_k_b_scale, 0, 0], bar, b_scale_smem)
        mbarrier.wait(bar, phase)

        # We know the destination 2D layout of the scales required to store them
        # into tensor memory. You could work backwards to figure out the layout with
        # which to load the scales from shared memory such that after unswizzling,
        # they have the right 2D layout for the store to TMEM. Instead, we will use
        # AutoLayout to let the compiler backwards propagate the layout.
        a_scale_layout: gl.constexpr = get_tmem_reg_layout(a_scale_desc.dtype, [BLOCK_M, BLOCK_K // VEC_SIZE],
                                                           scale_layout, gl.num_warps())
        b_scale_layout: gl.constexpr = get_tmem_reg_layout(b_scale_desc.dtype, [BLOCK_N, BLOCK_K // VEC_SIZE],
                                                           scale_layout, gl.num_warps())

        # Load the scales with AutoLayout. Subsequent operations, including the unswizzling,
        # will be generic over the layout.
        a_scale = a_scale_smem.load(gl.AutoLayout())
        b_scale = b_scale_smem.load(gl.AutoLayout())
        a_scale = unswizzle_scales_packed_block(a_scale, BLOCK_M, BLOCK_K, VEC_SIZE)
        b_scale = unswizzle_scales_packed_block(b_scale, BLOCK_N, BLOCK_K, VEC_SIZE)

        # Use `set_auto_layout` with the concrete scale layouts to create an anchor.
        # The compiler will propagate the layout backwards to resolve the auto layouts.
        a_scale = gl.set_auto_layout(a_scale, a_scale_layout)
        b_scale = gl.set_auto_layout(b_scale, b_scale_layout)

        # ======= Begin unchanged code from `simple_mma_scaled_kernel` =======
        a_scale_tmem.store(a_scale)
        b_scale_tmem.store(b_scale)

        a_format: gl.constexpr = "e2m1" if A_IS_FP4 else "e4m3"
        b_format: gl.constexpr = "e2m1" if B_IS_FP4 else "e4m3"
        tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                           use_acc=use_acc)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase)
        use_acc = True
        phase ^= 1

    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, gl.num_warps())
    acc = acc_tmem.load(acc_reg_layout)
    acc = acc.to(c_desc.dtype)
    acc_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
    acc_smem.store(acc)
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], acc_smem)
    tma.store_wait(0)
    # ======= End unchanged code from `simple_mma_scaled_kernel` =======


def mma_scaled_packed_block(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M, BLOCK_N, BLOCK_K, out_dtype=torch.float16):
    M, N = A.shape[0], B.shape[0]
    MIXED_PREC = A.dtype != B.dtype
    A_desc = make_operand_descriptor(A, BLOCK_M, BLOCK_K, MIXED_PREC)
    B_desc = make_operand_descriptor(B, BLOCK_N, BLOCK_K, MIXED_PREC)
    A_scale_desc = make_scales_descriptor(A_scale, BLOCK_M, BLOCK_K, VEC_SIZE)
    B_scale_desc = make_scales_descriptor(B_scale, BLOCK_N, BLOCK_K, VEC_SIZE)
    C_desc = make_output_descriptor(M, N, out_dtype, BLOCK_M, BLOCK_N)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mma_scaled_packed_block_kernel[grid](A_desc, B_desc, C_desc, A_scale_desc, B_scale_desc, VEC_SIZE)
    return C_desc.base


@pytest.mark.parametrize("M, N, K", [(2048, 2048, 4096)])
@pytest.mark.parametrize("BLOCK_N", [128, 256])
@pytest.mark.parametrize("BLOCK_K", [128, 256, 512])
@pytest.mark.parametrize("a_format, b_format",
                         list(itertools.product(["mxfp8", "mxfp4"], repeat=2)) + [("nvfp4", "nvfp4")])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_mma_scaled_packed_block(M, N, K, a_format, b_format, BLOCK_N, BLOCK_K):
    torch.manual_seed(0)
    A, A_scale, A_ref = random_quantized_tensor(M, K, a_format)
    B, B_scale, B_ref = random_quantized_tensor(N, K, b_format)
    BLOCK_M = 128
    VEC_SIZE = 16 if a_format == "nvfp4" else 32
    A_scale = swizzle_scales_packed_block(A_scale, VEC_SIZE)
    B_scale = swizzle_scales_packed_block(B_scale, VEC_SIZE)
    C_ref = A_ref @ B_ref.T
    C = mma_scaled_packed_block(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M, BLOCK_N, BLOCK_K)
    torch.testing.assert_close(C_ref, C.to(torch.float32), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    M, N, K = 8192, 8192, 8192
    BLOCK_M, BLOCK_N = 128, 256
    print(f"Benchmarking mma_scaled_packed_block ({M=}, {N=}, {K=})")
    print("===============================================================")
    print("|    format     |   tflops/s   |")
    print("|---------------|--------------|")
    for a_format, b_format in formats:
        A, A_scale, A_ref = random_quantized_tensor(M, K, a_format)
        B, B_scale, B_ref = random_quantized_tensor(N, K, b_format)
        # Use BLOCK_K=256 when both operands are fp4, otherwise use BLOCK_K=128.
        BLOCK_K = 256 if "fp4" in a_format and "fp4" in b_format else 128
        VEC_SIZE = 16 if a_format == "nvfp4" else 32

        A_scale = swizzle_scales_packed_block(A_scale, VEC_SIZE)
        B_scale = swizzle_scales_packed_block(B_scale, VEC_SIZE)

        ms = triton.testing.do_bench_cudagraph(
            lambda: mma_scaled_packed_block(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M, BLOCK_N, BLOCK_K))
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"| {a_format} x {b_format} | {tflops_per_sec:>8.2f}     |")
    print()

# %%
# |    format     |   tflops/s   |
# |---------------|--------------|
# | mxfp8 x mxfp8 |   900.97     |
# | mxfp4 x mxfp4 |  2081.76     |
# | mxfp8 x mxfp4 |  1000.48     |
# | nvfp4 x nvfp4 |  2002.05     |
#
# By using TMAs, we achieve a ~35% speedup. TMAs load large, contiguous blocks
# of memory more efficiently, and because TMA loads the scales directly into
# shared memory, we avoid most of the cost of the `convert_layout`.
#
# However, we still need to roundtrip the scales through registers to transfer
# them from shared memory to tensor memory. Next, we can apply `tcgen05_copy`,
# which we learned about in the previous tutorial, to asynchronously copy the
# scales from shared to tensor memory.
#
# To avoid this, we can instead view the shared memory in a new layout which undoes
# the swizzling. We do this by reshaping and permuting the shared memory descriptor,
# in the reverse of the way we generated the original swizzle pattern.


@gluon.jit
def unswizzle_scales_shared_memory(smem, BLOCK_MN: gl.constexpr, BLOCK_K: gl.constexpr, VEC_SIZE: gl.constexpr):
    smem = smem.reshape((smem.shape[1], smem.shape[2], 32, 4, 4))
    smem = smem.permute((0, 3, 2, 1, 4))
    return smem.reshape((BLOCK_MN, BLOCK_K // VEC_SIZE))


# %%
# But what will the layout of the final shared memory descriptor be, and will it
# be compatible with `tcgen05_copy`? To inspect the layout, we can write a small
# stub kernel and use `gl.static_print` to print constexprs.


@gluon.jit
def scales_layout_test(scales_desc, BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr, VEC_SIZE: gl.constexpr):
    smem = gl.allocate_shared_memory(scales_desc.dtype, scales_desc.block_type.shape, scales_desc.layout)
    gl.static_print(smem.type.layout)
    # We don't plan to execute this kernel, so we can use `smem` uninitialized
    # to get the forward type propagation to inspect the layout.
    smem = unswizzle_scales_shared_memory(smem, BLOCK_M, BLOCK_K, VEC_SIZE)
    gl.static_print(smem.type.layout)


if __name__ == "__main__":
    M, K = 2048, 4096
    BLOCK_M, BLOCK_K = 128, 256
    VEC_SIZE = 32
    scales = torch.empty(M, K, device="cuda", dtype=torch.uint8)
    scales = swizzle_scales_packed_block(scales, VEC_SIZE)
    scales_desc = make_scales_descriptor(scales, BLOCK_M, BLOCK_K, VEC_SIZE)
    # Invoke warmup to compile the kernel and resolve constexprs. Pass
    # TRITON_ALWAYS_COMPILE=1 to force recompilation as warmup will not run if
    # the kernel is in the cache.
    scales_layout_test.warmup(scales_desc, BLOCK_M, BLOCK_K, VEC_SIZE, grid=(1, ))

# %%
# The printed layouts are
#
# ```python
# NVMMASharedLayout(
#     swizzle_byte_width=0,
#     element_bitwidth=8,
#     rank=5,
#     transposed=False,
#     fp4_padded=False,
#     cga_layout=[]
# )
#
# SharedLinearLayout(
#    offset_bases=[[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]],
#    block_bases=[],
#    alignment=128
# )
# ```
#
# To see if this is compatible with `tcgen05_copy`, you would have to refer to the
# PTX documentation. Linear layouts can also be tricky to reason about. Instead,
# we can just try to use `tcgen05_copy` with this layout and see if the compiler complains.


@gluon.jit
def tcgen05_copy_layout_test(smem_layout: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr,
                             VEC_SIZE: gl.constexpr):
    smem = gl.allocate_shared_memory(gl.uint8, (BLOCK_M, BLOCK_K // VEC_SIZE), smem_layout)
    tmem = allocate_tensor_memory(gl.uint8, (BLOCK_M, BLOCK_K // VEC_SIZE), TensorMemoryScalesLayout())
    tcgen05_copy(smem, tmem)


if __name__ == "__main__":
    layout = gl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]],
        block_bases=[], alignment=128)
    BLOCK_M, BLOCK_K = 128, 256
    VEC_SIZE = 32
    tcgen05_copy_layout_test.warmup(layout, BLOCK_M, BLOCK_K, VEC_SIZE, grid=(1, ))

# %%
# This runs without errors, which means the layout is compatible with `tcgen05_copy`.
# If it was not compatible, the compiler would spit out an error like:
#
# ```
# failed to find valid tcgen05.copy layout from shared memory descriptor
# ```
#
# For example, `gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=2)`
# is not compatible and would trigger the above error. Also, if we change the original
# shared memory layout to have non-zero `swizzle_byte_width`, the unswizzled layout
# would trigger the same error. I.e. for NVMMASharedLayout, we have to turn off swizzling
# to use `tcgen05_copy`.
#
# This packed block layout for the scale factors was specifically designed to be
# compatible with TMAs and, when unswizzled in shared memory, produces a layout
# that is compatible with `tcgen05_copy`.
#
# For more detailed information on the scale factor layout, see
#  1. https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
#  2. https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
#
# With this information, we can rewrite the kernel to use `tcgen05_copy`.


@gluon.jit
def mma_scaled_tcgen05_copy_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, VEC_SIZE: gl.constexpr):
    # ======= Begin unchanged code from `mma_scaled_packed_block_kernel` =======
    A_IS_FP4: gl.constexpr = a_desc.dtype == gl.uint8
    B_IS_FP4: gl.constexpr = b_desc.dtype == gl.uint8
    A_ELEM_PER_BYTE: gl.constexpr = 2 if A_IS_FP4 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if B_IS_FP4 else 1
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)

    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale_desc.dtype, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale_desc.dtype, [BLOCK_N, BLOCK_K // VEC_SIZE], scale_layout)
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    use_acc = False

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)
    phase = 0
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_scale_smem = gl.allocate_shared_memory(a_scale_desc.dtype, a_scale_desc.block_type.shape, a_scale_desc.layout)
    b_scale_smem = gl.allocate_shared_memory(b_scale_desc.dtype, b_scale_desc.block_type.shape, b_scale_desc.layout)
    REP_M: gl.constexpr = a_scale_desc.block_type.shape[1]
    REP_N: gl.constexpr = b_scale_desc.block_type.shape[1]
    A_REP_K: gl.constexpr = a_scale_desc.block_type.shape[2]
    B_REP_K: gl.constexpr = b_scale_desc.block_type.shape[2]
    off_m_a_scale = pid_m * REP_M
    off_n_b_scale = pid_n * REP_N

    for k in range(0, K, BLOCK_K):
        off_k_a = k // A_ELEM_PER_BYTE
        off_k_b = k // B_ELEM_PER_BYTE
        if a_desc.layout.fp4_padded:
            off_k_a *= 2
        if b_desc.layout.fp4_padded:
            off_k_b *= 2
        off_k_a_scale = k // BLOCK_K * A_REP_K
        off_k_b_scale = k // BLOCK_K * B_REP_K

        mbarrier.expect(
            bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + a_scale_desc.block_type.nbytes +
            b_scale_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, off_k_a], bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [off_n, off_k_b], bar, b_smem)
        tma.async_copy_global_to_shared(a_scale_desc, [0, off_m_a_scale, off_k_a_scale, 0, 0], bar, a_scale_smem)
        tma.async_copy_global_to_shared(b_scale_desc, [0, off_n_b_scale, off_k_b_scale, 0, 0], bar, b_scale_smem)
        mbarrier.wait(bar, phase)

        # ======= End unchanged code from `mma_scaled_packed_block_kernel` =======

        # Unswizzle the scales in shared memory.
        a_scale = unswizzle_scales_shared_memory(a_scale_smem, BLOCK_M, BLOCK_K, VEC_SIZE)
        b_scale = unswizzle_scales_shared_memory(b_scale_smem, BLOCK_N, BLOCK_K, VEC_SIZE)
        # Issue the async copies to tensor memory. Recall `tcgen05_copy` is implicitly
        # pipelined with `tcgen05_mma_scaled`, so we don't need to explicitly
        # synchronize them.
        tcgen05_copy(a_scale, a_scale_tmem)
        tcgen05_copy(b_scale, b_scale_tmem)

        # ======= Begin unchanged code from `mma_scaled_packed_block_kernel` =======

        a_format: gl.constexpr = "e2m1" if A_IS_FP4 else "e4m3"
        b_format: gl.constexpr = "e2m1" if B_IS_FP4 else "e4m3"
        tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                           use_acc=use_acc)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase)
        use_acc = True
        phase ^= 1

    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, gl.num_warps())
    acc = acc_tmem.load(acc_reg_layout)
    acc = acc.to(c_desc.dtype)
    acc_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
    acc_smem.store(acc)
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], acc_smem)
    tma.store_wait(0)
    # ======= End unchanged code from `mma_scaled_packed_block_kernel` =======


def mma_scaled_tcgen05_copy(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M, BLOCK_N, BLOCK_K, out_dtype=torch.float16):
    M, N = A.shape[0], B.shape[0]
    MIXED_PREC = A.dtype != B.dtype
    A_desc = make_operand_descriptor(A, BLOCK_M, BLOCK_K, MIXED_PREC)
    B_desc = make_operand_descriptor(B, BLOCK_N, BLOCK_K, MIXED_PREC)
    C_desc = make_output_descriptor(M, N, out_dtype, BLOCK_M, BLOCK_N)
    A_scale_desc = make_scales_descriptor(A_scale, BLOCK_M, BLOCK_K, VEC_SIZE)
    B_scale_desc = make_scales_descriptor(B_scale, BLOCK_N, BLOCK_K, VEC_SIZE)

    # Replace the TMA descriptor layouts to have no swizzling in order for the
    # unswizzled layout to be compatible with `tcgen05_copy`.
    no_swizzle_layout = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5)
    A_scale_desc = replace(A_scale_desc, layout=no_swizzle_layout)
    B_scale_desc = replace(B_scale_desc, layout=no_swizzle_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mma_scaled_tcgen05_copy_kernel[grid](A_desc, B_desc, C_desc, A_scale_desc, B_scale_desc, VEC_SIZE)
    return C_desc.base


@pytest.mark.parametrize("M, N, K", [(2048, 2048, 4096)])
@pytest.mark.parametrize("BLOCK_N", [128, 256])
@pytest.mark.parametrize("BLOCK_K", [128, 256, 512])
@pytest.mark.parametrize("a_format, b_format",
                         list(itertools.product(["mxfp8", "mxfp4"], repeat=2)) + [("nvfp4", "nvfp4")])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_mma_scaled_tcgen05_copy(M, N, K, a_format, b_format, BLOCK_N, BLOCK_K):
    torch.manual_seed(0)
    A, A_scale, A_ref = random_quantized_tensor(M, K, a_format)
    B, B_scale, B_ref = random_quantized_tensor(N, K, b_format)
    BLOCK_M = 128
    VEC_SIZE = 16 if a_format == "nvfp4" else 32
    A_scale = swizzle_scales_packed_block(A_scale, VEC_SIZE)
    B_scale = swizzle_scales_packed_block(B_scale, VEC_SIZE)
    C_ref = A_ref @ B_ref.T
    C = mma_scaled_tcgen05_copy(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M, BLOCK_N, BLOCK_K)
    torch.testing.assert_close(C_ref, C.to(torch.float32), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    M, N, K = 8192, 8192, 8192
    BLOCK_M, BLOCK_N = 128, 256
    print(f"Benchmarking mma_scaled_tcgen05_copy ({M=}, {N=}, {K=})")
    print("===============================================================")
    print("|    format     |   tflops/s   |")
    print("|---------------|--------------|")
    for a_format, b_format in formats:
        A, A_scale, A_ref = random_quantized_tensor(M, K, a_format)
        B, B_scale, B_ref = random_quantized_tensor(N, K, b_format)
        # Use BLOCK_K=256 when both operands are fp4, otherwise use BLOCK_K=128.
        BLOCK_K = 256 if "fp4" in a_format and "fp4" in b_format else 128
        VEC_SIZE = 16 if a_format == "nvfp4" else 32

        A_scale = swizzle_scales_packed_block(A_scale, VEC_SIZE)
        B_scale = swizzle_scales_packed_block(B_scale, VEC_SIZE)

        ms = triton.testing.do_bench_cudagraph(
            lambda: mma_scaled_tcgen05_copy(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M, BLOCK_N, BLOCK_K))
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"| {a_format} x {b_format} | {tflops_per_sec:>8.2f}     |")
    print()

# %%
# |    format     |   tflops/s   |
# |---------------|--------------|
# | mxfp8 x mxfp8 |   929.07     |
# | mxfp4 x mxfp4 |  2147.76     |
# | mxfp8 x mxfp4 |  1035.60     |
# | nvfp4 x nvfp4 |  2092.39     |
#
# Using `tcgen05_copy`, we observe a modest speedup to the kernel. To achieve
# the remaining performance, we will demonstrate a software pipelined and
# warp-specialized version of the block-scaled matmul.
#
# Before we begin, notice that the `tcgen05_copy` of the scales into tensor memory
# followed by `tcgen05_mma_scaled` can be abstracted as a single async MMA instruction
# with 4 shared memory inputs. Then, we can pipeline it like a regular async MMA.


@gluon.jit
def async_mma_scaled_impl(a_smem, b_smem, a_scale_smem, b_scale_smem, acc_tmem, use_acc, pred):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_smem.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = a_smem.shape[0]
    BLOCK_N: gl.constexpr = b_smem.shape[0]
    BLOCK_K: gl.constexpr = a_smem.shape[1] * A_ELEM_PER_BYTE
    # Recall we use `uint8` to represent fp4 elements.
    VEC_SIZE: gl.constexpr = 32 if a_scale_smem.dtype == gl.uint8 else 16

    a_scale = unswizzle_scales_shared_memory(a_scale_smem, BLOCK_M, BLOCK_K, VEC_SIZE)
    b_scale = unswizzle_scales_shared_memory(b_scale_smem, BLOCK_N, BLOCK_K, VEC_SIZE)

    # We don't need to hoist the scales tensor memory allocations outside of the loop,
    # so we can pull them into this helper function.
    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype, a_scale.type.shape, scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype, b_scale.type.shape, scale_layout)
    tcgen05_copy(a_scale, a_scale_tmem)
    tcgen05_copy(b_scale, b_scale_tmem)

    a_format: gl.constexpr = "e2m1" if a_smem.dtype == gl.uint8 else "e4m3"
    b_format: gl.constexpr = "e2m1" if b_smem.dtype == gl.uint8 else "e4m3"
    tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                       use_acc=use_acc, pred=pred)


# This helper function computes all the load indexing and issues the async loads
# based on the current `pid_m`, `pid_n`, and `k` indices. The compiler will run
# loop-invariant code motion to hoist code that does not depend on `k`, like
# `pid_m * BLOCK_M`, outside of the inner loop, so we can safely abstract the
# load indexing without performance loss.
#
# Encapsulating the load indexing logic will help keep our pipelined kernel code
# clean, as pipelining can get messy.
@gluon.jit
def issue_loads(producer, pid_m, pid_n, k, a_desc, b_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs,
                b_scale_bufs, bars, pred):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_desc.dtype == gl.uint8 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if b_desc.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = b_desc.block_type.shape[0]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    REP_M: gl.constexpr = a_scale_desc.block_type.shape[1]
    REP_N: gl.constexpr = b_scale_desc.block_type.shape[1]
    A_REP_K: gl.constexpr = a_scale_desc.block_type.shape[2]
    B_REP_K: gl.constexpr = b_scale_desc.block_type.shape[2]

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    off_m_a_scale = pid_m * REP_M
    off_n_b_scale = pid_n * REP_N
    off_k_a = k // A_ELEM_PER_BYTE
    off_k_b = k // B_ELEM_PER_BYTE
    if a_desc.layout.fp4_padded:
        off_k_a *= 2
    if b_desc.layout.fp4_padded:
        off_k_b *= 2
    off_k_a_scale = (k // BLOCK_K) * A_REP_K
    off_k_b_scale = (k // BLOCK_K) * B_REP_K

    index = producer.index
    bar = bars.index(index)
    mbarrier.expect(
        bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + a_scale_desc.block_type.nbytes +
        b_scale_desc.block_type.nbytes, pred)
    tma.async_copy_global_to_shared(a_desc, [off_m, off_k_a], bar, a_bufs.index(index), pred)
    tma.async_copy_global_to_shared(b_desc, [off_n, off_k_b], bar, b_bufs.index(index), pred)
    tma.async_copy_global_to_shared(a_scale_desc, [0, off_m_a_scale, off_k_a_scale, 0, 0], bar,
                                    a_scale_bufs.index(index), pred)
    tma.async_copy_global_to_shared(b_scale_desc, [0, off_n_b_scale, off_k_b_scale, 0, 0], bar,
                                    b_scale_bufs.index(index), pred)
    return producer.next(pred)


@gluon.jit
def issue_mma(consumer, c_bars, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs, producer, p_bars, acc_tmem, use_acc, pred):
    c_index = consumer.index
    mbarrier.wait(c_bars.index(c_index), consumer.phase, pred)
    async_mma_scaled_impl(a_bufs.index(c_index), b_bufs.index(c_index), a_scale_bufs.index(c_index),
                          b_scale_bufs.index(c_index), acc_tmem, use_acc, pred)
    tcgen05_commit(p_bars.index(producer.index), pred)
    return consumer.next(pred), producer.next(pred)


@gluon.jit
def mma_scaled_pipelined_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, num_buffers: gl.constexpr,
                                SchedulerImpl: gl.constexpr):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_desc.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    # The scale loads are much smaller than the operand loads (by a factor of VEC_SIZE).
    # We could use fewer buffers for the scales than the operands to save shared memory
    # as the scale load latency is lower, but this is left as an exercise for the reader.
    a_scale_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [num_buffers] + a_scale_desc.block_type.shape,
                                             a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [num_buffers] + b_scale_desc.block_type.shape,
                                             b_scale_desc.layout)
    acc_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)

    load_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_bars.index(i), count=1)
    load_producer = t8.Counter.create(0, num_buffers)
    load_consumer = t8.Counter.create(0, num_buffers)

    # If BLOCK_N=256, double-buffering the accumulator will use all 512 columns
    # of tensor memory, which leaves no room for the scales' tensor memory.
    num_acc_buffers: gl.constexpr = 2 if BLOCK_N < 256 else 1
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, gl.num_warps())
    acc_bufs = allocate_tensor_memory(gl.float32, [num_acc_buffers, BLOCK_M, BLOCK_N], tmem_layout)
    acc_idx = 0

    mma_bars = gl.allocate_shared_memory(gl.int64, [num_acc_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_acc_buffers):
        mbarrier.init(mma_bars.index(i), count=1)
    mma_producer = t8.Counter.create(0, num_acc_buffers)
    mma_consumer = t8.Counter.create(0, num_acc_buffers)

    scheduler = SchedulerImpl.initialize(c_desc.shape[0], c_desc.shape[1], BLOCK_M, BLOCK_N)
    num_tiles = scheduler.get_num_tiles()

    # Peeled inner loop prologue. Use predicates to mask peeled iterations that
    # would be out-of-bounds if K is too small, but assume K > 0, i.e. we execute
    # at least one inner loop iteration.
    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        load_producer = issue_loads(load_producer, pid_m, pid_n, ki, a_desc, b_desc, a_scale_desc, b_scale_desc, a_bufs,
                                    b_bufs, a_scale_bufs, b_scale_bufs, load_bars, pred=ki < K)
    k = BLOCK_K * (num_buffers - 2)
    load_producer = issue_loads(load_producer, pid_m, pid_n, k, a_desc, b_desc, a_scale_desc, b_scale_desc, a_bufs,
                                b_bufs, a_scale_bufs, b_scale_bufs, load_bars, pred=k < K)

    load_consumer, mma_producer = issue_mma(load_consumer, load_bars, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs,
                                            mma_producer, mma_bars, acc_bufs.index(acc_idx), use_acc=False, pred=True)
    for _ in range(num_tiles):
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            load_producer = issue_loads(load_producer, pid_m, pid_n, k, a_desc, b_desc, a_scale_desc, b_scale_desc,
                                        a_bufs, b_bufs, a_scale_bufs, b_scale_bufs, load_bars, pred=True)
            load_consumer, mma_producer = issue_mma(load_consumer, load_bars, a_bufs, b_bufs,
                                                    a_scale_bufs, b_scale_bufs, mma_producer, mma_bars,
                                                    acc_bufs.index(acc_idx), use_acc=True, pred=True)
            # Wait for the N-1th MMA to complete so we can keep issuing loads.
            mbarrier.wait(mma_bars.index(mma_consumer.index), mma_consumer.phase)
            mma_consumer = mma_consumer.next()

        # Peel the next prologue and fuse it with the pipeline drain loop.
        epilogue_pid_m, epilogue_pid_n = pid_m, pid_n
        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        has_next_tile = idx < num_tiles
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            load_producer = issue_loads(load_producer, pid_m, pid_n, ki, a_desc, b_desc, a_scale_desc, b_scale_desc,
                                        a_bufs, b_bufs, a_scale_bufs, b_scale_bufs, load_bars, has_next_tile and ki < K)

            pred = K > ki + BLOCK_K
            load_consumer, mma_producer = issue_mma(load_consumer, load_bars, a_bufs, b_bufs,
                                                    a_scale_bufs, b_scale_bufs, mma_producer, mma_bars,
                                                    acc_bufs.index(acc_idx), use_acc=True, pred=pred)
            mbarrier.wait(mma_bars.index(mma_consumer.index), mma_consumer.phase, pred)
            mma_consumer = mma_consumer.next(pred)

        k = BLOCK_K * (num_buffers - 2)
        load_producer = issue_loads(load_producer, pid_m, pid_n, k, a_desc, b_desc, a_scale_desc, b_scale_desc, a_bufs,
                                    b_bufs, a_scale_bufs, b_scale_bufs, load_bars, pred=has_next_tile and k < K)
        cur_acc_buf = acc_bufs.index(acc_idx)

        # Compared to Hopper, we can overlap Blackwell MMAs a little bit more because
        # the accumulator is stored in tensor memory. When the accumulator is not
        # double-buffered, we will start the MMA of the next tile after loading the
        # final accumulator of the current tile, but before initiating the TMA store.
        # When the accumulator is double-buffered, we can the start first MMA of the next tile
        # before the last MMA of the current tile completes.
        if num_acc_buffers == 2:
            acc_idx ^= 1
            load_consumer, mma_producer = issue_mma(load_consumer, load_bars, a_bufs, b_bufs,
                                                    a_scale_bufs, b_scale_bufs, mma_producer, mma_bars,
                                                    acc_bufs.index(acc_idx), use_acc=False, pred=has_next_tile)
        mbarrier.wait(mma_bars.index(mma_consumer.index), mma_consumer.phase)
        mma_consumer = mma_consumer.next()
        acc = cur_acc_buf.load(acc_reg_layout)
        if num_acc_buffers == 1:
            load_consumer, mma_producer = issue_mma(load_consumer, load_bars, a_bufs, b_bufs,
                                                    a_scale_bufs, b_scale_bufs, mma_producer, mma_bars,
                                                    acc_bufs.index(acc_idx), use_acc=False, pred=has_next_tile)

        acc = acc.to(c_desc.dtype)
        # Pipeline the store by waiting for the previous store to complete.
        tma.store_wait(0)
        acc_smem.store(acc)
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [epilogue_pid_m * BLOCK_M, epilogue_pid_n * BLOCK_N], acc_smem)

    # Wait for the last store.
    tma.store_wait(0)
    for i in gl.static_range(num_buffers):
        mbarrier.invalidate(load_bars.index(i))
    for i in gl.static_range(num_acc_buffers):
        mbarrier.invalidate(mma_bars.index(i))


# %%
# We also provide an example warp-specialized implementation. The helpers we
# wrote simplify writing the warp-specialized code.


@aggregate
class PartitionArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_scale_desc: tma.tensor_descriptor
    b_scale_desc: tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    a_scale_bufs: gl.shared_memory_descriptor
    b_scale_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    SchedulerImpl: gl.constexpr

    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    M: gl.tensor
    N: gl.tensor
    K: gl.tensor

    @gluon.constexpr_function
    def __init__(self, a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs,
                 load_empty_bars, load_ready_bars, acc_bufs, acc_empty_bars, acc_ready_bars, SchedulerImpl, BLOCK_M,
                 BLOCK_N, BLOCK_K, M, N, K):
        self.a_desc = a_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.a_scale_desc = a_scale_desc
        self.b_scale_desc = b_scale_desc
        self.a_bufs = a_bufs
        self.b_bufs = b_bufs
        self.a_scale_bufs = a_scale_bufs
        self.b_scale_bufs = b_scale_bufs
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.acc_bufs = acc_bufs
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
        self.SchedulerImpl = gl.constexpr(SchedulerImpl)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.M = M
        self.N = N
        self.K = K


@gluon.jit
def mma_scaled_load_partition(p):
    state = t8.Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(p.M, p.N, p.BLOCK_M, p.BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        for k in range(0, p.K, p.BLOCK_K):
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)
            state = issue_loads(state, pid_m, pid_n, k, p.a_desc, p.b_desc, p.a_scale_desc, p.b_scale_desc, p.a_bufs,
                                p.b_bufs, p.a_scale_bufs, p.b_scale_bufs, p.load_ready_bars, pred=True)


@gluon.jit
def mma_scaled_mma_partition(p):
    load_state = t8.Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = t8.Counter.create(1, p.acc_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(p.M, p.N, p.BLOCK_M, p.BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        use_acc = False
        for k in range(0, p.K, p.BLOCK_K):
            _, load_state = issue_mma(load_state, p.load_ready_bars, p.a_bufs, p.b_bufs, p.a_scale_bufs, p.b_scale_bufs,
                                      load_state, p.load_empty_bars, acc_buf, use_acc, pred=True)
            use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()


@gluon.jit
def mma_scaled_epilogue_partition(p):
    acc_layout: gl.constexpr = get_tmem_reg_layout(p.c_desc.dtype, (p.BLOCK_M, p.BLOCK_N), p.acc_bufs.type.layout,
                                                   gl.num_warps())
    acc_state = t8.Counter.create(0, p.acc_empty_bars.shape[0])
    acc_smem = gl.allocate_shared_memory(p.c_desc.dtype, p.c_desc.block_type.shape, p.c_desc.layout)
    scheduler = p.SchedulerImpl.initialize(p.M, p.N, p.BLOCK_M, p.BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc = p.acc_bufs.index(acc_state.index).load(acc_layout)
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()

        tma.store_wait(0)
        acc_smem.store(acc.to(p.c_desc.dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(p.c_desc, [pid_m * p.BLOCK_M, pid_n * p.BLOCK_N], acc_smem)
    tma.store_wait(0)


@gluon.jit
def mma_scaled_warp_specialized_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, num_buffers: gl.constexpr,
                                       SchedulerImpl: gl.constexpr):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_desc.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    M = c_desc.shape[0]
    N = c_desc.shape[1]
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    a_scale_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [num_buffers] + a_scale_desc.block_type.shape,
                                             a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [num_buffers] + b_scale_desc.block_type.shape,
                                             b_scale_desc.layout)
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    num_acc_buffers: gl.constexpr = 2 if BLOCK_N < 256 else 1
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_bufs = allocate_tensor_memory(gl.float32, [num_acc_buffers, BLOCK_M, BLOCK_N], tmem_layout)
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [num_acc_buffers, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [num_acc_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_acc_buffers):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    p = PartitionArgs(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs,
                      load_empty_bars, load_ready_bars, acc_bufs, acc_empty_bars, acc_ready_bars, SchedulerImpl,
                      BLOCK_M, BLOCK_N, BLOCK_K, M, N, K)
    gl.warp_specialize([
        (mma_scaled_epilogue_partition, (p, )),
        (mma_scaled_mma_partition, (p, )),
        (mma_scaled_load_partition, (p, )),
    ], [1, 1], [24, 24])


def mma_scaled(A, B, A_scale, B_scale, VEC_SIZE, impl_kernel, GROUP_SIZE_M=8, out_dtype=torch.float16):
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 128 if torch.float8_e4m3fn in [A.dtype, B.dtype] else 256
    SchedulerImpl = t7.GroupedPersistentTileScheduler(GROUP_SIZE_M)
    M, N = A.shape[0], B.shape[0]
    MIXED_PREC = A.dtype != B.dtype
    A_desc = make_operand_descriptor(A, BLOCK_M, BLOCK_K, MIXED_PREC)
    B_desc = make_operand_descriptor(B, BLOCK_N, BLOCK_K, MIXED_PREC)
    C_desc = make_output_descriptor(M, N, out_dtype, BLOCK_M, BLOCK_N)
    A_scale_desc = make_scales_descriptor(A_scale, BLOCK_M, BLOCK_K, VEC_SIZE)
    B_scale_desc = make_scales_descriptor(B_scale, BLOCK_N, BLOCK_K, VEC_SIZE)

    # Replace the TMA descriptor layouts to have no swizzling in order for the
    # unswizzled layout to be compatible with `tcgen05_copy`.
    no_swizzle_layout = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5)
    A_scale_desc = replace(A_scale_desc, layout=no_swizzle_layout)
    B_scale_desc = replace(B_scale_desc, layout=no_swizzle_layout)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    # mma_scaled_pipelined_kernel[grid](A_desc, B_desc, C_desc, A_scale_desc, B_scale_desc, 3, SchedulerImpl)
    impl_kernel[grid](A_desc, B_desc, C_desc, A_scale_desc, B_scale_desc, 3, SchedulerImpl)
    return C_desc.base


@pytest.mark.parametrize("K", [128, 640, 704, 1152, 4096])
@pytest.mark.parametrize("M, N", [(2048, 2048), (500, 600), (128, 128), (8192, 8192)])
@pytest.mark.parametrize("a_format, b_format",
                         list(itertools.product(["mxfp8", "mxfp4"], repeat=2)) + [("nvfp4", "nvfp4")])
@pytest.mark.parametrize("impl_kernel", [mma_scaled_pipelined_kernel, mma_scaled_warp_specialized_kernel],
                         ids=("pipelined", "warp-specialized"))
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_mma_scaled_pipelined(M, N, K, a_format, b_format, impl_kernel):
    if a_format != b_format and K % 128 != 0:
        pytest.skip("fp4 packed tensor descriptor requires K to be a multiple of 128")
    torch.manual_seed(0)
    A, A_scale, A_ref = random_quantized_tensor(M, K, a_format)
    B, B_scale, B_ref = random_quantized_tensor(N, K, b_format)
    VEC_SIZE = 16 if a_format == "nvfp4" else 32
    A_scale = swizzle_scales_packed_block(A_scale, VEC_SIZE)
    B_scale = swizzle_scales_packed_block(B_scale, VEC_SIZE)
    C_ref = A_ref @ B_ref.T
    C = mma_scaled(A, B, A_scale, B_scale, VEC_SIZE, impl_kernel)
    torch.testing.assert_close(C_ref, C.to(torch.float32), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    M, N, K = 8192, 8192, 8192
    print(f"Benchmarking mma_scaled ({M=}, {N=}, {K=})")
    print("===============================================================")
    print("|    format     | pipelined tflops/s | warp-specialized tflops/s |")
    print("|---------------|--------------------|---------------------------|")
    for a_format, b_format in formats:
        A, A_scale, A_ref = random_quantized_tensor(M, K, a_format)
        B, B_scale, B_ref = random_quantized_tensor(N, K, b_format)
        # Use BLOCK_K=256 when both operands are fp4, otherwise use BLOCK_K=128.
        BLOCK_K = 256 if "fp4" in a_format and "fp4" in b_format else 128
        VEC_SIZE = 16 if a_format == "nvfp4" else 32

        A_scale = swizzle_scales_packed_block(A_scale, VEC_SIZE)
        B_scale = swizzle_scales_packed_block(B_scale, VEC_SIZE)

        ms = triton.testing.do_bench_cudagraph(
            lambda: mma_scaled(A, B, A_scale, B_scale, VEC_SIZE, mma_scaled_pipelined_kernel))
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"| {a_format} x {b_format} |           {tflops_per_sec:>8.2f} |", end="")

        ms = triton.testing.do_bench_cudagraph(
            lambda: mma_scaled(A, B, A_scale, B_scale, VEC_SIZE, mma_scaled_warp_specialized_kernel))
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"                  {tflops_per_sec:>8.2f} |")
    print()

# %%
# |    format     | pipelined tflops/s | warp-specialized tflops/s |
# |---------------|--------------------|---------------------------|
# | mxfp8 x mxfp8 |            2018.58 |                   2378.49 |
# | mxfp4 x mxfp4 |            3916.62 |                   4870.97 |
# | mxfp8 x mxfp4 |            2144.05 |                   2615.73 |
# | nvfp4 x nvfp4 |            3842.19 |                   4846.83 |
#
# As anticipated, we get a huge speedup. In fact, we get pretty close to the
# 5 petaflops NVIDIA marketing promised us.
#
# Although the software pipelined version is slower, it was useful nonetheless
# to demonstrate how to implement one as there are cases where software pipelining
# will be faster than warp-specialization. We also took the chance to demonstrate
# the extra overlap we can achieve with Blackwell MMAs compared to Hopper MMAs.
#
# We also showed how, with `tcgen05_copy`, we can abstract the MMA scaled into
# an async MMA operation and pipeline or warp-specialize it the same way as `tcgen05_mma`.
#
# The main takeaways from this tutorial:
# - The global memory layout of the scales is important and drastically affects
#   performance.
# - `tcgen05_copy` is a great way to copy the scales into tensor memory.
