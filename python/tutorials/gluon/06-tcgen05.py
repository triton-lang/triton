"""
The 5th Generation TensorCore^TM
================================

This tutorial covers the APIs for interacting with Tensor Cores on Blackwell
GPUs. Blackwell Tensor Cores introduce a new memory space called Tensor Memory
that must be used to interact with the async MMA instructions.

In this tutorial, we will cover allocating and interacting with Tensor Memory
and demonstrate how to use the `tcgen05` MMA instructions. We will build a
simple matmul kernel to demonstrate practical uses of the APIs and show an
example of how to pipeline MMA instructions.
"""

import itertools
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_32x32b_reg_layout,
    tma,
    mbarrier,
    tcgen05_mma,
    tcgen05_commit,
    fence_async_shared,
)


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")

# %%
# Tensor memory is a 2D memory space organized into 128 rows and 512 columns of
# 32-bit cells per SM. Accessing tensor memory is significantly faster than
# shared memory, but there are additional limitations:
#
# - Each warp can only access 32 rows of tensor memory based on its warp ID,
#   thus a whole warp group is required to collectively access all 128 rows.
# - Tensor memory is allocated by number of columns. The allocation size must be
#   a power of 2 in the range [32, 512].
# - In Gluon, tensor memory load and store operations require 4 or 8 warps.
# - In Gluon, only 2D tensors can be loaded from and stored to tensor memory.
# - Data can be asynchronously copied from shared memory to tensor memory, but
#   this API is not yet exposed in Gluon.
#
# Data stored in tensor memory has layouts, just like shared memory. Due to the
# tensor memory restrictions, the register layout of tensors being stored to or
# loaded from tensor memory is constrained by the tensor memory layout.
#
# A few more notes on tensor memory:
#
# - Tensor memory is essentially an extra register file. You will notice that
#   128 * 512 = 64K 32-bit cells, just like the SM register file.
# - Tensor memory can be used independent of MMA instructions. It can be used
#   in-place of shared memory to transfer data, as permitted by the layout
#   restrictions.
# - Tensor memory is dynamically allocated on the SM, so while tensor memory
#   does not directly affect occupancy, the allocation will block if there is
#   not enough tensor memory available.

# %%
# Tensor memory layouts organize data into 2D blocks:
#
# ```python
# TensorMemoryLayout(
#     block=(blockM, blockN),
#     unpacked=True,
# )
#
# The tensor is divided into (blockM, blockN) blocks, where blockM must be 64
# or 128. blockN must be a power of 2 between [1, 256]. For dtypes smaller than
# 32 bits, multiple elements can be packed into each 32-bit cell if
# unpacked=False, however blockN must then be at least `32 // bitwidth`.
#
# Note that when blockM=64, tensors with multiple blocks are packed in TMEM to
# use all 128 rows. This can complicate slicing TMEM descriptors.
#
# The underlying `tcgen05.st` and `tcgen05.ld` instructions are warp-level
# instructions that access TMEM in specific patterns. Combined with the warp
# row-addressing restrictions, this gives rise to the register layout
# restrictions on tensor memory. Certain tensor memory layouts support multiple
# register layouts, which affect the selected atom. In this tutorial, we will
# only use the `32x32b` atom: each lane stores and loads 1 row of TMEM.


@gluon.jit
def tmem_example_kernel(in_ptr, out_ptr, M: gl.constexpr, N: gl.constexpr, num_warps: gl.constexpr):
    global_memory_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])

    offs_m = gl.arange(0, M, gl.SliceLayout(1, global_memory_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, global_memory_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]

    input = gl.load(in_ptr + offs)

    # Allocate some tensor memory.
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        block=(64, 64),
        unpacked=True,
    )

    tmem = allocate_tensor_memory(
        element_ty=in_ptr.dtype.element_ty,
        shape=[M, N],
        layout=tmem_layout,
    )

    # Get the register layout needed to access the tensor memory using a helper.
    tmem_reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(
        M=64,
        N=64,
        shape=[M, N],
        num_warps=num_warps,
    )

    input = gl.convert_layout(input, tmem_reg_layout)
    tmem.store(input)
    output = tmem.load(tmem_reg_layout)
    output = gl.convert_layout(output, global_memory_layout)

    gl.store(out_ptr + offs, output)


@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_example_kernel(M, N, num_warps):
    input = torch.randn(M, N, dtype=torch.float32, device="cuda")
    output = torch.empty_like(input)

    tmem_example_kernel[(1, )](input, output, M, N, num_warps=num_warps)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


# %%
# Now let's illustrate how TMEM how is used to do MMA operations with a trivial
# kernel launched with grid size (1, ) that performs MMA on a small tensor.


@gluon.jit
def small_mma_kernel(a_desc, b_desc, c_desc, d_desc, tmem_block: gl.constexpr,  #
                     LHS_IN_TMEM: gl.constexpr, USE_COMMIT: gl.constexpr, num_warps: gl.constexpr):
    # Load A, B, and C tiles.
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    # A has shape [M, K].
    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    # B has shape [K, N].
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)
    # C has shape [M, N].
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)

    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [0, 0], bar, a_smem)
    tma.async_copy_global_to_shared(b_desc, [0, 0], bar, b_smem)
    tma.async_copy_global_to_shared(c_desc, [0, 0], bar, c_smem)
    mbarrier.wait(bar, phase=0)

    # Re-using an mbarrier for TMAs and tcgen05_mma can lead to undefined
    # behaviour. Make sure to use a separate mbarrier or re-initialize it.
    mbarrier.invalidate(bar)
    mbarrier.init(bar, count=1)

    # The accumulator operand must be provided in TMEM. The LHS operand can be
    # provided in either SMEM or TMEM. The RHS operand must be provided in SMEM.
    # SMEM operands must have an NVMMASharedLayout.
    M: gl.constexpr = d_desc.block_type.shape[0]
    N: gl.constexpr = d_desc.block_type.shape[1]
    K: gl.constexpr = a_desc.block_type.shape[1]

    # Copy operands into TMEM.
    # TODO: Use `tcgen05.cp` when it is exposed in Gluon.
    acc_tmem_layout: gl.constexpr = TensorMemoryLayout(tmem_block.value, unpacked=True)
    acc_tmem = allocate_tensor_memory(d_desc.dtype, [M, N], acc_tmem_layout)
    acc_reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(tmem_block[0], tmem_block[1], [M, N], num_warps)
    acc = c_smem.load(acc_reg_layout)
    acc_tmem.store(acc)

    if LHS_IN_TMEM:
        # When the LHS operand is fp16 or fp8, it is packed in TMEM.
        lhs_tmem_layout: gl.constexpr = TensorMemoryLayout(tmem_block.value, unpacked=False)
        lhs_tmem = allocate_tensor_memory(a_desc.dtype, [M, K], lhs_tmem_layout)

        lhs_reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(M, K, [M, K], num_warps)
        lhs = a_smem.load(lhs_reg_layout)
        lhs_tmem.store(lhs)
        a = lhs_tmem
    else:
        a = a_smem

    # tcgen05_mma is an asynchronous operation. Until the operation is complete,
    # we cannot read or write to the accumulator memory and we cannot write to
    # the operand memory. tcgen05_mma accesses shared memory through the async
    # proxy:
    #
    # ```python
    # b_smem.store(b)
    # fence_async_shared()
    # tcgen05_mma(a, b_smem, acc_tmem)
    # ```
    #
    # A fence is required between the shared store and tcgen05_mma to order
    # their shared memory accesses. Completion of the tcgen05_mma operation
    # implies its reads from shared memory are complete, thus it would be safe
    # to write to the shared memory inputs after waiting without a fence.
    #
    # Completion of tcgen05_mma operations is tracked with mbarriers. Invoking
    # tcgen05_commit on an mbarrier causes the mbarrier to be arrived on when
    # all previously issued tcgen05_mma operations have been completed. See
    # 04-tma.py for more details on how mbarriers work.
    #
    # To commit on an mbarrier, we can either explicitly invoke tcgen05_commit
    # or pass the mbarrier directly to tcgen05_mma. We can also conditionally
    # commit an mbarrier if necessary.
    #
    # tcgen05_mma is comprised of multiple async MMA instructions. The shape of
    # each instruction is determined by the TMEM layout. Selecting larger
    # instruction shapes generally results in better performance. Note that
    # tcgen05_mma only supports blockM=64 when there is 1 block.
    if USE_COMMIT:
        tcgen05_mma(a, b_smem, acc_tmem)
        tcgen05_commit(bar)
    else:
        tcgen05_mma(a, b_smem, acc_tmem, mbarriers=[bar], mbarrier_preds=[True])

    # Wait for the completion of the MMA.
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # Another important flag to consider is `use_acc`. When `use_acc=False`, the
    # current value of the accumulator in TMEM is ignored. This is an efficient
    # way to zero the accumulator.

    d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)
    acc = acc_tmem.load(acc_reg_layout)
    d_smem.store(acc)
    fence_async_shared()
    tma.async_copy_shared_to_global(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)


def small_mma(A, B, C, D, tmem_block, LHS_IN_TMEM, USE_COMMIT, num_warps):
    a_layout = gl.NVMMASharedLayout.get_default_for(A.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(B.shape, gl.float16)
    cd_layout = gl.NVMMASharedLayout.get_default_for(C.shape, gl.float32)

    a_desc = TensorDescriptor.from_tensor(A, A.shape, a_layout)
    b_desc = TensorDescriptor.from_tensor(B, B.shape, b_layout)
    c_desc = TensorDescriptor.from_tensor(C, C.shape, cd_layout)
    d_desc = TensorDescriptor.from_tensor(D, D.shape, cd_layout)

    small_mma_kernel[(1, )](
        a_desc, b_desc, c_desc, d_desc, tmem_block,  #
        LHS_IN_TMEM, USE_COMMIT, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (64, 128, 128), (64, 256, 256), (256, 64, 64)])
@pytest.mark.parametrize("LHS_IN_TMEM", [False, True])
@pytest.mark.parametrize("USE_COMMIT", [False, True])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_small_mma(M, N, K, LHS_IN_TMEM, USE_COMMIT, num_warps):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.randn(M, N, device="cuda", dtype=torch.float32)
    D = torch.empty_like(C)

    blockM = min(128, M)
    blockN = N

    small_mma(A, B, C, D, (blockM, blockN), LHS_IN_TMEM, USE_COMMIT, num_warps)
    torch.testing.assert_close(A @ B + C, D, atol=1e-3, rtol=1e-1)


# %%
# Let's use tcgen05_mma to build a simple blocked matmul kernel. Each program
# will process one block of the accumulator.


@gluon.jit
def blocked_matmul_kernel(a_desc, b_desc, c_desc, TRANSPOSE_B: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    # The block of C this program is processing is (pid_m, pid_n).
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)

    tma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar, count=1)
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_bar, count=1)
    phase = 0

    # Determine the TMEM layout.
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], unpacked=True)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    # We can zero-initialize the accumulator by setting `use_acc=False` on the
    # first iteration.
    use_acc = False
    for k in range(0, K, BLOCK_K):
        mbarrier.expect(tma_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], tma_bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [off_n, k] if TRANSPOSE_B else [k, off_n], tma_bar, b_smem)
        mbarrier.wait(tma_bar, phase=phase)

        # We can transpose B by creating a transposed view over tile of B in
        # shared memory. This forwards the transposition to tcgen05_mma, which
        # handles it for us.
        if TRANSPOSE_B:
            b = b_smem.permute((1, 0))
        else:
            b = b_smem

        # Issue and wait on the tcgen05_mma.
        tcgen05_mma(a_smem, b, acc_tmem, use_acc=use_acc)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase=phase)
        use_acc = True

        phase ^= 1  # toggle the parity phase between 0 and 1

    mbarrier.invalidate(tma_bar)
    mbarrier.invalidate(mma_bar)

    acc_reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, BLOCK_N, [BLOCK_M, BLOCK_N], num_warps)
    acc = acc_tmem.load(acc_reg_layout)

    # Downcast accumulator and store tile of C.
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def blocked_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)

    B_BLOCK_SHAPE = [BLOCK_N, BLOCK_K] if TRANSPOSE_B else [BLOCK_K, BLOCK_N]
    b_layout = gl.NVMMASharedLayout.get_default_for(B_BLOCK_SHAPE, gl.float16)
    b_desc = TensorDescriptor.from_tensor(B, B_BLOCK_SHAPE, b_layout)

    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    blocked_matmul_kernel[grid](a_desc, b_desc, c_desc, TRANSPOSE_B, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 128, 128)])
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_blocked_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn((N, K) if TRANSPOSE_B else (K, N), device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    blocked_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)

    C_ref = A @ (B.T if TRANSPOSE_B else B)
    torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)


# %%
# Let's benchmark our blocked matmul kernel. See the previous tutorial
# 05-wgmma.py for more information on hyperparameter selection.
#
# A few tcgen05_mma specific notes:
#
# - TMEM utilization affects occupancy
# - blockN=128 is typically the optimal instruction shape

if __name__ == "__main__":
    print("Benchmarking selected configs")
    print("=============================")
    M, N, K = 8192, 8192, 16 * 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    print("BLOCK_M BLOCK_N BLOCK_K num_warps time (ms) tflops/s")
    configs = []
    # Picking BLOCK_M != BLOCK_N makes the latency of one load longer than the
    # other. This would be OK if we pipelined them separately, but in our kernel
    # we pipelined them together.
    for BLOCK_MN, BLOCK_K, num_warps in itertools.product([64, 128], [64, 128, 256], [4]):
        if (BLOCK_MN * BLOCK_K) * 4 // 1024 > 224:  # too much SMEM
            continue
        configs.append((BLOCK_MN, BLOCK_K, num_warps))

        fn = lambda: blocked_matmul(A, B, C, BLOCK_MN, BLOCK_MN, BLOCK_K, False, num_warps)
        # Increase warmup and rep to get more stable results.
        ms = triton.testing.do_bench(fn, warmup=100, rep=500)
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"{BLOCK_MN:>7} {BLOCK_MN:>7} {BLOCK_K:>7} {num_warps:>9} {ms:>9.2f} {tflops_per_sec:>8.2f}")
    print()

# %%
# ```
# BLOCK_M BLOCK_N BLOCK_K num_warps time (ms) tflops/s
#      64      64      64         4      3.27   671.77
#      64      64     128         4      3.33   660.93
#      64      64     256         4      4.18   526.10
#     128     128      64         4      2.45   898.61
#     128     128     128         4      2.16  1019.46
#     128     128     256         4      3.91   563.13
# ```
#
# Our first attempt yields 1020 TFLOPS with no pipelining.
#
# Since tcgen05_mma is asynchronous, we can overlap it with the TMA loads to
# reduce SM idle time. Even though the instruction is asynchronous, tcgen05
# instructions are implicitly pipelined, meaning their execution order is
# guaranteed whenever you have:
#
# - two or more tcgen05_mma instructions with the same shape and accumulator dtype
# - a tcgen05_mma followed by tcgen05_commit
# - a tcgen05_cp followed by tcgen05_mma, and vice versa
#
# Thus, we don't need to explicitly synchronize two async MMAs. Combined with
# an mbarrier completion mechanism, it is possible to precisely track MMA
# completion. We can use this to build a fine-grained pipelining schedule.


@gluon.jit
def get_and_increment(counter):
    return counter % 2, counter // 2 & 1, counter + 1


# This pipelined kernel processes two blocks at the same time with software
# pipelining by juggling between them. The kernel partitions along M. The
# kernel expects BLOCK_M = BLOCK_N = 128 and double-buffers all inputs. If
# BLOCK_K is 128, this kernel will use 192 KB of SMEM.
#
# The schedule the kernel uses is:
#
#     U1, B1, V1,
#     U2, B2, V2,
#     UB1, U3, VB1, B3, V3, ..., UB(N-2), UN, VB(N-2), BN, VN
#     UB(N-1), VB(N-1)
#     UBN, VBN,
#     UB epilogue, VB epilogue
#
# This yields a 3:2 ratio of loads to MMAs. We can use the same mbarrier to
# track U and B loads.
@gluon.jit
def blocked_matmul_pipelined_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * (2 * BLOCK_M)
    off_n = pid_n * BLOCK_N

    # u := upper tile, v := lower tile
    u_bufs = gl.allocate_shared_memory(dtype, [2] + a_desc.block_type.shape, a_desc.layout)
    v_bufs = gl.allocate_shared_memory(dtype, [2] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [2] + b_desc.block_type.shape, b_desc.layout)

    # Use two accumulators!
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], unpacked=True)
    ub_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    vb_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    mma_ub_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mma_vb_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    load_ub_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    load_v_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        mbarrier.init(mma_ub_bars.index(i), count=1)
        mbarrier.init(mma_vb_bars.index(i), count=1)
        mbarrier.init(load_ub_bars.index(i), count=1)
        mbarrier.init(load_v_bars.index(i), count=1)

    load_counter = 0
    mma_counter = 0
    k = 0
    ub_acc = False
    vb_acc = False

    # U1, B1
    load_index, load_phase, load_counter = get_and_increment(load_counter)
    load_ub_bar = load_ub_bars.index(load_index)
    mbarrier.expect(load_ub_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))
    tma.async_copy_global_to_shared(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))
    # V1
    load_v_bar = load_v_bars.index(load_index)
    mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m + BLOCK_M, k], load_v_bar, v_bufs.index(load_index))
    k += BLOCK_K

    # U2, B2
    load_index, load_phase, load_counter = get_and_increment(load_counter)
    load_ub_bar = load_ub_bars.index(load_index)
    mbarrier.expect(load_ub_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))
    tma.async_copy_global_to_shared(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))
    # V2
    load_v_bar = load_v_bars.index(load_index)
    mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m + BLOCK_M, k], load_v_bar, v_bufs.index(load_index))
    k += BLOCK_K

    for _ in range(gl.cdiv(K, BLOCK_K) - 2):
        # wait Ui and Bi, UBi
        mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
        mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
        tcgen05_mma(u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=ub_acc)
        tcgen05_commit(mma_ub_bars.index(mma_index))
        ub_acc = True
        # wait Vi, VBi
        mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
        tcgen05_mma(v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=vb_acc)
        tcgen05_commit(mma_vb_bars.index(mma_index))
        vb_acc = True

        # wait UBi, U(i+2)
        load_index, load_phase, load_counter = get_and_increment(load_counter)
        mbarrier.wait(mma_ub_bars.index(mma_index), mma_phase)
        load_ub_bar = load_ub_bars.index(load_index)
        mbarrier.expect(load_ub_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))

        # wait VBi, B(i+2), V(i+2)
        mbarrier.wait(mma_vb_bars.index(mma_index), mma_phase)
        tma.async_copy_global_to_shared(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))
        load_v_bar = load_v_bars.index(load_index)
        mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m + BLOCK_M, k], load_v_bar, v_bufs.index(load_index))
        k += BLOCK_K

    acc_reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, BLOCK_N, [BLOCK_M, BLOCK_N], num_warps)

    mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
    ub_bar = mma_ub_bars.index(mma_index)
    vb_bar = mma_vb_bars.index(mma_index)
    epilogue_phase = mma_phase

    # wait U(N-1) and B(N-1), UB(N-1)
    mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
    tcgen05_mma(u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=True)
    # wait V(N-1), VB(N-1)
    mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
    tcgen05_mma(v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=True)

    # Wait UN and BN, UBN
    mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
    mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
    tcgen05_mma(u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=True)
    tcgen05_commit(ub_bar)
    # Wait VN and VBN
    mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
    tcgen05_mma(v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=True)
    tcgen05_commit(vb_bar)

    # Wait UBN, UB epilogue
    mbarrier.wait(ub_bar, epilogue_phase)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    ub = ub_tmem.load(acc_reg_layout)
    c_smem.store(ub.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)

    # Wait VBN, VB epilogue
    mbarrier.wait(vb_bar, epilogue_phase)
    vb = vb_tmem.load(acc_reg_layout)
    tma.store_wait(pendings=0)
    c_smem.store(vb.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m + BLOCK_M, off_n], c_smem)
    tma.store_wait(pendings=0)


def blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, 2 * BLOCK_M), triton.cdiv(N, BLOCK_N))
    blocked_matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 128, 128)])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_blocked_matmul_pipelined(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):

    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)


if __name__ == "__main__":
    print("Benchmarking pipelined matmul")
    print("=============================")
    print("BLOCK_M BLOCK_N BLOCK_K num_warps time (ms) tflops/s")
    # Since the kernel was designed with specific hyperparameters in mind, we
    # will only benchmark those.
    for BLOCK_M, BLOCK_N, BLOCK_K, num_warps in itertools.product([128], [128], [64, 128], [4, 8]):
        fn = lambda: blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
        ms = triton.testing.do_bench(fn, warmup=200, rep=1000)
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"{BLOCK_M:>7} {BLOCK_N:>7} {BLOCK_K:>7} {num_warps:>9} {ms:>9.2f} {tflops_per_sec:>8.2f}")
    print()

# %%
# ```
# BLOCK_M BLOCK_N BLOCK_K num_warps time (ms) tflops/s
# 128     128      64         4      2.20  1000.51
# 128     128      64         8      1.97  1113.49
# 128     128     128         4      2.21  1040.27
# 128     128     128         8      2.17  1011.47
# ```
#
# Although we deliver a modest speedup on the same hyperparameters from the
# non-pipelined kernel, it turns out that BLOCK_K=64 yields much better
# performance. When BLOCK_K=64 we get 2x occupancy, suggesting that the pipeline
# schedule can be improved.
#
# Interestingly, num_warps=8 matters significantly for BLOCK_K=64, and this is
# likely due to the longer epilogue. After we introduce warp specialization, we
# will see that it can be a much more efficient way to finely pipeline a kernel.
