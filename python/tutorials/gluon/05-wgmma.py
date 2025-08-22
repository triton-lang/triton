"""
Warp-Group MMA
==============

Warp-Group MMA (also known as WGMMA or MMAv3) is a Hopper-specific instruction
for performing matrix multiply-accumulate operations using the Tensor Cores.
WGMMA instructions are asynchronous, meaning they can be pipelined.

In this tutorial, we will cover how to use WGMMAs in Gluon. We will build a
simple matmul kernel to demonstrate practical uses of WGMMA, and show an example
where WGMMAs can be pipelined for better performance.
"""

import pytest
import torch
import triton
import itertools
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma_init,
    warpgroup_mma,
    warpgroup_mma_wait,
)


def is_hopper():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 9


if __name__ == "__main__" and not is_hopper():
    raise RuntimeError("This tutorial requires a Hopper NVIDIA GPU")

# %%
# Let's illustrate WGMMA with a trivial kernel launched with grid size (1, ).
# This kernel performs MMA on a small tensor.
#
# warpgroup_mma performs d = a * b + c. The `a` operand can be passed as
# registers or through shared memory. The `b` operand must be passed through
# shared memory, and the `c` operand must be passed through registers.
#
# warpgroup_mma itself is composed of many smaller `wgmma.mma_async` PTX
# instructions, which supports a limited set of instruction shapes.
#
# The instruction shape is specified as [m, n, k], where
#
# - `k` is always 256 / A.dtype.primitive_bitwidth
# - `m` is always 16
# - `n` can be can chosen as follows:
#
# For floating point dtypes, `n` must be a positive multiple of 8, up to and
# including 256. WGMMA supports 8-bit integers, but `n` must be chosen from:
#
#   224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 24, 16, 8
#
# `n` must be chosen such that it evenly divides into `BLOCK_N`, the inner
# dimension of the MMA tile, and it must be less than or equal to `maxN`, where
# `maxN` is computed as:
#
#     mReps = ceildiv(M, m)
#     nReps = ceildiv(num_warps, mReps)
#     maxN = max(N // nReps, 8)
#
# warpgroup_mma divides the MMA across warps using `warps_per_cta`, in the
# same way `BlockedLayout.warps_per_cta` tiles a tensor across warps. The
# smallest indivisible unit of `warps_per_cta` is `[4, 1]`. Note that this
# means WGMMA requires at least 4 warps, which together make up one warp group.
# To choose the right `warps_per_cta`, start from the atom `[4, 1]` and simply
# double it along any dimension until it matches the number of warps. Note that
# since `m=16` and must be at least 4 wraps along M, the M dimension must be at
# least 64.
#
# Note when `num_warps=8`, we can choose `[4, 2]` or `[8, 1]`, but recall from
# 02-layouts that this can affect the performance of, e.g., reductions.
#
# warpgroup_mma is an asynchronous operation whose completion is tracked by
# commit groups, like async copies and TMA stores. Issuing a WGMMA operation
# implicitly commits it to a WGMMA group, and we can wait until there are N
# outstanding operations.
#
# Because warpgroup_mma is an asynchronous, until the operation is complete,
# we cannot access the result even though it is in registers, and we cannot
# write to any of the shared memory inputs. WGMMA accesses shared memory through
# the async proxy. Since TMAs also access shared memory through the async proxy,
# we don't need fences between TMA and WGMMA instructions.
#
# ```python
# b_smem.store(b)
# fence_async_shared()
# warpgroup_mma(a, b_smem, c, is_async=True)
# ```
#
# A fence is needed between the shared store and warpgroup_mma to order their
# shared memory accesses.
#
# Completion of the WGMMA implies its reads from shared memory are complete.
# Thus, it is safe to write to the shared memory inputs after waiting:
#
# ```python
# d = warpgroup_mma(a, b_smem, c, is_async=True)
# d = warpgroup_mma_wait(num_outstanding=0, deps=(d, ))
# b_smem.store(b)
# ```
#
# If the LHS operand is supplied in registers via a shared load, completion of
# the WGMMA implies the shared load is complete, and subsequent accesses to the
# buffer via the async proxy do not require a fence:
#
# ```python
# a = a_smem.load(dot_operand_layout)
# d = warpgroup_mma(a, b_smem, c, is_async=True)
# d = warpgroup_mma_wait(num_outstanding=0, deps=(d, ))
# tma.async_copy_global_to_shared(a_desc, [0, 0], bar, a_smem)
# ```

# %%
# Let's implement a simple matmul kernel that uses WGMMA.


@gluon.jit
def small_mma_kernel(a_desc, b_desc, c_desc, d_desc,  #
                     LHS_IN_REG: gl.constexpr, INSTR_SHAPE_N: gl.constexpr, num_warps: gl.constexpr):
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
    mbarrier.invalidate(bar)

    # Let's parameterize the kernel over LHS_IN_REG and INSTR_SHAPE_N to see how
    # it can affect performance.
    m: gl.constexpr = 16
    k: gl.constexpr = 256 // a_desc.dtype.primitive_bitwidth
    n: gl.constexpr = INSTR_SHAPE_N
    warps_per_cta: gl.constexpr = [num_warps, 1]

    # The MMA shape is passed through the layout of `c`, which must always have
    # an NVMMADistributedLayout.
    c_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )

    # When A is passed through registers, it must have the following layout:
    a_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=c_layout,
        k_width=32 // a_desc.dtype.primitive_bitwidth,
    )

    # When an operand is passed through shared memory, it must have an
    # NVMMASharedLayout. TMA requires using an NVMMASharedLayout.
    gl.static_assert(isinstance(a_smem.type.layout, gl.NVMMASharedLayout))
    gl.static_assert(isinstance(b_smem.type.layout, gl.NVMMASharedLayout))

    if LHS_IN_REG:
        a = a_smem.load(a_reg_layout)
    else:
        a = a_smem

    c = c_smem.load(c_layout)
    # Issue the async WGMMA. Note that `is_async=False` is the default value,
    # and all this does is immediately wait for 0 outstanding operations. In
    # this tutorial, we will always use `is_async=True`.
    #
    # Another important flag to consider is `use_acc`. When `use_acc=False`, the
    # `c` input is ignored and the accumulator is zero-initialized. This can be
    # an efficient way to zero the accumulator.
    d = warpgroup_mma(a, b_smem, c, is_async=True, use_acc=True)

    # To ensure correct ordering between `warpgroup_mma`, the wait, and uses of
    # the result, you must thread the `warpgroup_mma` result through the wait
    # via the `deps` argument and use the return value of the
    # `warpgroup_mma_wait`.
    #
    # Wait for 0 outstanding operations, so we know the WGMMA is complete.
    d = warpgroup_mma_wait(num_outstanding=0, deps=(d, ))

    d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)
    d_smem.store(d)
    fence_async_shared()
    tma.async_copy_shared_to_global(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)


def small_mma(A, B, C, D, INSTR_SHAPE_N, LHS_IN_REG=False, num_warps=4):
    a_layout = gl.NVMMASharedLayout.get_default_for(A.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(B.shape, gl.float16)
    cd_layout = gl.NVMMASharedLayout.get_default_for(C.shape, gl.float32)

    a_desc = TensorDescriptor.from_tensor(A, A.shape, a_layout)
    b_desc = TensorDescriptor.from_tensor(B, B.shape, b_layout)
    c_desc = TensorDescriptor.from_tensor(C, C.shape, cd_layout)
    d_desc = TensorDescriptor.from_tensor(D, D.shape, cd_layout)

    small_mma_kernel[(1, )](
        a_desc, b_desc, c_desc, d_desc,  #
        LHS_IN_REG, INSTR_SHAPE_N, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(64, 32, 32), (64, 256, 128)])
@pytest.mark.parametrize("LHS_IN_REG", [False, True])
@pytest.mark.parametrize("INSTR_SHAPE_N", [16, 64])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
def test_small_mma(M, N, K, LHS_IN_REG, INSTR_SHAPE_N, num_warps):
    maxN = max(N // triton.cdiv(num_warps, triton.cdiv(M, 16)), 8)
    if INSTR_SHAPE_N > maxN:
        pytest.skip(f"INSTR_SHAPE_N={INSTR_SHAPE_N} is too large for M={M}, N={N}, num_warps={num_warps}")

    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.randn(M, N, device="cuda", dtype=torch.float32)
    D = torch.empty_like(C)
    small_mma(A, B, C, D, INSTR_SHAPE_N, LHS_IN_REG, num_warps)
    torch.testing.assert_close(A @ B + C, D, atol=1e-3, rtol=1e-1)


# %%
# Let's study the performance impact of our knobs on WGMMA.

if __name__ == "__main__":
    print("Benchmarking WGMMA")
    print("==================")
    M, N, K = 64, 128, 128
    num_warps = 4
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.randn(M, N, device="cuda", dtype=torch.float32)
    D = torch.empty_like(C)

    print("LHS_IN_REG INSTR_SHAPE_N time (us)")
    for LHS_IN_REG, INSTR_SHAPE_N in itertools.product([False, True], [16, 32, 64, 128]):
        fn = lambda: small_mma(A, B, C, D, INSTR_SHAPE_N, LHS_IN_REG, num_warps)
        ms = triton.testing.do_bench(fn)
        print(f"{LHS_IN_REG!s:>10} {INSTR_SHAPE_N:>13} {ms*1000:>9.2f}")
    print()

# %%
# ```
# LHS_IN_REG INSTR_SHAPE_N time (us)
#      False            16      9.47
#      False            32      8.48
#      False            64      8.32
#      False           128      8.32
#       True            16      9.32
#       True            32      8.60
#       True            64      8.37
#       True           128      8.36
# ```
#
# Picking the largest N results in the best performance, because each
# `wgmma.mma_async` instruction will process more data. In our case, placing LHS
# in registers is slower because we had to load the data out of shared memory.
# However, if the data was already in registers, it would be faster to use it in
# registers instead of placing it in shared memory.

# %%
# Just like `warpgroup_mma` is composed of multiple `wgmma.mma_async`
# instructions tiled to cover our block size, we can also tile `warpgroup_mma`
# to cover a much larger matmul. We can tile along K within each kernel and span
# (M, N) with multiple programs. This leads to the classic blocked matmul
# implementation. Let's implement a basic version to demonstrate WGMMA.


# This decorator allows us to invoke the function from a Gluon constexpr.
@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    warps_per_cta = [4, 1]
    m = 16
    # Tile the atom until we have enough warps.
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        # Tile along M only if it would not cause broadcasting.
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


@gluon.jit
def blocked_matmul_kernel(a_desc, b_desc, c_desc,  #
                          TRANSPOSE_B: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)

    # The block of C this program is processing is (pid_m, pid_n).
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Determine the WGMMA layout.
    mma_layout: gl.constexpr = pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    for k in range(0, K, BLOCK_K):
        # Load tiles of A and B.
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_smem)
        if TRANSPOSE_B:
            tma.async_copy_global_to_shared(b_desc, [off_n, k], bar, b_smem)
        else:
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_smem)
        mbarrier.wait(bar, phase=phase)
        phase ^= 1  # toggle the parity phase between 0 and 1

        # We can transpose B by creating a transposed view over tile of B in
        # shared memory. This forwards the transposition to WGMMA, which handles
        # it for us.
        if TRANSPOSE_B:
            b = b_smem.permute((1, 0))
        else:
            b = b_smem

        acc = warpgroup_mma(a_smem, b, acc, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc, ))

    mbarrier.invalidate(bar)

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
@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
def test_blocked_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn((N, K) if TRANSPOSE_B else (K, N), device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    blocked_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)

    C_ref = A @ (B.T if TRANSPOSE_B else B)
    torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)


# %%
# We can benchmark this kernel as a baseline, but we need to pick the best block
# sizes. Rather than autotuning over all possibilities, we can apply some
# principles to narrow down the search space.
#
# We should try to pick the largest `n` for the WGMMA layout. Based on the
# formula for `maxN` this requires `BLOCK_N>=256`. Because our kernel does not
# overlap the TMA loads with WGMMA, we will want more than program resident on
# each SM so that when one kernel stalls, the SM can switch to the other. This
# is known as "occupancy". In detail, each SM has limited resources, and the
# resource usage of a kernel determines its max occupancy. The SM schedules work
# by warp using its warp scheduler, which can efficiently swap executing warps,
# almost like hyperthreading.
#
# Based on register and smem constraints, we can filter configs for the desired
# occupancy. Keep in mind that these are rules of thumb. It's hard to know for
# sure if these lead to the best block sizes.


def find_configs(occupancy, dtype, num_buffers=1):
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()

    # Assume ~1 KB of smem used by mbarriers, compiler-generated code, etc.
    smem = 228 * 1024 // occupancy - 1024

    configs = []
    BLOCK_MNK = [32, 64, 128, 256]
    for BLOCK_M, BLOCK_N, BLOCK_K, num_warps in itertools.product(BLOCK_MNK, BLOCK_MNK, BLOCK_MNK, [4, 8]):
        # Assume ~16 regs per thread of baseline usage.
        regs = 64 * 1024 // occupancy - 16 * num_warps * 32

        a_smem = BLOCK_M * BLOCK_K * dtype_bytes
        b_smem = BLOCK_N * BLOCK_K * dtype_bytes
        acc_smem = BLOCK_M * BLOCK_N * dtype_bytes
        # SMEM for A and B does not coexist with C.
        if max((a_smem + b_smem) * num_buffers, acc_smem) > smem:
            continue

        # The accumulator is the only in-memory tensor in f32.
        acc_regs = BLOCK_M * BLOCK_N
        # Max regs per thread is 256. Being near this can also cause spills.
        if acc_regs // num_warps // 32 >= 256:
            continue
        if acc_regs > regs:
            continue

        instr_shape_n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps).value
        configs.append((BLOCK_M, BLOCK_N, BLOCK_K, num_warps, instr_shape_n, occupancy))

    def filter_configs(configs, instr_shape_n):
        max_n_configs = [cfg for cfg in configs if cfg[4] == instr_shape_n]
        # Filter for configs with the largest BLOCK_M * BLOCK_K.
        max_block_mk = max(cfg[0] * cfg[2] for cfg in max_n_configs)
        return [cfg for cfg in max_n_configs if cfg[0] * cfg[2] == max_block_mk]

    top_instr_shape_n = sorted({cfg[4] for cfg in configs}, reverse=True)
    result_configs = filter_configs(configs, top_instr_shape_n[0])
    if len(top_instr_shape_n) > 1:
        result_configs += filter_configs(configs, top_instr_shape_n[1])
    return result_configs


if __name__ == "__main__":
    print("Benchmarking selected configs")
    print("=============================")
    # Just in case, check occupancy 1 configs.
    configs = find_configs(occupancy=1, dtype=torch.float16)
    configs += find_configs(occupancy=2, dtype=torch.float16)
    # Benchmark the configs over a large matmul. Keep in mind that the best
    # hyperparameters can depend on the matmul shapes.
    M, N, K = 8192, 8192, 16 * 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    print("BLOCK_M BLOCK_N BLOCK_K num_warps instr_shape_n occupancy time (ms) tflops/s")
    for BLOCK_M, BLOCK_N, BLOCK_K, num_warps, instr_shape_n, occupancy in configs:
        fn = lambda: blocked_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, False, num_warps)
        ms = triton.testing.do_bench(fn)
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"{BLOCK_M:>7} {BLOCK_N:>7} {BLOCK_K:>7} {num_warps:>9} {instr_shape_n:>13} "
              f"{occupancy:>9} {ms:>9.2f} {tflops_per_sec:>8.2f}")
    print()

# %%
# ```
# BLOCK_M BLOCK_N BLOCK_K num_warps instr_shape_n occupancy time (ms) tflops/s
#     128     256     256         8           256         1      5.34   412.14
#     256     128     256         8           128         1      5.67   387.74
#      64     256     128         4           256         2      4.64   474.03
#      64     128     256         4           128         2      6.18   355.60
#     128     128     128         4           128         2      4.98   441.88
#     128     128     128         8           128         2      5.79   380.08
# ```
#
# The hypothesis that having occupancy 2 with `BLOCK_N=256` would be the best
# has held over our limited sample of hyperparameters. Autotuning over all
# hyperparameters is an exercise for the reader.

# %%
# 466 TFLOPS is not a bad start. However, we aren't using the fact that WGMMA is
# asynchronous, and we aren't pipelining the TMA loads as shown in previous
# tutorials.
#
# For now, let's keep the loads synchronous and focus on pipelining the WGMMA.
# This requires us to double-buffer the operands, since we will be loading into
# the next set of buffers while WGMMA reads from the previous.


@gluon.jit
def blocked_matmul_pipelined_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    # Allocate 2 buffers for each A and B.
    a_smem = gl.allocate_shared_memory(dtype, [2] + a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, [2] + b_desc.block_type.shape, b_desc.layout)
    index = 0

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    mma_layout: gl.constexpr = pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    acc = warpgroup_mma_init(gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout))

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    for k in range(0, K, BLOCK_K):
        a = a_smem.index(index)
        b = b_smem.index(index)

        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a)
        tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b)
        mbarrier.wait(bar, phase=phase)
        phase ^= 1

        # Since `warpgroup_mma_wait` is a no-op when there are no WGMMAs in
        # flight, we can overlap the WGMMA by waiting first, then issuing the
        # async WGMMA.
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc, ))
        acc = warpgroup_mma(a, b, acc, is_async=True)

        # Move to the next buffer. The TMA load will start while the WGMMA is
        # still running.
        index ^= 1

    # Wait for the last WGMMA to complete.
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc, ))

    mbarrier.invalidate(bar)

    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    blocked_matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 128, 128)])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
def test_blocked_matmul_pipelined(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):

    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)


# %%
# Search for another set of configs. Apply simiar principles to prune down the
# potential configs. Our previous best block config will use 160 KB of smem, too
# much for an occupancy of 2, but leaves performance on the table by not using
# the remaining 68 KB. It's likely the best kernel reduces BLOCK_N in favour of
# keeping 2 occupancy.

if __name__ == "__main__":
    print("Benchmarking pipelined matmul")
    print("=============================")
    configs = find_configs(occupancy=1, dtype=torch.float16, num_buffers=2)
    configs += find_configs(occupancy=2, dtype=torch.float16, num_buffers=2)
    # Add our previous best config since it doesn't get selected.
    configs.append([64, 256, 128, 4, 256, 2])

    print("BLOCK_M BLOCK_N BLOCK_K num_warps instr_shape_n occupancy time (ms) tflops/s")
    for BLOCK_M, BLOCK_N, BLOCK_K, num_warps, instr_shape_n, occupancy in configs:
        fn = lambda: blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
        ms = triton.testing.do_bench(fn)
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"{BLOCK_M:>7} {BLOCK_N:>7} {BLOCK_K:>7} {num_warps:>9} {instr_shape_n:>13} "
              f"{occupancy:>9} {ms:>9.2f} {tflops_per_sec:>8.2f}")
    print()

# %%
# ```
# BLOCK_M BLOCK_N BLOCK_K num_warps instr_shape_n occupancy time (ms) tflops/s
#     128     256     128         8           256         1      5.16   426.06
#     256     128     128         8           128         1      5.70   385.85
#      64     256      64         4           256         2      5.27   417.50
#      64     128     128         4           128         2      5.71   384.98
#     128     128      64         4           128         2      4.44   495.31
#     128     128      64         8           128         2      4.92   446.81
#      64     256     128         4           256         2      6.05   363.36
# ```
#
# We see indeed that the best config ends up with instr_shape_n=128. Note that
# our previous best config is over 100 TFLOPS slower now! Pipelining the WGMMA
# delivers a modest 5% speedup overall, but we had to re-tune the
# hyperparameters.
#
# Pipelining both the async TMA loads and the WGMMA is left as an exercise to
# the reader.
#
# Main takeaways:
#
# - WGMMA is a Hopper-specific instruction that performs block-level MMA.
# - WGMMA is asynchronous and can be overlapped with other operations.
# - WGMMA has a bunch of restrictions on its layout.
# - LHS operand can be in shared memory or registers.
# - WGMMA can handle transposed inputs, and we can create transposed views.
# - Pipelining the WGMMA leads to better performance by enabling overlap.
# - Hyperparameter tuning is critical for performance.
