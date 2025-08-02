"""
Warp-Group MMA
==============

WGMMA (also known as MMAv3) is a Hopper-specific instruction for performing
matrix multiply-accumulate operations using the Tensor Cores. WGMMA instructions
are asynchronous, meaning they can be pipelined.

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
    warpgroup_mma,
    warpgroup_mma_wait,
)


def is_hopper():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


if __name__ == "__main__" and not is_hopper():
    raise RuntimeError("This tutorial requires a Hopper NVIDIA GPU")

# %%
# Let's illustrate WGMMA with a trivial kernel launched with grid size (1, ).
# This kernel performs MMA on a small tensor.


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

    # `warpgroup_mma` performs d = a * b + c. The `a` operand can be passed as
    # registers or through shared memory. The `b` operand must be passed through
    # shared memory, and the `c` operand must be passed through registers.
    #
    # `warpgroup_mma` itself is composed of many smaller `wgmma.mma_async` PTX
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
    # `n` must be chosen such that it evenly divides into `N`, and it must be
    # less than or equal to `maxN`, where `maxN` is computed as:
    #
    #     mReps = ceildiv(M, m)
    #     nReps = ceildiv(num_warps, mReps)
    #     maxN = max(N // nReps, 8)
    #
    # `warpgroup_mma` divides the MMA across warps using `warps_per_cta`, in the
    # same way `BlockedLayout.warps_per_cta` tiles a tensor across warps. The
    # smallest indivisible unit of `warps_per_cta` is `[4, 1]`. Note that this
    # means WGMMA requires at least 4 warps. To choose the right
    # `warps_per_cta`, start from the atom `[4, 1]` and simply double it along
    # any dimension until it matches the number of warps. Note that since `m=16`
    # and must be at least 4 wraps along M, the M dimension must be at least 64.
    #
    # Note when `num_warps=8`, we can choose `[4, 2]` or `[8, 1]`, but recall
    # from 02-layouts that this can affect the performance of, e.g., reductions.

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
    d = warpgroup_mma(a, b_smem, c, is_async=True)

    # WGMMA is an asynchronous operation. Until the operation is complete, we
    # cannot access the result, even though it is in registers, and we cannot
    # write to any of the shared memory inputs. To ensure a correct ordering
    # between `warpgroup_mma`, the wait, and uses of the result, we must pass
    # the result through the wait as one of its `deps` arguments.
    #
    # WGMMA accesses shared memory through the async proxy, like TMAs. This
    # means `fence_async_shared` is sometimes required to prevent hazards.
    #
    # The completion of WGMMA operations is tracked by commit groups, like
    # async copies and TMA stores. Issuing a WGMMA operation implicitly commits
    # it to a WGMMA group. Thus, we can wait for the completion of the operation
    # by waiting until there are 0 outstanding operations.
    d = warpgroup_mma_wait(num_outstanding=0, deps=(d, ))

    # Note that `is_async=False` is the default value, and all this does is
    # immediately wait for 0 outstanding operations. In this tutorial, we will
    # always use `is_async=True`.
    #
    # Another important flag to consider is `use_acc`. When `use_acc=False`, the
    # `c` input is ignored and the accumulator is zero-initialized. This can be
    # an efficient way to zero the accumulator.

    d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)
    d_smem.store(d)
    fence_async_shared()
    tma.async_copy_shared_to_global(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)


def small_mma(A, B, C, D, INSTR_SHAPE_N, LHS_IN_REG=False, num_warps=4):
    a_layout = gl.NVMMASharedLayout.get_default_for(A.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(B.shape, gl.float16)
    cd_layout = gl.NVMMASharedLayout.get_default_for(C.shape, gl.float32)

    a_desc = TensorDescriptor(A, A.shape, A.stride(), A.shape, a_layout)
    b_desc = TensorDescriptor(B, B.shape, B.stride(), B.shape, b_layout)
    c_desc = TensorDescriptor(C, C.shape, C.stride(), C.shape, cd_layout)
    d_desc = TensorDescriptor(D, D.shape, D.stride(), D.shape, cd_layout)

    small_mma_kernel[(1, )](
        a_desc, b_desc, c_desc, d_desc,  #
        LHS_IN_REG, INSTR_SHAPE_N, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(64, 32, 32), (64, 256, 128)])
@pytest.mark.parametrize("LHS_IN_REG", [False, True])
@pytest.mark.parametrize("INSTR_SHAPE_N", [16, 64])
@pytest.mark.parametrize("num_warps", [4, 8])
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
@gl.constexpr_function
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


@gl.constexpr_function
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


@gl.constexpr_function
def int_log2(x):
    return x.bit_length() - 1


@gluon.jit
def blocked_mma_kernel(a_desc, b_desc, c_desc,  #
                       TRANSPOSE_B: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    K = a_desc.shape[1]

    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)

    # The block of C this program is processing is (pid_m, pid_n).
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Determine the WGMMA layout.
    m: gl.constexpr = 16
    k: gl.constexpr = 256 // a_desc.dtype.primitive_bitwidth
    n: gl.constexpr = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
    warps_per_cta: gl.constexpr = get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=c_desc.dtype, layout=mma_layout)

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

    # Store tile of C.
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc)
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def blocked_mma(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K], a_layout)

    B_BLOCK_SHAPE = [BLOCK_N, BLOCK_K] if TRANSPOSE_B else [BLOCK_K, BLOCK_N]
    b_layout = gl.NVMMASharedLayout.get_default_for(B_BLOCK_SHAPE, gl.float16)
    b_desc = TensorDescriptor(B, B.shape, B.stride(), B_BLOCK_SHAPE, b_layout)

    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float32)
    c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    blocked_mma_kernel[grid](a_desc, b_desc, c_desc, TRANSPOSE_B, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 128, 128)])
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("num_warps", [4, 8])
def test_blocked_mma(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn((N, K) if TRANSPOSE_B else (K, N), device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)

    blocked_mma(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)

    C_ref = A @ (B.T if TRANSPOSE_B else B)
    torch.testing.assert_close(C_ref.to(torch.float32), C, rtol=1e-3, atol=1e-1)


# %%
# We can benchmark this kernel as a baseline, but we need to pick the best block
# sizes. Rather than autotuning over all possibilities, we can apply some
# principles to narrow down the search space.
#
# We should try to pick the largest `n` for the WGMMA layout. Based on the
# formula for `maxN` this requires `BLOCK_N>=256`. Because our kernel does not
# overlap the TMA loads with WGMMA, we will need at least 2 occupancy so SMs
# have work to switch to while waiting.
#
# Based on register and smem constraints, we can filter configs for the desired
# occupancy. Keep in mind that these are rules of thumb. It's hard to know for
# sure if these lead to the best block sizes.


def find_configs(occupancy, in_dtype, acc_dtype):
    in_dtype_bytes = torch.tensor([], dtype=in_dtype).element_size()
    acc_dtype_bytes = torch.tensor([], dtype=acc_dtype).element_size()

    # Assume ~1 KB of smem used by mbarriers, compiler-generated code, etc.
    smem = 228 * 1024 // occupancy - 1024

    configs = []
    for BLOCK_M, BLOCK_N, BLOCK_K, num_warps in itertools.product([64, 128, 256], [64, 128, 256], [64, 128, 256],
                                                                  [4, 8]):
        # Assume ~16 regs per thread of baseline usage.
        regs = 64 * 1024 // occupancy - 16 * num_warps * 32

        a_smem = BLOCK_M * BLOCK_K * in_dtype_bytes
        b_smem = BLOCK_N * BLOCK_K * in_dtype_bytes
        acc_smem = BLOCK_M * BLOCK_N * acc_dtype_bytes
        # SMEM for A and B does not coexist with C.
        if max(a_smem + b_smem, acc_smem) > smem:
            continue

        # The accumulator is the only in-memory tensor.
        acc_regs = BLOCK_M * BLOCK_N * acc_dtype_bytes // 4
        # Max regs per thread is 256. Being near this can also cause spills.
        if acc_regs // num_warps // 32 >= 256:
            continue
        if acc_regs > regs:
            continue

        instr_shape_n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps).value
        configs.append((BLOCK_M, BLOCK_N, BLOCK_K, num_warps, instr_shape_n))

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
    print("Finding possible configs")
    print("========================")
    # Just in case, check occupancy 1 configs.
    configs = find_configs(occupancy=1, in_dtype=torch.float16, acc_dtype=torch.float32)
    configs += find_configs(occupancy=2, in_dtype=torch.float16, acc_dtype=torch.float32)
    # Benchmark the configs over a large matmul. Keep in mind that the best
    # hyperparameters can depend on the matmul shapes.
    M, N, K = 8192, 8192, 16 * 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    print("BLOCK_M BLOCK_N BLOCK_K num_warps instr_shape_n time (ms) tflops/s")
    for BLOCK_M, BLOCK_N, BLOCK_K, num_warps, instr_shape_n in configs:
        fn = lambda: blocked_mma(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, False, num_warps)
        ms = triton.testing.do_bench(fn)
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"{BLOCK_M:>7} {BLOCK_N:>7} {BLOCK_K:>7} {num_warps:>9} {instr_shape_n:>13} "
              f"{ms:>9.2f} {tflops_per_sec:>8.2f}")

# %%
# ```
# BLOCK_M BLOCK_N BLOCK_K num_warps instr_shape_n time (ms) tflops/s
#     128     256     256         8           256      5.36   410.35
#     256     128     256         8           128      5.77   381.02
#      64     256     128         4           256      4.72   466.33
#      64     128     256         4           128      6.34   346.87
#     128     128     128         4           128      4.97   442.38
#     128     128     128         8           128      5.76   381.51
# ```
#
# Key takeaways from this example:
#
# - Inputs can be transposed by creating permuted views over shared memory.
# -

# # Epilogue subtiling is a technique to reduce shared memory usage by
# # splitting up the TMA store of the accumulator. We subtile by
# # EPILOGUE_SUBTILE_FACTOR along N. However, there are restrictions to
# # subtiling:
# #
# # - Swizzled shared memory can only be split down to the swizzle tile.
# #   This depends on the shared memory layout.
# # - We can only split along the contiguous dimension, otherwise we can't
# #   store the tiles correctly.
# # - There must be at least EPILOGUE_SUBTILE_FACTOR contiguous elements
# #   per thread along N in the register layout.
# contig_dim_size: gl.constexpr = c_desc.type.layout.swizzle_byte_width * 8 / c_desc.dtype.primitive_bitwidth
# TILE_SIZE_N: gl.constexpr = BLOCK_N // EPILOGUE_SUBTILE_FACTOR
# gl.static_assert(TILE_SIZE_N >= contig_dim_size, "C descriptor layout cannot be subtiled this much")

# # Split the accumulator into subtiles.
# cs = (acc, )
# for i in gl.static_range(int_log2(EPILOGUE_SUBTILE_FACTOR)):
#     next_cs = ()
#     for j in gl.static_range(len(cs)):
#         c = cs[j]
#         next_cs += cs[j].reshape([c.shape[0], 2, c.shape[1] // 2]).permute(0, 2, 1).split()
#     cs = next_cs

# # Store each subtile sequentially.
# c_subtile_smem = gl.allocate_shared_memory(c_desc.dtype, [BLOCK_M, TILE_SIZE_N], c_desc.layout)
# for i in gl.static_range(len(cs)):
#     c_subtile_smem.store(cs[i])
#     fence_async_shared()
#     tma.async_copy_shared_to_global(c_desc, [off_m, off_n + i * TILE_SIZE_N], c_subtile_smem)
#     tma.store_wait(pendings=0)
