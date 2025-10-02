"""
Intra-Kernel Profiling Examples using Proton DSL for Triton and Gluon Kernels
"""

import argparse

import torch
import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()

NUM_WARPS = 8


def is_hopper():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 9


def config_helper(description: str):
    # Configure command line arguments for profiling options
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--op-measure",
        action="store_true",
        default=False,
        help="Enable operation measurement. Otherwise, we profile timeline trace. (default: False)",
    )
    parser.add_argument(
        "--warp-sampling",
        action="store_true",
        default=False,
        help="Enable warp sampling during profiling (default: False)",
    )
    parser.add_argument(
        "--increase-accuracy",
        action="store_true",
        default=False,
        help="Enable increased-accuracy during profiling (default: False).",
    )
    parser.add_argument(
        "--warp-ids",
        type=str,
        default="0, 2",
        help="Comma-separated list of warp IDs for warp sampling (default: '0, 2')",
    )

    args = parser.parse_args()

    # Configure profiling options based on accuracy requirements
    # Default uses clock_64 for long-running kernels with higher overhead
    opts = ""
    # `clock_32` provides lower overhead per record, `time_shift`` post-processes to reduce noise
    if args.increase_accuracy:
        opts = "clock32,time_shift"

    # Set up profiling mode based on warp sampling preferences
    if args.warp_sampling:
        # Selective warp sampling allows capturing more events within buffer constraints
        # by only profiling specified warps (e.g. "0,1,2,3")
        mode = proton.mode.Default(
            optimizations=opts,
            sampling_strategy="selective",
            sampling_options=args.warp_ids,
        )
    else:
        # Profile all warps - provides complete picture but uses more buffer space
        mode = proton.mode.Default(optimizations=opts)

    return args.op_measure, mode


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pl.enter_scope("kernel")
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    with pl.scope("load_and_add"):
        with pl.scope("load_x_issue"):
            x = tl.load(x_ptr + offsets, mask=mask)
        with pl.scope("load_y_issue"):
            y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
    pl.exit_scope("kernel")


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=NUM_WARPS)
    return output


if __name__ == "__main__":
    description = "Triton Vector Add with Proton Intra-Kernel Profiling"
    print(description)

    # Explicit Proton DSL enablement for Triton kernels.
    # Be careful NOT to insert proton ops in loops (use the ttgir override approach instead).
    pl.enable_semantic("triton")

    op_measure, mode = config_helper(description)

    # Start profiling with appropriate backend and output format
    if op_measure:
        # Operation measurement mode generates scope-level metrics
        # View results with: proton-viewer -m normalized_cycles vector-add.hatchet
        # Note: cycles are averaged across all warps/CTAs - adjust for warp specialization
        proton.start("vector-add", backend="instrumentation", mode=mode)
    else:
        # Timeline trace mode generates Chrome trace format for visualization
        # Output file: vector-add.chrome_trace
        proton.start("vector-add", data="trace", backend="instrumentation", mode=mode)

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = add(x, y)
    torch.testing.assert_close(output_torch, output_triton, rtol=1e-3, atol=1e-1)
    proton.finalize()


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
def blocked_matmul_pipelined_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    pl.enter_scope("blocked_matmul_pipelined_kernel")

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

        with pl.scope("tma_loads_issue"):
            tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a)
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b)

        with pl.scope("tma_loads_wait"):
            mbarrier.wait(bar, phase=phase)
        phase ^= 1

        # Since `warpgroup_mma_wait` is a no-op when there are no WGMMAs in
        # flight, we can overlap the WGMMA by waiting first, then issuing the
        # async WGMMA.
        with pl.scope("wgmma_wait"):
            acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc, ))

        with pl.scope("wgmma_issue"):
            acc = warpgroup_mma(a, b, acc, is_async=True)

        # Move to the next buffer. The TMA load will start while the WGMMA is
        # still running.
        index ^= 1

    # Wait for the last WGMMA to complete.
    with pl.scope("wgmma_last_wait"):
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc, ))

    mbarrier.invalidate(bar)

    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)

    pl.exit_scope("blocked_matmul_pipelined_kernel")


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


if __name__ == "__main__":
    if not is_hopper():
        raise RuntimeError("This tutorial requires a Hopper NVIDIA GPU")

    description = "Gluon Matrix Multiplication with Proton Intra-Kernel Profiling"
    print(description)

    M, N, K = 512, 512, 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128

    op_measure, mode = config_helper(description)

    # Start profiling with appropriate backend and output format
    if op_measure:
        # Operation measurement mode generates scope-level metrics
        # View results with: proton-viewer -m normalized_cycles gemm.hatchet
        # Note: cycles are averaged across all warps/CTAs - adjust for warp specialization
        proton.start("gemm", backend="instrumentation", mode=mode)
    else:
        # Timeline trace mode generates Chrome trace format for visualization
        # Output file: gemm.chrome_trace
        proton.start("gemm", data="trace", backend="instrumentation", mode=mode)

    blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)

    # Complete profiling and write output files
    proton.finalize()
