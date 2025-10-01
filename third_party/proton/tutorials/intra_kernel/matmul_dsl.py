"""
Matrix Multiplication with Triton Intra-Kernel Profiling using Proton DSL

This tutorial demonstrates how to profile a Gluon-based matrix multiplication
kernel using Proton's Domain Specific Language (DSL) for intra-kernel profiling.
The implementation uses NVIDIA Hopper architecture features like WGMMA and TMA
for optimized performance.
"""

import argparse
import importlib.util

import torch
import triton
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

# Import the Gluon WGMMA tutorial module for utility functions
module_path = "../../../../python/tutorials/gluon/05-wgmma.py"
spec = importlib.util.spec_from_file_location("wgmma_tutorial", module_path)
t5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t5)


@gluon.jit
def blocked_matmul_pipelined_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    pl.enter_scope("blocked_matmul_pipelined_kernel")

    # Allocate 2 buffers for each A and B.
    a_smem = gl.allocate_shared_memory(
        dtype, [2] + a_desc.block_type.shape, a_desc.layout
    )
    b_smem = gl.allocate_shared_memory(
        dtype, [2] + b_desc.block_type.shape, b_desc.layout
    )
    index = 0

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    mma_layout: gl.constexpr = t5.pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    acc = warpgroup_mma_init(
        gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
    )

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
            acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

        with pl.scope("wgmma_issue"):
            acc = warpgroup_mma(a, b, acc, is_async=True)

        # Move to the next buffer. The TMA load will start while the WGMMA is
        # still running.
        index ^= 1

    # Wait for the last WGMMA to complete.
    with pl.scope("wgmma_last_wait"):
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

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
    if not t5.is_hopper():
        raise RuntimeError("This tutorial requires a Hopper NVIDIA GPU")

    # Configure command line arguments for profiling options
    parser = argparse.ArgumentParser(
        description="Matrix multiplication with Triton intra kernel profiling"
    )
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
        default=True,
        help="Enable increased-accuracy during profiling (default: True)",
    )
    parser.add_argument(
        "--warp-ids",
        type=str,
        default="0, 2",
        help="Comma-separated list of warp IDs for warp sampling (default: '0, 2')",
    )

    args = parser.parse_args()

    M, N, K = 512, 512, 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
    num_warps = 8

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

    # Start profiling with appropriate backend and output format
    if args.op_measure:
        # Operation measurement mode generates scope-level metrics
        # View results with: proton-viewer -m normalized_cycles gemm.hatchet
        # Note: cycles are averaged across all warps/CTAs - adjust for warp specialization
        proton.start("gemm", backend="instrumentation", mode=mode)
    else:
        # Timeline trace mode generates Chrome trace format for visualization
        # Output file: gemm.chrome_trace
        proton.start("gemm", data="trace", backend="instrumentation", mode=mode)

    blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)

    # Complete profiling and write output files
    proton.finalize()
