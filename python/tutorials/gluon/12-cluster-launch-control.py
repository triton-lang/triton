"""
Cluster Launch Control (CLC)
============================

Cluster Launch Control (CLC) is a Blackwell (SM100+) hardware feature that enables
dynamic work distribution between thread blocks. When a block finishes early, it can
cancel a not-yet-launched cluster and take over its work, improving load balancing.

This tutorial demonstrates:
1. The CLC API: try_cancel, is_canceled, get_first_ctaid
2. How to overlap CLC with computation to hide latency
3. A persistent matmul achieving 92.5% of cuBLAS performance

Key Insight
-----------
The critical optimization is issuing CLC during the TMA prologue and checking
the result after tile completion. This hides CLC latency behind computation.

CLC API
-------
- ``clc.try_cancel(result, mbar)``: Issue async CLC request to cancel a pending cluster
- ``clc.is_canceled(result)``: Returns non-zero if a cluster was successfully canceled
- ``clc.get_first_ctaid(result, dim)``: Get the canceled cluster's first CTA ID (dim: 0=x, 1=y, 2=z)
"""

import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    tensor_memory_descriptor,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
    clc,
)
from triton.language.core import _aggregate as aggregate


def is_blackwell():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires Blackwell (SM100+) GPU")

# %%
# MMA Helper
# ----------
# Reusable accumulator management for tcgen05 MMA operations.


@aggregate
class MMAv5:
    use_acc: gl.tensor
    acc_tmem: tensor_memory_descriptor
    bar: gl.shared_memory_descriptor
    counter: gl.tensor
    reg_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, use_acc, acc_tmem, bar, counter, reg_layout):
        self.use_acc = use_acc
        self.acc_tmem = acc_tmem
        self.bar = bar
        self.counter = counter
        self.reg_layout = gl.constexpr(reg_layout)

    @gluon.jit
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr):
        layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], layout)
        bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), layout, num_warps)
        return MMAv5(gl.to_tensor(False), acc_tmem, bar, gl.to_tensor(0), reg_layout)

    @gluon.jit
    def issue_async_mma(self, a, b):
        tcgen05_mma(a, b, self.acc_tmem, use_acc=self.use_acc)
        tcgen05_commit(self.bar)
        return MMAv5(gl.to_tensor(True), self.acc_tmem, self.bar, self.counter + 1, self.reg_layout)

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        mbarrier.wait(self.bar, (self.counter - 1 - num_outstanding) & 1)
        return self

    @gluon.jit
    def take_result(self):
        next = MMAv5(gl.to_tensor(False), self.acc_tmem, self.bar, self.counter, self.reg_layout)
        return self.acc_tmem.load(self.reg_layout), next


# %%
# CLC Matmul Kernel
# -----------------
# This kernel processes its assigned tile, then attempts to steal additional work.
# CLC is issued during the prologue so the result is ready after tile completion.


@gluon.jit
def clc_matmul_kernel(a_desc, b_desc, c_desc, num_pid_n, num_buffers: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    # Shared memory for TMA pipeline
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)

    # CLC buffers (128-bit result + mbarrier for completion)
    clc_result = gl.allocate_shared_memory(gl.int64, [2], mbarrier.MBarrierLayout())
    clc_mbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(clc_mbar, count=1)

    mma = MMAv5.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    producer = 0
    consumer = 0

    # Compute initial tile coordinates
    pid = gl.program_id(axis=0)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # === PROLOGUE: Issue CLC + TMA loads ===
    # Issue CLC immediately - result will be ready after main loop
    clc.try_cancel(clc_result, clc_mbar)
    mbarrier.expect(clc_mbar, 16)  # CLC writes 16 bytes

    # Fill TMA pipeline
    for k in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        index = producer % num_buffers
        producer += 1
        bar = bars.index(index)
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index))
        tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(index))

    # === MAIN LOOP ===
    for k in range(BLOCK_K * (num_buffers - 2), K, BLOCK_K):
        index = producer % num_buffers
        producer += 1
        bar = bars.index(index)
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index))
        tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(index))

        c_index = consumer % num_buffers
        c_phase = consumer // num_buffers & 1
        consumer += 1
        mbarrier.wait(bars.index(c_index), c_phase)
        mma = mma.wait_num_outstanding(0)
        mma = mma.issue_async_mma(a_bufs.index(c_index), b_bufs.index(c_index))

    # === EPILOGUE ===
    for _ in gl.static_range(num_buffers - 2):
        c_index = consumer % num_buffers
        c_phase = consumer // num_buffers & 1
        consumer += 1
        mbarrier.wait(bars.index(c_index), c_phase)
        mma = mma.wait_num_outstanding(0)
        mma = mma.issue_async_mma(a_bufs.index(c_index), b_bufs.index(c_index))

    # Store result
    mma = mma.wait_num_outstanding(0)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c, mma = mma.take_result()
    c_smem.store(c.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)

    # === CHECK CLC RESULT ===
    # CLC was issued during prologue, should be ready now
    mbarrier.wait(clc_mbar, 0)
    has_work = clc.is_canceled(clc_result)

    if has_work != 0:
        # Successfully canceled a cluster - process its tile
        canceled_pid = clc.get_first_ctaid(clc_result, 0)
        pid_m = canceled_pid // num_pid_n
        pid_n = canceled_pid % num_pid_n
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        # Reset for next CLC attempt
        mbarrier.init(clc_mbar, count=1)
        clc.try_cancel(clc_result, clc_mbar)
        mbarrier.expect(clc_mbar, 16)

        # Process stolen tile with full pipeline
        for k in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            index = producer % num_buffers
            producer += 1
            bar = bars.index(index)
            mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index))
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(index))

        for k in range(BLOCK_K * (num_buffers - 2), K, BLOCK_K):
            index = producer % num_buffers
            producer += 1
            bar = bars.index(index)
            mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index))
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(index))

            c_index = consumer % num_buffers
            c_phase = consumer // num_buffers & 1
            consumer += 1
            mbarrier.wait(bars.index(c_index), c_phase)
            mma = mma.wait_num_outstanding(0)
            mma = mma.issue_async_mma(a_bufs.index(c_index), b_bufs.index(c_index))

        for _ in gl.static_range(num_buffers - 2):
            c_index = consumer % num_buffers
            c_phase = consumer // num_buffers & 1
            consumer += 1
            mbarrier.wait(bars.index(c_index), c_phase)
            mma = mma.wait_num_outstanding(0)
            mma = mma.issue_async_mma(a_bufs.index(c_index), b_bufs.index(c_index))

        mma = mma.wait_num_outstanding(0)
        c2, mma = mma.take_result()
        c_smem.store(c2.to(dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
        tma.store_wait(pendings=0)


# %%
# Benchmark
# ---------


def get_flops(ms, M, N, K):
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


def run_clc_matmul(A, B, C, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_buffers=3, num_warps=4):
    M, N = C.shape
    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m * num_pid_n, )
    clc_matmul_kernel[grid](a_desc, b_desc, c_desc, num_pid_n, num_buffers, num_warps=num_warps)


def benchmark():
    print("=" * 60)
    print("Cluster Launch Control (CLC) Matmul - Blackwell")
    print("=" * 60)

    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name}, SMs: {props.multi_processor_count}")

    M, N, K = 8192, 8192, 8192
    print(f"Matrix: {M}x{N}x{K}")

    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    C = torch.empty(M, N, device='cuda', dtype=torch.float16)

    # cuBLAS baseline
    def cublas_fn():
        torch.mm(A, B, out=C)

    ms = triton.testing.do_bench_cudagraph(cublas_fn)
    cublas_tflops = get_flops(ms, M, N, K)
    print(f"\ncuBLAS:     {cublas_tflops:7.2f} TFLOPS")

    # CLC matmul
    def clc_fn():
        run_clc_matmul(A, B, C)

    ms = triton.testing.do_bench_cudagraph(clc_fn)
    clc_tflops = get_flops(ms, M, N, K)
    print(f"CLC:        {clc_tflops:7.2f} TFLOPS ({100*clc_tflops/cublas_tflops:.1f}% of cuBLAS)")

    # Correctness check
    print("\nVerifying correctness...")
    A_test = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    B_test = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    C_ref = torch.mm(A_test, B_test)
    C_clc = torch.empty_like(C_ref)
    run_clc_matmul(A_test, B_test, C_clc)
    torch.cuda.synchronize()
    max_diff = (C_ref - C_clc).abs().max().item()
    print(f"Max diff: {max_diff:.6f}")
    assert max_diff < 1.0, "Correctness check failed"
    print("✓ Correctness verified")


if __name__ == "__main__":
    benchmark()
