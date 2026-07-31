"""
Event-based intra-kernel profiling for pipelined TCGen05 matmul.

This example adapts ``python/tutorials/gluon/06-tcgen05.py`` and
``python/tutorials/gluon/08-warp-specialization.py`` and records the lifetimes
of asynchronous TMA and TCGen05 MMA transactions. Unlike a regular Proton
scope, an event can start in one loop iteration and end at the matching barrier
wait in a later iteration.

Run either example on a Blackwell GPU:

    python3 example_events.py
    python3 example_events.py -ws

The warp-specialized matmul uses scopes for synchronous issue and wait regions.
It also starts TMA and TCGen05 events in producer partitions and ends them in
consumer partitions, so its trace contains flows across warps.
Open the resulting ``*.chrome_trace`` file in Perfetto or
``chrome://tracing``.
"""

import argparse

import torch
import triton
import triton.profiler as proton
import triton.profiler.language as pl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
    tma,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


@gluon.jit
def get_and_increment(counter):
    return counter % 2, counter // 2 & 1, counter + 1


@gluon.jit
def start_buffered_event(index, event0, event1):
    if index == 0:
        pl.start_event(event0)
    else:
        pl.start_event(event1)


@gluon.jit
def end_buffered_event(index, event0, event1):
    if index == 0:
        pl.end_event(event0)
    else:
        pl.end_event(event1)


@gluon.jit
def ws_epilogue_partition(a_desc, b_desc, c_desc, a_bufs, b_bufs, load_empty_bars, load_ready_bars, acc_tmem,
                          mma_done_bar, tma_event0, tma_event1, mma_event0, mma_event1, off_m, off_n):
    with pl.scope("ws_mma_wait"):
        mbarrier.wait(mma_done_bar, phase=0)
        # No later buffer-reuse wait observes the final two MMAs, so drain
        # their events after the barrier that tracks all preceding MMA work.
        pl.end_event(mma_event0)
        if gl.cdiv(a_desc.shape[1], a_desc.block_type.shape[1]) > 1:
            pl.end_event(mma_event1)

    with pl.scope("ws_epilogue_store"):
        c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
        c_smem.store(acc_tmem.load().to(c_desc.dtype))
        fence_async_shared()
        tma.async_store(c_desc, [off_m, off_n], c_smem)
        tma.store_wait(pendings=0)


@gluon.jit
def ws_load_partition(a_desc, b_desc, c_desc, a_bufs, b_bufs, load_empty_bars, load_ready_bars, acc_tmem, mma_done_bar,
                      tma_event0, tma_event1, mma_event0, mma_event1, off_m, off_n):
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    num_buffers: gl.constexpr = a_bufs.type.shape[0]
    for tile in range(gl.cdiv(a_desc.shape[1], block_k)):
        index = tile % num_buffers
        phase = tile // num_buffers & 1
        with pl.scope("ws_load_empty_wait"):
            mbarrier.wait(load_empty_bars.index(index), phase ^ 1)
            # Reusing this buffer proves that its previous MMA completed.
            if tile >= num_buffers:
                end_buffered_event(index, mma_event0, mma_event1)

        with pl.scope("ws_load_ready_issue"):
            ready_bar = load_ready_bars.index(index)
            mbarrier.expect(ready_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            start_buffered_event(index, tma_event0, tma_event1)
            k = tile * block_k
            tma.async_load(a_desc, [off_m, k], ready_bar, a_bufs.index(index))
            tma.async_load(b_desc, [k, off_n], ready_bar, b_bufs.index(index))


@gluon.jit
def ws_mma_partition(a_desc, b_desc, c_desc, a_bufs, b_bufs, load_empty_bars, load_ready_bars, acc_tmem, mma_done_bar,
                     tma_event0, tma_event1, mma_event0, mma_event1, off_m, off_n):
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    num_buffers: gl.constexpr = a_bufs.type.shape[0]
    use_acc = False
    for tile in range(gl.cdiv(a_desc.shape[1], block_k)):
        index = tile % num_buffers
        phase = tile // num_buffers & 1
        with pl.scope("ws_load_ready_wait"):
            mbarrier.wait(load_ready_bars.index(index), phase)
            end_buffered_event(index, tma_event0, tma_event1)
        with pl.scope("ws_load_empty_issue"):
            start_buffered_event(index, mma_event0, mma_event1)
            tcgen05_mma(a_bufs.index(index), b_bufs.index(index), acc_tmem, use_acc=use_acc)
            tcgen05_commit(load_empty_bars.index(index))
        use_acc = True
    with pl.scope("ws_mma_issue"):
        tcgen05_commit(mma_done_bar)


@gluon.jit
def warp_specialized_matmul_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    block_m: gl.constexpr = c_desc.block_type.shape[0]
    block_n: gl.constexpr = c_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    num_buffers: gl.constexpr = 2

    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    tmem_layout: gl.constexpr = TensorMemoryLayout([block_m, block_n], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [block_m, block_n], tmem_layout)
    mma_done_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_done_bar, count=1)

    tma_event0 = pl.allocate_event("ws_tma_load[0]")
    tma_event1 = pl.allocate_event("ws_tma_load[1]")
    # MMAs using different operand buffers may be in flight concurrently.
    mma_event0 = pl.allocate_event("ws_tcgen05_mma[0]")
    mma_event1 = pl.allocate_event("ws_tcgen05_mma[1]")
    off_m = gl.program_id(axis=0) * block_m
    off_n = gl.program_id(axis=1) * block_n
    partition_args = (a_desc, b_desc, c_desc, a_bufs, b_bufs, load_empty_bars, load_ready_bars, acc_tmem, mma_done_bar,
                      tma_event0, tma_event1, mma_event0, mma_event1, off_m, off_n)

    # As in python/tutorials/gluon/08-warp-specialization.py, the epilogue is
    # the default partition while TMA loading and MMA issue run in worker
    # partitions. Four warps per partition make each start/end pair cross from
    # one set of warp lines to another in the trace.
    gl.warp_specialize([
        (ws_epilogue_partition, partition_args),
        (ws_load_partition, partition_args),
        (ws_mma_partition, partition_args),
    ], [4, 4], [24, 24])


@gluon.jit
def matmul_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    block_m: gl.constexpr = c_desc.block_type.shape[0]
    block_n: gl.constexpr = c_desc.block_type.shape[1]
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    k_size = a_desc.shape[1]

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * (2 * block_m)
    off_n = pid_n * block_n

    # The pipeline computes an upper and lower output tile and double-buffers
    # their inputs.
    u_bufs = gl.allocate_shared_memory(dtype, [2] + a_desc.block_type.shape, a_desc.layout)
    v_bufs = gl.allocate_shared_memory(dtype, [2] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [2] + b_desc.block_type.shape, b_desc.layout)

    tmem_layout: gl.constexpr = TensorMemoryLayout([block_m, block_n], col_stride=1)
    ub_tmem = allocate_tensor_memory(gl.float32, [block_m, block_n], tmem_layout)
    vb_tmem = allocate_tensor_memory(gl.float32, [block_m, block_n], tmem_layout)

    mma_ub_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mma_vb_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    load_ub_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    load_v_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        mbarrier.init(mma_ub_bars.index(i), count=1)
        mbarrier.init(mma_vb_bars.index(i), count=1)
        mbarrier.init(load_ub_bars.index(i), count=1)
        mbarrier.init(load_v_bars.index(i), count=1)

    # Allocation associates each event with a static name. TMA events are
    # double-buffered because both pipeline stages can be in flight at once.
    # Starts and ends may execute in different iterations and may be recorded
    # by different warps.
    tma_ub_event0 = pl.allocate_event("tma_load_upper_and_b[0]")
    tma_ub_event1 = pl.allocate_event("tma_load_upper_and_b[1]")
    tma_v_event0 = pl.allocate_event("tma_load_lower[0]")
    tma_v_event1 = pl.allocate_event("tma_load_lower[1]")
    mma_ub_event = pl.allocate_event("tcgen05_mma_upper")
    mma_vb_event = pl.allocate_event("tcgen05_mma_lower")
    store_event = pl.allocate_event("tma_store")

    load_counter = 0
    mma_counter = 0
    k = 0
    ub_acc = False
    vb_acc = False

    # Prime both stages of the TMA pipeline. Each event ends only when the
    # corresponding mbarrier wait is reached below.
    for _ in gl.static_range(2):
        load_index, load_phase, load_counter = get_and_increment(load_counter)
        load_ub_bar = load_ub_bars.index(load_index)
        mbarrier.expect(load_ub_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        start_buffered_event(load_index, tma_ub_event0, tma_ub_event1)
        tma.async_load(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))
        tma.async_load(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))

        load_v_bar = load_v_bars.index(load_index)
        mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
        start_buffered_event(load_index, tma_v_event0, tma_v_event1)
        tma.async_load(a_desc, [off_m + block_m, k], load_v_bar, v_bufs.index(load_index))
        k += block_k

    for _ in range(gl.cdiv(k_size, block_k) - 2):
        mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)

        # Complete the TMA transactions issued in an earlier iteration, then
        # start asynchronous MMAs over the loaded buffers.
        mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
        end_buffered_event(mma_index, tma_ub_event0, tma_ub_event1)
        pl.start_event(mma_ub_event)
        tcgen05_mma(u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=ub_acc)
        tcgen05_commit(mma_ub_bars.index(mma_index))
        ub_acc = True

        mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
        end_buffered_event(mma_index, tma_v_event0, tma_v_event1)
        pl.start_event(mma_vb_event)
        tcgen05_mma(v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=vb_acc)
        tcgen05_commit(mma_vb_bars.index(mma_index))
        vb_acc = True

        # The MMA completion waits close events started above. The same shared
        # buffers can then be reused for the next pair of TMA transactions.
        load_index, load_phase, load_counter = get_and_increment(load_counter)
        mbarrier.wait(mma_ub_bars.index(mma_index), mma_phase)
        pl.end_event(mma_ub_event)
        load_ub_bar = load_ub_bars.index(load_index)
        mbarrier.expect(load_ub_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        start_buffered_event(load_index, tma_ub_event0, tma_ub_event1)
        tma.async_load(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))

        mbarrier.wait(mma_vb_bars.index(mma_index), mma_phase)
        pl.end_event(mma_vb_event)
        tma.async_load(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))
        load_v_bar = load_v_bars.index(load_index)
        mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
        start_buffered_event(load_index, tma_v_event0, tma_v_event1)
        tma.async_load(a_desc, [off_m + block_m, k], load_v_bar, v_bufs.index(load_index))
        k += block_k

    mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
    ub_bar = mma_ub_bars.index(mma_index)
    vb_bar = mma_vb_bars.index(mma_index)
    epilogue_phase = mma_phase

    # Drain the last two pipeline stages. One MMA event covers the final pair
    # of implicitly ordered MMA instructions and ends at their commit barrier.
    mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
    end_buffered_event(mma_index, tma_ub_event0, tma_ub_event1)
    pl.start_event(mma_ub_event)
    tcgen05_mma(u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=ub_acc)

    mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
    end_buffered_event(mma_index, tma_v_event0, tma_v_event1)
    pl.start_event(mma_vb_event)
    tcgen05_mma(v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=vb_acc)

    mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
    mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
    end_buffered_event(mma_index, tma_ub_event0, tma_ub_event1)
    tcgen05_mma(u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=True)
    tcgen05_commit(ub_bar)

    mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
    end_buffered_event(mma_index, tma_v_event0, tma_v_event1)
    tcgen05_mma(v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=True)
    tcgen05_commit(vb_bar)

    mbarrier.wait(ub_bar, epilogue_phase)
    pl.end_event(mma_ub_event)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    ub = ub_tmem.load()
    c_smem.store(ub.to(dtype))
    fence_async_shared()
    pl.start_event(store_event)
    tma.async_store(c_desc, [off_m, off_n], c_smem)

    mbarrier.wait(vb_bar, epilogue_phase)
    pl.end_event(mma_vb_event)
    vb = vb_tmem.load()
    tma.store_wait(pendings=0)
    pl.end_event(store_event)
    c_smem.store(vb.to(dtype))
    fence_async_shared()
    pl.start_event(store_event)
    tma.async_store(c_desc, [off_m + block_m, off_n], c_smem)
    tma.store_wait(pendings=0)
    pl.end_event(store_event)


def matmul(a, b, c, block_m, block_n, block_k, num_warps):
    m, n = c.shape
    a_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([block_k, block_n], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float16)
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n], b_layout)
    c_desc = TensorDescriptor.from_tensor(c, [block_m, block_n], c_layout)
    grid = (triton.cdiv(m, 2 * block_m), triton.cdiv(n, block_n))
    matmul_kernel[grid](a_desc, b_desc, c_desc, num_warps=num_warps)


def warp_specialized_matmul(a, b, c, block_m, block_n, block_k, num_warps):
    a_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([block_k, block_n], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float16)
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n], b_layout)
    c_desc = TensorDescriptor.from_tensor(c, [block_m, block_n], c_layout)
    grid = (triton.cdiv(c.shape[0], block_m), triton.cdiv(c.shape[1], block_n))
    warp_specialized_matmul_kernel[grid](a_desc, b_desc, c_desc, num_warps=num_warps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-ws", "--warp-specialized", action="store_true", help="profile a warp-specialized matmul")
    args = parser.parse_args()

    if not is_blackwell():
        raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")

    torch.manual_seed(0)
    m, n, k = 2048, 2048, 2048
    block_m, block_n, block_k = 128, 128, 128
    mode = proton.mode.Default(optimizations="clock32,time_shift")
    if args.warp_specialized:
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        c = torch.empty(m, n, device="cuda", dtype=torch.float16)
        trace_name = "warp-specialized-events"
        proton.start(trace_name, data="trace", backend="instrumentation", mode=mode)
        warp_specialized_matmul(a, b, c, block_m, block_n, block_k, num_warps=4)
        proton.finalize()
        torch.testing.assert_close(a @ b, c, rtol=1e-3, atol=1e-1)
    else:
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        c = torch.empty(m, n, device="cuda", dtype=torch.float16)
        trace_name = "tcgen05-events"
        proton.start(trace_name, data="trace", backend="instrumentation", mode=mode)
        matmul(a, b, c, block_m, block_n, block_k, num_warps=4)
        proton.finalize()
        torch.testing.assert_close(a @ b, c, rtol=1e-3, atol=1e-1)
    print(f"Wrote {trace_name}.chrome_trace")
