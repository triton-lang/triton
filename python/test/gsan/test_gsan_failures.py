from __future__ import annotations

import functools
import inspect
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_mma,
)

from triton._internal_testing import is_blackwell, is_cuda, is_hopper_or_newer, run_in_process
from triton.experimental.gsan import create_mem_pool
from triton.experimental.gsan._testing_utils import atomic_poll, shadow_cell_from_address
from triton.tools.tensor_descriptor import TensorDescriptor

pytestmark = pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")

RELEASE_ACQUIRE_SYNC_CASES = (
    pytest.param("release", "acquire", id="release-acquire"),
    pytest.param("release", "acq_rel", id="release-acq-rel"),
    pytest.param("acq_rel", "acquire", id="acq-rel-acquire"),
    pytest.param("acq_rel", "acq_rel", id="acq-rel-acq-rel"),
)

CROSS_SM_SEMANTIC_MISMATCH_CASES = (
    pytest.param("relaxed", "acquire", "gpu", id="producer-relaxed-consumer-acquire-scope-gpu"),
    pytest.param("relaxed", "acquire", "sys", id="producer-relaxed-consumer-acquire-scope-sys"),
    pytest.param("release", "relaxed", "gpu", id="producer-release-consumer-relaxed-scope-gpu"),
    pytest.param("release", "relaxed", "sys", id="producer-release-consumer-relaxed-scope-sys"),
)

TRANSITIVE_RELAY_MISMATCH_CASES = (
    pytest.param("release", "relaxed", "gpu", id="relay-relaxed-scope-gpu"),
    pytest.param("release", "relaxed", "sys", id="relay-relaxed-scope-sys"),
    pytest.param("acq_rel", "release", "gpu", id="relay-release-scope-gpu"),
    pytest.param("acq_rel", "release", "sys", id="relay-release-scope-sys"),
)


@triton.jit
def _raw_kernel(ptr, scratch_ptr, counter_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(ptr, 1)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        value = tl.load(ptr)
        tl.store(scratch_ptr, value)


@triton.jit
def _war_kernel(ptr, scratch_ptr, counter_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        value = tl.load(ptr)
        tl.store(scratch_ptr, value)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        tl.store(ptr, 1)


@triton.jit
def _waw_kernel(ptr, scratch_ptr, counter_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(ptr, 1)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        tl.store(ptr, 2)


@triton.jit
def _pdl_producer_kernel(payload_ptr):
    pid = tl.program_id(0)
    tl.store(payload_ptr + pid, 1000 + pid)
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _pdl_consumer_without_wait_kernel(payload_ptr, scratch_ptr):
    value = tl.load(payload_ptr + 1)
    tl.store(scratch_ptr, value)


@triton.jit
def _pdl_conditional_wait_kernel(should_wait_ptr, scratch_ptr):
    if tl.load(should_wait_ptr) != 0:
        tl.extra.cuda.gdc_wait()
    tl.store(scratch_ptr, 1)


@triton.jit
def _pdl_transitively_acquired_producer_kernel(payload_ptr, flag_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem="release", scope="gpu")
    else:
        atomic_poll(flag_ptr, 1, sem="acquire", scope="gpu")
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _pdl_transitively_acquired_consumer_without_wait_kernel(payload_ptr, scratch_ptr):
    value = tl.load(payload_ptr)
    tl.store(scratch_ptr + tl.program_id(0), value)


@triton.jit
def _cross_sm_atomic_sync_kernel(payload_ptr, flag_ptr, counter_ptr, scratch_ptr, producer_sem: tl.constexpr,
                                 consumer_sem: tl.constexpr, scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem=producer_sem, scope=scope)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    elif pid == 1:
        atomic_poll(counter_ptr, 1)
        ready = 0
        while ready != 1:
            ready = tl.atomic_add(flag_ptr, 0, sem=consumer_sem, scope=scope)
        result = tl.load(payload_ptr)
        tl.store(scratch_ptr, result)


@triton.jit
def _atomic_poll_cross_sm_sync_kernel(payload_ptr, flag_ptr, scratch_ptr, producer_sem: tl.constexpr,
                                      consumer_sem: tl.constexpr, scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem=producer_sem, scope=scope)
    elif pid == 1:
        tl.atomic_poll(flag_ptr, 1, sem=consumer_sem, scope=scope)
        result = tl.load(payload_ptr)
        tl.store(scratch_ptr, result)


@triton.jit
def _atomic_load_store_cross_sm_sync_kernel(payload_ptr, flag_ptr, counter_ptr, scratch_ptr, producer_sem: tl.constexpr,
                                            consumer_sem: tl.constexpr, scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_store(flag_ptr, 1, sem=producer_sem, scope=scope)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    elif pid == 1:
        atomic_poll(counter_ptr, 1)
        ready = tl.atomic_load(flag_ptr, sem=consumer_sem, scope=scope)
        while ready != 1:
            ready = tl.atomic_load(flag_ptr, sem=consumer_sem, scope=scope)
        result = tl.load(payload_ptr)
        tl.store(scratch_ptr, result)


@triton.jit
def _transitive_atomic_sync_kernel(payload_ptr, flag0_ptr, flag1_ptr, counter_ptr, scratch_ptr,
                                   release_sem: tl.constexpr, relay_sem: tl.constexpr, scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag0_ptr, 1, sem=release_sem, scope=scope)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    elif pid == 1:
        atomic_poll(counter_ptr, 1)
        ready = 0
        while ready != 1:
            ready = tl.atomic_add(flag0_ptr, 0, sem=relay_sem, scope=scope)
        tl.atomic_xchg(flag1_ptr, 1, sem=release_sem, scope=scope)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    elif pid == 2:
        atomic_poll(counter_ptr, 2)
        ready = 0
        while ready != 1:
            ready = tl.atomic_add(flag1_ptr, 0, sem="acquire", scope=scope)
        result = tl.load(payload_ptr)
        tl.store(scratch_ptr, result)


@triton.jit
def _tma_raw_kernel(ptr, scratch_ptr, counter_ptr, m_size, n_size, row_idx, col_idx, stride_0, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        desc = tl.make_tensor_descriptor(ptr, [m_size, n_size], [stride_0, 1], [BLOCK, BLOCK])
        values = tl.full((BLOCK, BLOCK), 1, dtype=tl.int32)
        desc.store([row_idx, col_idx], values)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        value = tl.load(ptr + row_idx * stride_0 + col_idx)
        tl.store(scratch_ptr, value)


@triton.jit
def _host_tma_war_kernel(target_ptr, target_desc, scratch_desc, counter_ptr, row_idx, col_idx, stride_0):
    pid = tl.program_id(0)
    if pid == 0:
        block = target_desc.load([row_idx, col_idx])
        scratch_desc.store([row_idx, col_idx], block)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        tl.store(target_ptr + row_idx * stride_0 + col_idx, 1)


@triton.jit
def _host_tma_gather_war_kernel(target_ptr, target_desc, x_offsets_ptr, scratch_ptr, counter_ptr, row_idx, y_offset,
                                stride_0, scratch_stride_0, scratch_stride_1, BLOCK_X: tl.constexpr):
    BLOCK_Y: tl.constexpr = target_desc.block_shape[1]
    pid = tl.program_id(0)
    if pid == 0:
        x_offsets = tl.load(x_offsets_ptr + tl.arange(0, BLOCK_X))
        values = target_desc.gather(x_offsets, y_offset)
        indices_x = tl.arange(0, BLOCK_X)[:, None] * scratch_stride_0
        indices_y = tl.arange(0, BLOCK_Y)[None, :] * scratch_stride_1
        tl.store(scratch_ptr + indices_x + indices_y, values)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        tl.store(target_ptr + row_idx * stride_0 + y_offset, 1)


@triton.jit
def _host_tma_scatter_war_kernel(target_ptr, target_desc, x_offsets_ptr, src_ptr, src_stride_0, src_stride_1,
                                 scratch_ptr, counter_ptr, row_idx, y_offset, stride_0, BLOCK_X: tl.constexpr):
    BLOCK_Y: tl.constexpr = target_desc.block_shape[1]
    pid = tl.program_id(0)
    if pid == 0:
        value = tl.load(target_ptr + row_idx * stride_0 + y_offset)
        tl.store(scratch_ptr, value)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        x_offsets = tl.load(x_offsets_ptr + tl.arange(0, BLOCK_X))
        indices_x = tl.arange(0, BLOCK_X)[:, None] * src_stride_0
        indices_y = tl.arange(0, BLOCK_Y)[None, :] * src_stride_1
        values = tl.load(src_ptr + indices_x + indices_y)
        target_desc.scatter(values, x_offsets, y_offset)


@triton.jit
def _host_tma_atomic_flag_publish_kernel(payload_ptr, flag_ptr, flag_desc, counter_ptr, scratch_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem="release", scope="gpu")
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        BLOCK_X: tl.constexpr = flag_desc.block_shape[0]
        BLOCK_Y: tl.constexpr = flag_desc.block_shape[1]
        values = tl.full((BLOCK_X, BLOCK_Y), 1, dtype=tl.int32)
        # TMA atomics on the released flag are relaxed.gpu and must not acquire
        # the producer's prior payload store.
        flag_desc.atomic_add([0, 0], values)
        result = tl.load(payload_ptr)
        tl.store(scratch_ptr, result)


@triton.jit
def _mixed_scope_release_rmw_kernel(counter_ptr, ready_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.atomic_add(counter_ptr, 1, sem="release", scope="gpu")
        tl.atomic_xchg(ready_ptr, 1, sem="relaxed", scope="gpu")
    elif pid == 1:
        atomic_poll(ready_ptr, 1)
        tl.atomic_add(counter_ptr, 1, sem="release", scope="sys")


@gluon.jit
def _gluon_cluster_barrier_kernel(payload_ptr, out_ptr, RELAXED: gl.constexpr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])

    offsets = gl.arange(0, 256, data_layout)
    payload_ptrs = payload_ptr + offsets * 0
    out_ptrs = out_ptr + offsets * 0
    gl.barrier(cluster=True)
    gl.store(payload_ptrs, 1, mask=offsets == 0)
    if RELAXED:
        hopper.cluster.barrier(relaxed=True)
    else:
        gl.barrier(cluster=True)
    value = gl.load(payload_ptrs, mask=offsets == 128, other=0)
    gl.store(out_ptrs, value, mask=offsets == 128)


@gluon.jit
def _gluon_mbarrier_current_epoch_kernel(payload_ptr, out_ptr, AFTER_ARRIVAL: gl.constexpr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])

    barrier = hopper.mbarrier.allocate_mbarrier(two_ctas=True)
    hopper.mbarrier.init(barrier, count=1)
    offsets = gl.arange(0, 256, data_layout)

    payload_ptrs = payload_ptr + offsets * 0
    if not AFTER_ARRIVAL:
        gl.store(payload_ptrs, 1, mask=offsets == 128)
    hopper.mbarrier.arrive(barrier, count=1)
    hopper.mbarrier.wait(barrier, phase=0)
    if AFTER_ARRIVAL:
        gl.store(payload_ptrs, 1, mask=offsets == 128)
    hopper.cluster.barrier(relaxed=True)
    value = gl.load(payload_ptrs, mask=offsets == 0, other=0)
    gl.store(out_ptr + offsets * 0, value, mask=offsets == 0)
    hopper.mbarrier.invalidate(barrier)


@gluon.jit
def _gluon_tcgen05_commit_no_release_kernel(payload_ptr, out_ptr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])
    signal_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([128, 128], gl.float16, cga_layout=((0, 0), ))
    signal_desc = gl.allocate_shared_memory(gl.float16, [128, 128], signal_layout)

    barrier = hopper.mbarrier.allocate_mbarrier()
    hopper.mbarrier.init(barrier, count=2)
    offsets = gl.arange(0, 256, data_layout)
    payload_ptrs = payload_ptr + offsets * 0
    out_ptrs = out_ptr + offsets * 0
    gl.store(payload_ptrs, 1, mask=offsets == 0)
    tcgen05_commit(barrier, descs=[signal_desc])
    hopper.mbarrier.wait(barrier, phase=0)
    value = gl.load(payload_ptrs, mask=offsets == 128, other=0)
    gl.store(out_ptrs, value, mask=offsets == 128)
    hopper.mbarrier.invalidate(barrier)


@gluon.jit
def _gluon_mbarrier_multicast_partial_wait_kernel(input_desc, payload_ptr, out_ptr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1], [2]])
    signal = gl.allocate_shared_memory(gl.float16, [256, 128], input_desc.layout)
    b_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([128, 256], gl.float16, cga_layout=((0, 1), (0, 2)))
    b = gl.allocate_shared_memory(gl.float16, [128, 256], b_layout)
    acc_layout: gl.constexpr = TensorMemoryLayout([128, 128], col_stride=1, cga_layout=((1, 0), (0, 1)), two_ctas=True)
    acc = allocate_tensor_memory(gl.float32, [256, 256], acc_layout)

    # Four CTAs lower this to a two-element mbarrier with CGA layout [[0], [1]].
    # Each pair of CTAs arrives on a different barrier, and only its leader CTA
    # executes the lowered wait because the first CTA-layout basis is zero.
    barrier = hopper.mbarrier.allocate_mbarrier(two_ctas=True)
    hopper.mbarrier.init(barrier, count=1)
    offsets = gl.arange(0, 512, data_layout)
    payload_ptrs = payload_ptr + offsets * 0
    out_ptrs = out_ptr + offsets * 0
    gl.store(payload_ptrs, 1, mask=offsets == 0)
    hopper.cluster.barrier(relaxed=True)
    hopper.mbarrier.expect(barrier, input_desc.nbytes_per_cta)
    hopper.tma.async_load(input_desc, [0, 0], barrier, signal, multicast=True)
    hopper.mbarrier.wait(barrier, phase=0, deps=[signal])
    value = gl.load(payload_ptrs, mask=offsets == 128, other=0)
    gl.store(out_ptrs, value, mask=offsets == 128)
    tcgen05_mma(signal, b, acc, use_acc=False)
    hopper.cluster.barrier(relaxed=True)
    hopper.mbarrier.invalidate(barrier)


def _cuda_byte_allocator(size: int, _align: int, _stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


def run_with_gsan(fn):

    @functools.wraps(fn)
    def wrapped(*args, **kwargs) -> None:
        triton.knobs.compilation.instrumentation_mode = "gsan"
        pool = create_mem_pool()
        with torch.cuda.use_mem_pool(pool):
            fn(*args, **kwargs)

    return wrapped


@run_with_gsan
def _run_raw_case() -> None:
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _raw_kernel[(2, )](target, scratch, counter, num_warps=1)


@run_with_gsan
def _run_war_case() -> None:
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _war_kernel[(2, )](target, scratch, counter, num_warps=1)


@run_with_gsan
def _run_waw_case() -> None:
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _waw_kernel[(2, )](target, scratch, counter, num_warps=1)


@run_with_gsan
def _run_pdl_without_wait_case() -> None:
    payload = torch.zeros(2, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    _pdl_producer_kernel[(2, )](payload, num_warps=1)
    _pdl_consumer_without_wait_kernel[(1, )](payload, scratch, num_warps=1, launch_pdl=True)
    torch.cuda.synchronize()


@run_with_gsan
def _run_pdl_missing_wait_case() -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    should_wait = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    _pdl_producer_kernel[(1, )](payload, num_warps=1)
    _pdl_conditional_wait_kernel[(1, )](should_wait, scratch, num_warps=1, launch_pdl=True)
    torch.cuda.synchronize()


@run_with_gsan
def _run_pdl_persistent_state_without_wait_case() -> None:
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(num_sms, dtype=torch.int32, device="cuda")
    _pdl_transitively_acquired_producer_kernel[(num_sms, )](payload, flag, num_warps=1)
    _pdl_transitively_acquired_consumer_without_wait_kernel[(num_sms, )](payload, scratch, num_warps=1, launch_pdl=True)
    torch.cuda.synchronize()


@run_with_gsan
def _run_mixed_scope_release_rmw_case() -> None:
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    ready = torch.zeros(1, dtype=torch.int32, device="cuda")
    _mixed_scope_release_rmw_kernel[(2, )](counter, ready, num_warps=1)


@run_with_gsan
def _run_gluon_relaxed_cluster_barrier_case() -> None:
    probe_payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    probe_out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _gluon_cluster_barrier_kernel[(1, )](probe_payload, probe_out, RELAXED=False, num_warps=4, num_ctas=2)
    torch.cuda.synchronize()
    probe_cell = shadow_cell_from_address(probe_payload.data_ptr())
    assert probe_cell.write_clock.thread_id != probe_cell.read_clocks[0].thread_id

    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _gluon_cluster_barrier_kernel[(1, )](payload, out, RELAXED=True, num_warps=4, num_ctas=2)


@run_with_gsan
def _run_gluon_mbarrier_current_epoch_case(after_arrival: bool) -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _gluon_mbarrier_current_epoch_kernel[(1, )](payload, out, after_arrival, num_warps=4, num_ctas=2)


@run_with_gsan
def _run_gluon_tcgen05_commit_no_release_case() -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _gluon_tcgen05_commit_no_release_kernel[(1, )](payload, out, num_warps=4, num_ctas=2)


@run_with_gsan
def _run_gluon_mbarrier_multicast_partial_wait_case() -> None:
    source = torch.zeros((256, 128), dtype=torch.float16, device="cuda")
    signal_layout = gl.NVMMASharedLayout.get_default_for([256, 128], gl.float16, cga_layout=((1, 0), (0, 0)))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(source, [256, 128], signal_layout)
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _gluon_mbarrier_multicast_partial_wait_kernel[(1, )](input_desc, payload, out, num_warps=4, num_ctas=4)


@run_with_gsan
def _run_tma_raw_case() -> None:
    block = 32
    m_size = 35
    n_size = 37
    padded_n = 40
    row_idx = 5
    col_idx = 8

    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    target_storage = torch.zeros((m_size, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:, :n_size]
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    triton.set_allocator(_cuda_byte_allocator)
    _tma_raw_kernel[(2, )](target, scratch, counter, m_size, n_size, row_idx, col_idx, target.stride(0), BLOCK=block)


@run_with_gsan
def _run_host_tma_war_case() -> None:
    block = 32
    m_size = 35
    n_size = 37
    padded_n = 40
    row_idx = 5
    col_idx = 8

    target_storage = torch.zeros((m_size, padded_n), dtype=torch.int32, device="cuda")
    scratch_storage = torch.zeros_like(target_storage)
    target = target_storage[:, :n_size]
    scratch = scratch_storage[:, :n_size]
    target_desc = TensorDescriptor.from_tensor(target, [block, block])
    scratch_desc = TensorDescriptor.from_tensor(scratch, [block, block])
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _host_tma_war_kernel[(2, )](target, target_desc, scratch_desc, counter, row_idx, col_idx, target.stride(0))


@run_with_gsan
def _run_host_tma_gather_war_case() -> None:
    block_x = 8
    block_y = 8
    m_size = 11
    n_size = 13
    padded_m = 16
    padded_n = 16
    row_idx = 5
    y_offset = 8
    x_offsets_values = [5, 7, 9, 10, 1, 3, 11, 13]

    target_storage = torch.zeros((padded_m, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:m_size, :n_size]
    x_offsets = torch.tensor(x_offsets_values, dtype=torch.int32, device="cuda")
    target_desc = TensorDescriptor.from_tensor(target, [1, block_y])
    scratch = torch.zeros((block_x, block_y), dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _host_tma_gather_war_kernel[(2, )](target, target_desc, x_offsets, scratch, counter, row_idx, y_offset,
                                       target.stride(0), scratch.stride(0), scratch.stride(1), BLOCK_X=block_x)


@run_with_gsan
def _run_host_tma_scatter_war_case() -> None:
    block_x = 8
    block_y = 8
    m_size = 11
    n_size = 13
    padded_m = 16
    padded_n = 16
    row_idx = 5
    y_offset = 8
    x_offsets_values = [5, 7, 9, 10, 1, 3, 11, 13]

    target_storage = torch.zeros((padded_m, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:m_size, :n_size]
    x_offsets = torch.tensor(x_offsets_values, dtype=torch.int32, device="cuda")
    target_desc = TensorDescriptor.from_tensor(target, [1, block_y])
    src = torch.arange(1, block_x * block_y + 1, dtype=torch.int32, device="cuda").reshape(block_x, block_y)
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _host_tma_scatter_war_kernel[(2, )](target, target_desc, x_offsets, src, src.stride(0), src.stride(1), scratch,
                                        counter, row_idx, y_offset, target.stride(0), BLOCK_X=block_x)


@run_with_gsan
def _run_host_tma_atomic_flag_publish_case() -> None:
    flag = torch.zeros((1, 16), dtype=torch.int32, device="cuda")
    flag_desc = TensorDescriptor.from_tensor(flag, [1, 16])
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _host_tma_atomic_flag_publish_kernel[(2, )](payload, flag, flag_desc, counter, scratch, num_warps=1)


@run_with_gsan
def _run_cross_sm_atomic_sync_case(producer_sem: str, consumer_sem: str, scope: str) -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _cross_sm_atomic_sync_kernel[(2, )](
        payload,
        flags,
        counter,
        scratch,
        producer_sem=producer_sem,
        consumer_sem=consumer_sem,
        scope=scope,
        num_warps=1,
    )


@run_with_gsan
def _run_atomic_poll_cross_sm_sync_case(producer_sem: str, consumer_sem: str, scope: str) -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _atomic_poll_cross_sm_sync_kernel[(2, )](
        payload,
        flag,
        scratch,
        producer_sem=producer_sem,
        consumer_sem=consumer_sem,
        scope=scope,
        num_warps=4,
    )


@run_with_gsan
def _run_atomic_load_store_cross_sm_sync_case(producer_sem: str, consumer_sem: str, scope: str) -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _atomic_load_store_cross_sm_sync_kernel[(2, )](
        payload,
        flag,
        counter,
        scratch,
        producer_sem=producer_sem,
        consumer_sem=consumer_sem,
        scope=scope,
        num_warps=1,
    )


@run_with_gsan
def _run_transitive_atomic_sync_case(release_sem: str, relay_sem: str, scope: str) -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag0 = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag1 = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _transitive_atomic_sync_kernel[(3, )](
        payload,
        flag0,
        flag1,
        counter,
        scratch,
        release_sem=release_sem,
        relay_sem=relay_sem,
        scope=scope,
        num_warps=1,
    )


def _expected_file_line(source_function, marker: str) -> str:
    source_lines, starting_line = inspect.getsourcelines(source_function)
    for line_offset, line in enumerate(source_lines):
        if marker in line:
            return f"{Path(__file__).name}:{starting_line + line_offset}"
    raise AssertionError(f"Could not find marker {marker!r} for function {source_function!r}")


def _run_failure_case(case: str, *, runner, source_function, marker: str, error: str, runner_args=(),
                      runner_kwargs=None) -> None:
    if torch.cuda.device_count() < 1:
        pytest.skip("requires at least 1 CUDA device")

    if runner_kwargs is None:
        runner_kwargs = {}

    result = run_in_process(runner, runner_args, runner_kwargs)
    print(result.driver_stderr_output)
    assert isinstance(result.exc, RuntimeError), (f"case={case} completed without the expected GSan failure\n"
                                                  f"exc={result.exc!r}\n"
                                                  f"driver stderr:\n{result.driver_stderr_output}")
    assert "GSanLibrary.cu" not in result.driver_stderr_output
    assert Path(__file__).name in result.driver_stderr_output
    assert _expected_file_line(source_function, marker) in result.driver_stderr_output
    assert error in result.driver_stderr_output


def test_read_after_write():
    _run_failure_case("raw", runner=_run_raw_case, source_function=_raw_kernel.fn, marker="value = tl.load(ptr)",
                      error="Read after write race detected")


def test_write_after_read():
    _run_failure_case("war", runner=_run_war_case, source_function=_war_kernel.fn, marker="tl.store(ptr, 1)",
                      error="Write after read race detected")


def test_write_after_write():
    _run_failure_case("waw", runner=_run_waw_case, source_function=_waw_kernel.fn, marker="tl.store(ptr, 2)",
                      error="Write after write race detected")


@pytest.mark.skipif(not is_hopper_or_newer(), reason="PDL requires SM90 or newer")
def test_programmatic_dependent_launch_requires_wait():
    _run_failure_case(
        "pdl_missing_wait",
        runner=_run_pdl_missing_wait_case,
        source_function=_pdl_conditional_wait_kernel.fn,
        marker="def _pdl_conditional_wait_kernel",
        error="kernel launched with programmatic dependent launch did not call gdc_wait",
    )


@pytest.mark.skipif(not is_hopper_or_newer(), reason="PDL requires SM90 or newer")
def test_programmatic_dependent_launch_without_wait_reports_race():
    _run_failure_case(
        "pdl_without_wait",
        runner=_run_pdl_without_wait_case,
        source_function=_pdl_consumer_without_wait_kernel.fn,
        marker="value = tl.load(payload_ptr + 1)",
        error="Read after write race detected",
    )


@pytest.mark.skipif(not is_hopper_or_newer(), reason="PDL requires SM90 or newer")
def test_programmatic_dependent_launch_does_not_inherit_persistent_sm_dependencies():
    _run_failure_case(
        "pdl_persistent_state_without_wait",
        runner=_run_pdl_persistent_state_without_wait_case,
        source_function=_pdl_transitively_acquired_consumer_without_wait_kernel.fn,
        marker="value = tl.load(payload_ptr)",
        error="Read after write race detected",
    )


def test_mixed_scope_release_rmw_accumulation():
    _run_failure_case(
        "mixed_scope_release_rmw",
        runner=_run_mixed_scope_release_rmw_case,
        source_function=_mixed_scope_release_rmw_kernel.fn,
        marker='tl.atomic_add(counter_ptr, 1, sem="release", scope="sys")',
        error="GSan detected atomic release accumulation with mixed scopes, which is not supported.",
    )


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
def test_relaxed_cluster_barrier_does_not_synchronize_vector_clocks():
    _run_failure_case(
        "relaxed_cluster_barrier",
        runner=_run_gluon_relaxed_cluster_barrier_case,
        source_function=_gluon_cluster_barrier_kernel.fn,
        marker="value = gl.load(payload_ptr",
        error="Read after write race detected",
    )


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
@pytest.mark.parametrize("after_arrival", [False, True], ids=["before-arrival", "after-arrival"])
def test_mbarrier_wait_does_not_acquire_current_epoch(after_arrival):
    _run_failure_case(
        f"mbarrier_current_epoch_{after_arrival}",
        runner=_run_gluon_mbarrier_current_epoch_case,
        runner_args=(after_arrival, ),
        source_function=_gluon_mbarrier_current_epoch_kernel.fn,
        marker="value = gl.load(payload_ptrs",
        error="Read after write race detected",
    )


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tcgen05_commit_does_not_publish_vector_clock():
    _run_failure_case(
        "tcgen05_commit_no_release",
        runner=_run_gluon_tcgen05_commit_no_release_case,
        source_function=_gluon_tcgen05_commit_no_release_kernel.fn,
        marker="value = gl.load(payload_ptrs",
        error="Read after write race detected",
    )


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_mbarrier_multicast_partial_wait_only_acquires_on_leader_ctas():
    _run_failure_case(
        "mbarrier_multicast_partial_wait",
        runner=_run_gluon_mbarrier_multicast_partial_wait_case,
        source_function=_gluon_mbarrier_multicast_partial_wait_kernel.fn,
        marker="value = gl.load(payload_ptrs",
        error="Read after write race detected",
    )


def test_tma_read_after_write():
    _run_failure_case("tma_raw", runner=_run_tma_raw_case, source_function=_tma_raw_kernel.fn,
                      marker="value = tl.load(ptr + row_idx * stride_0 + col_idx)",
                      error="Read after write race detected")


def test_host_tma_write_after_read():
    _run_failure_case("host_tma_war", runner=_run_host_tma_war_case, source_function=_host_tma_war_kernel.fn,
                      marker="tl.store(target_ptr + row_idx * stride_0 + col_idx, 1)",
                      error="Write after read race detected")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_host_tma_gather_write_after_read():
    _run_failure_case("host_tma_gather_war", runner=_run_host_tma_gather_war_case,
                      source_function=_host_tma_gather_war_kernel.fn,
                      marker="tl.store(target_ptr + row_idx * stride_0 + y_offset, 1)",
                      error="Write after read race detected")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_host_tma_scatter_write_after_read():
    _run_failure_case("host_tma_scatter_war", runner=_run_host_tma_scatter_war_case,
                      source_function=_host_tma_scatter_war_kernel.fn,
                      marker="target_desc.scatter(values, x_offsets, y_offset)", error="Write after read race detected")


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires Hopper or newer")
def test_host_tma_atomic_on_release_flag_does_not_publish_data():
    _run_failure_case("host_tma_atomic_flag_publish", runner=_run_host_tma_atomic_flag_publish_case,
                      source_function=_host_tma_atomic_flag_publish_kernel.fn, marker="result = tl.load(payload_ptr)",
                      error="Read after write race detected")


@pytest.mark.parametrize("producer_sem, consumer_sem, scope", CROSS_SM_SEMANTIC_MISMATCH_CASES)
def test_cross_sm_semantic_mismatch_read_after_write(producer_sem, consumer_sem, scope):
    _run_failure_case(f"cross_sm_semantic_mismatch_{producer_sem}_{consumer_sem}_{scope}",
                      runner=_run_cross_sm_atomic_sync_case, runner_args=(producer_sem, consumer_sem, scope),
                      source_function=_cross_sm_atomic_sync_kernel.fn, marker="result = tl.load(payload_ptr)",
                      error="Read after write race detected")


@pytest.mark.parametrize("producer_sem, consumer_sem, scope", CROSS_SM_SEMANTIC_MISMATCH_CASES)
def test_atomic_poll_semantic_mismatch_read_after_write(producer_sem, consumer_sem, scope):
    _run_failure_case(f"atomic_poll_semantic_mismatch_{producer_sem}_{consumer_sem}_{scope}",
                      runner=_run_atomic_poll_cross_sm_sync_case, runner_args=(producer_sem, consumer_sem, scope),
                      source_function=_atomic_poll_cross_sm_sync_kernel.fn, marker="result = tl.load(payload_ptr)",
                      error="Read after write race detected")


@pytest.mark.parametrize("producer_sem, consumer_sem, scope", CROSS_SM_SEMANTIC_MISMATCH_CASES)
def test_atomic_load_store_semantic_mismatch_read_after_write(producer_sem, consumer_sem, scope):
    _run_failure_case(f"atomic_load_store_semantic_mismatch_{producer_sem}_{consumer_sem}_{scope}",
                      runner=_run_atomic_load_store_cross_sm_sync_case, runner_args=(producer_sem, consumer_sem, scope),
                      source_function=_atomic_load_store_cross_sm_sync_kernel.fn,
                      marker="result = tl.load(payload_ptr)", error="Read after write race detected")


@pytest.mark.parametrize("producer_sem, consumer_sem", RELEASE_ACQUIRE_SYNC_CASES)
def test_cross_sm_cta_scope_read_after_write(producer_sem, consumer_sem):
    _run_failure_case(f"cross_sm_cta_scope_{producer_sem}_{consumer_sem}", runner=_run_cross_sm_atomic_sync_case,
                      runner_args=(producer_sem, consumer_sem, "cta"), source_function=_cross_sm_atomic_sync_kernel.fn,
                      marker="ready = tl.atomic_add(flag_ptr, 0, sem=consumer_sem, scope=scope)",
                      error="Read after write race detected")


@pytest.mark.parametrize("release_sem, relay_sem, scope", TRANSITIVE_RELAY_MISMATCH_CASES)
def test_transitive_release_acquire_requires_middle_acquire(release_sem, relay_sem, scope):
    _run_failure_case(f"transitive_sync_{release_sem}_{relay_sem}_{scope}", runner=_run_transitive_atomic_sync_case,
                      runner_args=(release_sem, relay_sem, scope), source_function=_transitive_atomic_sync_kernel.fn,
                      marker="result = tl.load(payload_ptr)", error="Read after write race detected")


@pytest.mark.parametrize("release_sem, relay_sem", RELEASE_ACQUIRE_SYNC_CASES)
def test_transitive_cta_scope_read_after_write(release_sem, relay_sem):
    _run_failure_case(f"transitive_cta_scope_{release_sem}_{relay_sem}", runner=_run_transitive_atomic_sync_case,
                      runner_args=(release_sem, relay_sem, "cta"), source_function=_transitive_atomic_sync_kernel.fn,
                      marker="ready = tl.atomic_add(flag0_ptr, 0, sem=relay_sem, scope=scope)",
                      error="Read after write race detected")


@gluon.jit
def _published_map_race(storage, template, data, counter, output, KIND: gl.constexpr):
    if gl.program_id(0) == 0:
        if KIND == "payload":
            gl.store(data + 64, 1)
        else:
            # An access to the last word must conflict with the whole map.
            word = gl.load(storage + 124)
            gl.store(output, word)
        gl.atomic_add(counter, 1, sem="relaxed")
    else:
        gl.atomic_poll(counter, 1, sem="relaxed")
        if KIND == "publication":
            hopper.tma.publish_tensor_descriptor(storage, template, data, [16, 128], [128, 1])
        else:
            layout: gl.constexpr = gl.NVMMASharedLayout(128, 8, rank=2, fp4_padded=True)
            desc = hopper.tma.load_tensor_descriptor(storage, [16, 128], [128, 1], [16, 64], gl.uint8, layout)
            shared = gl.allocate_shared_memory(gl.uint8, [16, 64], layout)
            bar = hopper.mbarrier.allocate_mbarrier()
            hopper.mbarrier.init(bar, count=1)
            hopper.mbarrier.expect(bar, desc.nbytes_per_cta)
            hopper.tma.async_load(desc, [0, 64], bar, shared)
            hopper.mbarrier.wait(bar, 0)
            hopper.mbarrier.invalidate(bar)
            regs: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
            values = shared.load(regs).to(gl.int32)
            gl.store(output, gl.sum(gl.sum(values, 0), 0))


@gluon.jit
def _initialize_published_map(storage, template, data):
    hopper.tma.publish_tensor_descriptor(storage, template, data, [16, 128], [128, 1])


def _run_published_map_race(kind):
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonTensorDescriptor

    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        data = torch.zeros((16, 128), device="cuda", dtype=torch.uint8)
        storage = torch.empty(128, device="cuda", dtype=torch.uint8)
        counter = torch.zeros((), device="cuda", dtype=torch.int32)
        output = torch.empty(1, device="cuda", dtype=torch.uint8)
        layout = gl.NVMMASharedLayout(128, 8, rank=2, fp4_padded=True)
        template = GluonTensorDescriptor.from_tensor(data, [16, 64], layout)
        _initialize_published_map[(1, )](storage, template, data)
        _published_map_race[(2, )](storage, template, data, counter, output, kind)
        torch.cuda.synchronize()


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind", ["publication", "payload"])
def test_published_tensor_map_race(kind):
    _run_failure_case(
        kind,
        runner=_run_published_map_race,
        runner_args=(kind, ),
        source_function=_published_map_race.fn,
        marker="hopper.tma.publish_tensor_descriptor" if kind == "publication" else "hopper.tma.async_load",
        error="Write after read race detected" if kind == "publication" else "Read after write race detected",
    )


@gluon.jit
def _tensor_map_load(desc, output):
    tile = gl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout)
    bar = hopper.mbarrier.allocate_mbarrier()
    hopper.mbarrier.init(bar, count=1)
    hopper.mbarrier.expect(bar, desc.nbytes_per_cta)
    hopper.tma.async_load(desc, [0, 0], bar, tile)
    hopper.mbarrier.wait(bar, 0)
    hopper.mbarrier.invalidate(bar)
    regs: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0], cga_layout=desc.layout.cga_layout)
    rows = gl.arange(0, 16, gl.SliceLayout(1, regs))
    cols = gl.arange(0, 64, gl.SliceLayout(0, regs))
    gl.store(output + rows[:, None] * 64 + cols[None, :], tile.load(regs))


@gluon.jit
def _tensor_map_republish(storage, template, first, second, output, FRESH: gl.constexpr, RAW: gl.constexpr):
    layout: gl.constexpr = gl.NVMMASharedLayout(128, 16, rank=2)
    hopper.tma.publish_tensor_descriptor(storage, template, first, [16, 64], [64, 1])
    desc = hopper.tma.load_tensor_descriptor(storage, [16, 64], [64, 1], [16, 64], gl.float16, layout)
    _tensor_map_load(desc, output)
    if RAW:
        # Even a same-value generic write invalidates the proxy publication.
        word = (storage + 124).cast(gl.pointer_type(gl.int32))
        if RAW == 2:
            gl.atomic_xchg(word, gl.load(word), sem="relaxed")
        else:
            gl.store(word, gl.load(word))
    else:
        hopper.tma.publish_tensor_descriptor(storage, template, second, [16, 64], [64, 1])
    if FRESH:
        # An acquire refreshes the map for existing handles too.
        hopper.tma.load_tensor_descriptor(storage, [16, 64], [64, 1], [16, 64], gl.float16, layout)
    # The first transfer is complete, so this isolates descriptor acquisition.
    _tensor_map_load(desc, output + 1024)


@gluon.jit
def _tensor_map_publish(storage, template, data):
    hopper.tma.publish_tensor_descriptor(storage, template, data, [16, 64], [64, 1])
    tl.extra.cuda.gdc_launch_dependents()


@gluon.jit
def _tensor_map_copy(source, destination):
    offsets = gl.arange(0, 128, layout=gl.BlockedLayout([1], [32], [4], [0]))
    gl.store(destination + offsets, gl.load(source + offsets))


@gluon.jit
def _tensor_map_consume(storage, output, PDL: gl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
    layout: gl.constexpr = gl.NVMMASharedLayout(128, 16, rank=2)
    desc = hopper.tma.load_tensor_descriptor(storage, [16, 64], [64, 1], [16, 64], gl.float16, layout)
    _tensor_map_load(desc, output + gl.program_id(0) * 1024)


@gluon.jit(noinline=True)
def _tensor_map_consume_noinline(storage, output):
    _tensor_map_consume(storage, output)


@gluon.jit
def _tensor_map_pdl_publish(storage, template, data):
    hopper.tma.publish_tensor_descriptor(storage + 128 * gl.program_id(0), template, data, [16, 64], [64, 1])
    tl.extra.cuda.gdc_launch_dependents()


@gluon.jit
def _tensor_map_pdl_consume(storage, output, EARLY: gl.constexpr):
    if not EARLY:
        tl.extra.cuda.gdc_wait()
    layout: gl.constexpr = gl.NVMMASharedLayout(128, 16, rank=2)
    # Neighboring producer CTAs exercise the existing inter-SM launch clocks.
    pointer = storage + 128 * ((gl.program_id(0) + 1) % gl.num_programs(0))
    desc = hopper.tma.load_tensor_descriptor(pointer, [16, 64], [64, 1], [16, 64], gl.float16, layout)
    if EARLY:
        tl.extra.cuda.gdc_wait()
    _tensor_map_load(desc, output + 1024 * gl.program_id(0))


@gluon.jit
def _tensor_map_other_cta(storage, output, ready):
    layout: gl.constexpr = gl.NVMMASharedLayout(128, 16, rank=2)
    # The first import is removed before instrumentation in the negative case.
    desc = hopper.tma.load_tensor_descriptor(storage, [16, 64], [64, 1], [16, 64], gl.float16, layout)
    if gl.program_id(0) == 0:
        desc = hopper.tma.load_tensor_descriptor(storage, [16, 64], [64, 1], [16, 64], gl.float16, layout)
        gl.atomic_xchg(ready, 1, sem="release")
    else:
        gl.atomic_poll(ready, 1, sem="acquire")
    _tensor_map_load(desc, output + gl.program_id(0) * 1024)


@gluon.jit
def _tensor_map_idle():
    pass


@gluon.jit
def _tensor_map_ws_acquire(storage, desc, output, bar, ACQUIRE_HERE: gl.constexpr, EARLY: gl.constexpr):
    if ACQUIRE_HERE:
        if EARLY:
            hopper.mbarrier.arrive(bar, count=1)
        desc = hopper.tma.load_tensor_descriptor(storage, [16, 64], [64, 1], [16, 64], gl.float16, desc.layout)
        if not EARLY:
            hopper.mbarrier.arrive(bar, count=1)
    else:
        hopper.mbarrier.wait(bar, 0)
        _tensor_map_load(desc, output)


@gluon.jit
def _tensor_map_ws(storage, output, HANDOFF: gl.constexpr, EARLY: gl.constexpr = False, CTAS: gl.constexpr = 1,
                   NOINLINE: gl.constexpr = False):
    layout: gl.constexpr = gl.NVMMASharedLayout(128, 16, rank=2, cga_layout=[] if CTAS == 1 else [[0, 0]])
    if HANDOFF:
        # Each CTA transfers its own acquisition between its execution regions.
        bar = hopper.mbarrier.allocate_mbarrier()
        hopper.mbarrier.init(bar, count=1)
        # This first import is removed in the handoff test. Both regions retain
        # the same descriptor pointer, and only the publishing region acquires.
        desc = hopper.tma.load_tensor_descriptor(storage, [16, 64], [64, 1], [16, 64], gl.float16, layout)
        gl.warp_specialize([
            (_tensor_map_ws_acquire, (storage, desc, output, bar, True, EARLY)),
            (_tensor_map_ws_acquire, (storage, desc, output, bar, False, EARLY)),
        ], [4])
        hopper.mbarrier.invalidate(bar)
    elif NOINLINE:
        gl.warp_specialize([(_tensor_map_idle, ()), (_tensor_map_consume_noinline, (storage, output))], [4])
    else:
        gl.warp_specialize([(_tensor_map_idle, ()), (_tensor_map_consume, (storage, output, False))], [4])


def _without_tensor_map_acquire(kernel, directory, *, first_only=False):
    """Remove typed fences before GSan; retain ordinary operations and locations."""
    source = kernel.asm["ttgir"]
    lines = source.splitlines()
    count = 0
    kept = []
    for line in lines:
        if "ttng.tensormap_fenceproxy_acquire " in line and (count == 0 or not first_only):
            count += 1
        else:
            kept.append(line)
    assert count > 0
    path = Path(directory) / "missing-tensor-map-acquire.ttgir"
    path.write_text("\n".join(kept) + "\n")
    return triton.compile(str(path), options={"instrumentation_mode": "gsan", "num_ctas": kernel.metadata.num_ctas})


def _run_tensor_map_protocol(case):
    import tempfile
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonTensorDescriptor

    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool), tempfile.TemporaryDirectory() as directory:
        first = torch.full((16, 64), 1, device="cuda", dtype=torch.float16)
        second = torch.full_like(first, 2)
        storage = torch.empty(128, device="cuda", dtype=torch.uint8)
        output = torch.empty((2, 16, 64), device="cuda", dtype=torch.float16)
        layout = gl.NVMMASharedLayout(128, 16, rank=2)
        template = GluonTensorDescriptor.from_tensor(first, [16, 64], layout)
        if case in ("pdl_grid", "pdl_early"):
            storage = torch.empty(128 * 128, device="cuda", dtype=torch.uint8)
            output = torch.empty((128, 16, 64), device="cuda", dtype=torch.float16)
            _tensor_map_pdl_publish[(128, )](storage, template, first)
            _tensor_map_pdl_consume[(128, )](storage, output, case == "pdl_early", launch_pdl=True)
            torch.testing.assert_close(output, first.expand_as(output))
        elif case in ("stale", "fresh", "raw_write", "raw_atomic"):
            _tensor_map_republish[(1, )](storage, template, first, second, output, case == "fresh",
                                         2 if case == "raw_atomic" else int(case == "raw_write"))
            torch.testing.assert_close(output[0], first)
            torch.testing.assert_close(output[1], second)
        else:
            _tensor_map_publish[(1, )](storage, template, first)
            if case == "raw_copy":
                destination = torch.empty_like(storage)
                _tensor_map_copy[(1, )](storage, destination)
                storage = destination
            if case == "raw_copy":
                _tensor_map_consume[(2, )](storage, output)
            elif case == "other_cta":
                ready = torch.zeros((), device="cuda", dtype=torch.int32)
                kernel = _tensor_map_other_cta.warmup(storage, output, ready, grid=(2, ), instrumentation_mode="")
                kernel = _without_tensor_map_acquire(kernel, directory, first_only=True)
                kernel[(2, 1, 1)](storage, output, ready)
            elif case in ("missing", "previous_kernel"):
                if case == "previous_kernel":
                    _tensor_map_consume[(2, )](storage, output)
                kernel = _tensor_map_consume.warmup(storage, output, False, grid=(2, ), instrumentation_mode="")
                kernel = _without_tensor_map_acquire(kernel, directory)
                kernel[(2, 1, 1)](storage, output)
            elif case in ("worker", "worker_noinline", "handoff", "cluster_handoff", "early_handoff"):
                if case not in ("worker", "worker_noinline"):
                    ctas = 2 if case == "cluster_handoff" else 1
                    kernel = _tensor_map_ws.warmup(storage, output, True, case == "early_handoff", ctas, grid=(1, ),
                                                   num_ctas=ctas, instrumentation_mode="")
                    kernel = _without_tensor_map_acquire(kernel, directory, first_only=True)
                    kernel[(1, 1, 1)](storage, output)
                else:
                    _tensor_map_ws[(1, )](storage, output, False, NOINLINE=case == "worker_noinline")
                torch.testing.assert_close(output[0], first)
            else:
                for data in (first, second, first):
                    _tensor_map_publish[(1, )](storage, template, data)
                    _tensor_map_consume[(2, )](storage, output, case == "pdl", launch_pdl=case == "pdl")
                    torch.testing.assert_close(output, data.expand_as(output))
        torch.cuda.synchronize()


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper")
@pytest.mark.parametrize(
    "case", ["fresh", "ordered", "pdl", "pdl_grid", "worker", "worker_noinline", "handoff", "cluster_handoff"])
def test_tensor_map_proxy_protocol_valid(case):
    result = run_in_process(_run_tensor_map_protocol, (case, ))
    print(result.driver_stderr_output)
    assert result.exc is None, (result.exc, result.driver_stderr_output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper")
@pytest.mark.parametrize("case", [
    "stale", "raw_write", "raw_atomic", "raw_copy", "missing", "previous_kernel", "early_handoff", "other_cta",
    "pdl_early"
])
def test_tensor_map_proxy_protocol_invalid(case):
    result = run_in_process(_run_tensor_map_protocol, (case, ))
    print(result.driver_stderr_output)
    assert isinstance(result.exc, RuntimeError), (result.exc, result.driver_stderr_output)
    error = "proxy release after its last write" if case.startswith(
        "raw_") else "acquiring its current version in this CTA"
    if case == "pdl_early":
        error = "Read after write race detected"
    assert error in result.driver_stderr_output
    assert "GSanLibrary.cu" not in result.driver_stderr_output
    assert Path(__file__).name in result.driver_stderr_output
