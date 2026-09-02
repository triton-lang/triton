from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia import hopper
from triton.tools.tensor_descriptor import TensorDescriptor

from triton._internal_testing import is_blackwell, is_cuda, is_ampere_or_newer, is_hopper_or_newer, is_sm12x
from triton.experimental.gsan import create_mem_pool
from triton._C.libtriton.gsan_testing import AtomicScope, SHADOW_GRANULARITY_BYTES, ScalarClock
from triton.experimental.gsan._testing_utils import (atomic_poll, load_one_i32, shadow_cell_from_address, store_one_i32,
                                                     thread_state_from_smid)


@pytest.fixture()
def with_gsan(fresh_knobs):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        yield


def _clock_buffer_snapshot_idx(token: int, state, tid: int) -> int:
    return (token % state.clock_buffer_size) * state.num_threads + tid


ATOMIC_SCOPE_CASES = (
    pytest.param("cta", AtomicScope.CTA, id="scope-cta"),
    pytest.param("gpu", AtomicScope.GPU, id="scope-gpu"),
    pytest.param("sys", AtomicScope.SYSTEM, id="scope-sys"),
)

ATOMIC_SEMANTIC_CASES = (
    pytest.param("relaxed", False, id="sem-relaxed"),
    pytest.param("acquire", False, id="sem-acquire"),
    pytest.param("release", True, id="sem-release"),
    pytest.param("acq_rel", True, id="sem-acq-rel"),
)

RELEASE_SEMANTIC_CASES = (
    pytest.param("release", id="sem-release"),
    pytest.param("acq_rel", id="sem-acq-rel"),
)

ACQUIRE_SEMANTIC_CASES = (
    pytest.param("acquire", id="sem-acquire"),
    pytest.param("acq_rel", id="sem-acq-rel"),
)


def _assert_atomic_rmw_shadow(real_address: int, expected_scope: AtomicScope, *, is_release: bool) -> None:
    cell = shadow_cell_from_address(real_address)
    tid = cell.write_clock.thread_id
    state = thread_state_from_smid(tid)

    if is_release:
        token = cell.write_clock.epoch
        snapshot_idx = _clock_buffer_snapshot_idx(token, state, tid)
        published_epoch = state.clock_buffer[snapshot_idx]

        assert cell.write_clock == ScalarClock(token, tid, expected_scope, is_release=True)
        assert token == state.clock_buffer_head
        assert state.clock_buffer_dirty
        assert cell.read_clocks[0] == ScalarClock(published_epoch, tid, expected_scope)
        assert state.vector_clock[tid] == published_epoch + 1
    else:
        epoch = state.vector_clock[tid]
        assert cell.write_clock == ScalarClock(epoch, tid, expected_scope)
        assert cell.read_clocks[0] == ScalarClock(epoch, tid, expected_scope)

    assert cell.num_reads == 1


def _assert_atomic_read_only_shadow(real_address: int, expected_scope: AtomicScope) -> None:
    cell = shadow_cell_from_address(real_address)
    tid = cell.read_clocks[0].thread_id
    epoch = thread_state_from_smid(tid).vector_clock[tid]

    assert cell.write_clock == ScalarClock(0, 0, AtomicScope.NON_ATOMIC)
    assert cell.read_clocks[0] == ScalarClock(epoch, tid, expected_scope)
    assert cell.num_reads == 1


def _assert_cross_sm_sync(payload_ptr: torch.Tensor, flag_ptr: torch.Tensor, expected_scope: AtomicScope) -> None:
    payload_cell = shadow_cell_from_address(payload_ptr.data_ptr())
    flag_cell = shadow_cell_from_address(flag_ptr.data_ptr())
    producer_tid = payload_cell.write_clock.thread_id
    producer_epoch = payload_cell.write_clock.epoch
    consumer_tid = payload_cell.read_clocks[0].thread_id
    consumer_state = thread_state_from_smid(consumer_tid)

    assert flag_cell.write_clock.scope == expected_scope
    assert flag_cell.write_clock.is_release
    assert consumer_state.vector_clock[producer_tid] >= producer_epoch


def _assert_no_gsan_runtime_output(capfd) -> None:
    captured = capfd.readouterr()
    assert "GSanLibrary.cu" not in captured.out + captured.err


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
def test_load_store_updates_shadow(with_gsan):
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")

    store_one_i32[(1, )](target, num_warps=1)
    cell0 = shadow_cell_from_address(target.data_ptr())

    tid = cell0.write_clock.thread_id
    epoch0 = thread_state_from_smid(tid).vector_clock[tid]

    assert cell0.write_clock.thread_id == tid
    assert cell0.write_clock.epoch == epoch0
    assert cell0.read_clocks[0].thread_id == 0
    assert cell0.read_clocks[0].epoch == 0
    assert cell0.num_reads == 0

    load_one_i32[(1, )](target, scratch, num_warps=1)
    cell1 = shadow_cell_from_address(target.data_ptr())
    epoch1 = thread_state_from_smid(tid).vector_clock[tid]

    assert epoch1 == epoch0 + 1
    assert cell1.write_clock == cell0.write_clock
    assert cell1.read_clocks[0] == ScalarClock(epoch1, tid, AtomicScope.NON_ATOMIC)
    # Scalar accesses are instrumented once via the redundant-thread predicate.
    assert cell1.num_reads == 1


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
def test_cuda_graph_capture_is_rejected(with_gsan):
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros_like(target)
    load_one_i32[(1, )](target, scratch, num_warps=1)
    torch.cuda.synchronize()

    cuda_utils = triton.runtime.driver.active.utils
    assert not cuda_utils.is_stream_capturing(torch.cuda.current_stream().cuda_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scratch.zero_()
        assert cuda_utils.is_stream_capturing(torch.cuda.current_stream().cuda_stream)
        with pytest.raises(RuntimeError, match="GSan does not support CUDA graph capture"):
            load_one_i32[(1, )](target, scratch, num_warps=1)

    assert not cuda_utils.is_stream_capturing(torch.cuda.current_stream().cuda_stream)


@triton.jit
def _pdl_producer_kernel(payload_ptr):
    pid = tl.program_id(0)
    tl.store(payload_ptr + pid, 1000 + pid)
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _pdl_consumer_kernel(payload_ptr, result_ptr):
    tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    value = tl.load(payload_ptr + pid)
    tl.store(result_ptr + pid, value)


@triton.jit
def _pdl_stage_kernel(payload_ptr, INDEX: tl.constexpr, WAIT_PREDECESSOR: tl.constexpr):
    if WAIT_PREDECESSOR:
        tl.extra.cuda.gdc_wait()
    tl.store(payload_ptr + INDEX, 1000 + INDEX)
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _pdl_two_back_consumer_kernel(payload_ptr, result_ptr):
    value = tl.load(payload_ptr)
    tl.store(result_ptr, value)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="PDL requires SM90 or newer")
def test_programmatic_dependent_launch_wait_synchronizes_vector_clocks(with_gsan, capfd):
    payload = torch.zeros(2, dtype=torch.int32, device="cuda")
    result = torch.full_like(payload, -1)

    _pdl_producer_kernel[(2, )](payload, num_warps=1)
    compiled = _pdl_consumer_kernel[(2, )](payload, result, num_warps=1, launch_pdl=True)
    torch.cuda.synchronize()

    torch.testing.assert_close(result, torch.tensor([1000, 1001], device="cuda", dtype=torch.int32))
    assert "griddepcontrol.wait" in compiled.asm["ptx"]
    assert "red.relaxed.gpu.max.u32" in compiled.asm["ptx"]
    _assert_no_gsan_runtime_output(capfd)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="PDL requires SM90 or newer")
def test_normal_launch_after_programmatic_dependent_launch_acquires_all_predecessors(with_gsan, capfd):
    payload = torch.zeros(2, dtype=torch.int32, device="cuda")
    result = torch.full((1, ), -1, dtype=torch.int32, device="cuda")

    _pdl_stage_kernel[(1, )](payload, INDEX=0, WAIT_PREDECESSOR=False, num_warps=1)
    _pdl_stage_kernel[(1, )](payload, INDEX=1, WAIT_PREDECESSOR=True, num_warps=1, launch_pdl=True)
    _pdl_two_back_consumer_kernel[(1, )](payload, result, num_warps=1)
    torch.cuda.synchronize()

    torch.testing.assert_close(result, torch.tensor([1000], dtype=torch.int32, device="cuda"))
    _assert_no_gsan_runtime_output(capfd)


@gluon.jit
def _gluon_ws_completion_default(out_ptr, layout: gl.constexpr):
    offsets = gl.arange(0, 128, layout=layout)
    gl.store(out_ptr + offsets, offsets)


@gluon.jit
def _gluon_ws_completion_worker(out_ptr, layout: gl.constexpr):
    offsets = 128 + gl.arange(0, 128, layout=layout)
    gl.store(out_ptr + offsets, offsets)


@gluon.jit
def _gluon_ws_pdl_wait_default(payload_ptr, result_ptr, layout: gl.constexpr):
    pass


@gluon.jit
def _gluon_ws_pdl_wait_worker(payload_ptr, result_ptr, layout: gl.constexpr):
    tl.extra.cuda.gdc_wait()
    offsets = gl.arange(0, 128, layout=layout)
    values = gl.load(payload_ptr + offsets)
    gl.store(result_ptr + offsets, values)


@gluon.jit(noinline=True)
def _gluon_ws_pdl_wait_noinline_worker(payload_ptr, result_ptr, layout: gl.constexpr):
    tl.extra.cuda.gdc_wait()
    offsets = gl.arange(0, 128, layout=layout)
    values = gl.load(payload_ptr + offsets)
    gl.store(result_ptr + offsets, values)


@gluon.jit
def _gluon_ws_pdl_wait_kernel(payload_ptr, result_ptr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    gl.warp_specialize([
        (_gluon_ws_pdl_wait_default, (payload_ptr, result_ptr, layout)),
        (_gluon_ws_pdl_wait_worker, (payload_ptr, result_ptr, layout)),
    ], [4], [24])


@gluon.jit
def _gluon_ws_pdl_wait_noinline_kernel(payload_ptr, result_ptr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    gl.warp_specialize([
        (_gluon_ws_pdl_wait_default, (payload_ptr, result_ptr, layout)),
        (_gluon_ws_pdl_wait_noinline_worker, (payload_ptr, result_ptr, layout)),
    ], [4], [24])


@gluon.jit
def _gluon_ws_completion_kernel(out_ptr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    gl.warp_specialize([
        (_gluon_ws_completion_default, (out_ptr, layout)),
        (_gluon_ws_completion_worker, (out_ptr, layout)),
    ], [4], [24])


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
def test_gluon_warp_specialize_completes(with_gsan):
    expected = torch.arange(256, dtype=torch.int32, device="cuda")

    out = torch.full((256, ), -1, dtype=torch.int32, device="cuda")
    _gluon_ws_completion_kernel[(1, )](out, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="PDL requires SM90 or newer")
def test_programmatic_dependent_launch_wait_inside_warp_specialize(with_gsan, capfd):
    payload = torch.empty((128, ), dtype=torch.int32, device="cuda")
    result = torch.full_like(payload, -1)

    _pdl_producer_kernel[(128, )](payload, num_warps=1)
    compiled = _gluon_ws_pdl_wait_kernel[(1, )](payload, result, num_warps=4, launch_pdl=True)
    torch.cuda.synchronize()

    torch.testing.assert_close(result, torch.arange(1000, 1128, device="cuda", dtype=torch.int32))
    assert "griddepcontrol.wait" in compiled.asm["ptx"]
    _assert_no_gsan_runtime_output(capfd)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="PDL requires SM90 or newer")
def test_programmatic_dependent_launch_wait_inside_noinline_warp_specialize(with_gsan, capfd):
    payload = torch.empty((128, ), dtype=torch.int32, device="cuda")
    result = torch.full_like(payload, -1)

    _pdl_producer_kernel[(128, )](payload, num_warps=1)
    compiled = _gluon_ws_pdl_wait_noinline_kernel[(1, )](payload, result, num_warps=4, launch_pdl=True)
    torch.cuda.synchronize()

    torch.testing.assert_close(result, torch.arange(1000, 1128, device="cuda", dtype=torch.int32))
    assert "tt.call" in compiled.asm["ttgir"]
    assert "griddepcontrol.wait" in compiled.asm["ptx"]
    _assert_no_gsan_runtime_output(capfd)


@gluon.jit
def _gluon_ws_noinline_default(out_ptr, layout: gl.constexpr):
    pass


@gluon.jit(noinline=True)
def _gluon_ws_noinline_worker(out_ptr, layout: gl.constexpr):
    pass


@gluon.jit
def _gluon_ws_noinline_kernel(out_ptr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])
    gl.warp_specialize(
        [
            (_gluon_ws_noinline_default, (out_ptr, layout)),
            (_gluon_ws_noinline_worker, (out_ptr, layout)),
        ],
        [4],
        [24],
    )


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
def test_gluon_two_cta_warp_specialize_noinline_call(with_gsan):
    out = torch.full((256, ), -1, dtype=torch.int32, device="cuda")

    compiled = _gluon_ws_noinline_kernel[(1, )](out, num_warps=4, num_ctas=2)
    assert "tt.call" in compiled.asm["ttgir"]
    torch.cuda.synchronize()
    assert torch.all(out == -1).item()


@gluon.jit
def _gluon_cluster_barrier_sync_kernel(payload_ptr, out_ptr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])

    offsets = gl.arange(0, 256, data_layout)
    payload0_ptrs = payload_ptr + offsets * 0
    payload1_ptrs = payload_ptr + 1 + offsets * 0
    out0_ptrs = out_ptr + offsets * 0
    out1_ptrs = out_ptr + 1 + offsets * 0
    gl.store(payload0_ptrs, 1, mask=offsets == 0)
    gl.store(payload1_ptrs, 2, mask=offsets == 128)
    gl.barrier(cluster=True)
    value0 = gl.load(payload0_ptrs, mask=offsets == 128, other=0)
    value1 = gl.load(payload1_ptrs, mask=offsets == 0, other=0)
    gl.store(out0_ptrs, value0, mask=offsets == 128)
    gl.store(out1_ptrs, value1, mask=offsets == 0)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
def test_gluon_cluster_barrier_synchronizes_vector_clocks(with_gsan):
    payload = torch.zeros(2, dtype=torch.int32, device="cuda")
    out = torch.full((2, ), -1, dtype=torch.int32, device="cuda")

    _gluon_cluster_barrier_sync_kernel[(1, )](payload, out, num_warps=4, num_ctas=2)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, torch.tensor([1, 2], dtype=torch.int32, device="cuda"))
    for offset in range(2):
        payload_cell = shadow_cell_from_address(payload.data_ptr() + offset * payload.element_size())
        producer_tid = payload_cell.write_clock.thread_id
        producer_epoch = payload_cell.write_clock.epoch
        consumer_tid = payload_cell.read_clocks[0].thread_id
        assert consumer_tid != producer_tid
        consumer_state = thread_state_from_smid(consumer_tid)
        assert consumer_state.vector_clock[producer_tid] >= producer_epoch


@gluon.jit
def _gluon_atomic_cluster_sync_kernel(payload_ptr, counter_ptr, out_ptr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])

    offsets = gl.arange(0, 256, data_layout)
    payload_ptrs = payload_ptr + offsets * 0
    out_ptrs = out_ptr + offsets * 0
    gl.store(payload_ptrs, 1, mask=offsets == 0)
    gl.atomic_add(counter_ptr, 1, sem="release", scope="gpu")
    value = gl.load(payload_ptrs, mask=offsets == 128, other=0)
    gl.store(out_ptrs, value, mask=offsets == 128)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
def test_gluon_cluster_synchronizing_atomic_synchronizes_vector_clocks(with_gsan):
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")

    _gluon_atomic_cluster_sync_kernel[(1, )](payload, counter, out, num_warps=4, num_ctas=2)
    torch.cuda.synchronize()

    assert counter.item() == 1
    assert out.item() == 1
    payload_cell = shadow_cell_from_address(payload.data_ptr())
    producer_tid = payload_cell.write_clock.thread_id
    producer_epoch = payload_cell.write_clock.epoch
    consumer_tid = payload_cell.read_clocks[0].thread_id
    assert consumer_tid != producer_tid
    consumer_state = thread_state_from_smid(consumer_tid)
    assert consumer_state.vector_clock[producer_tid] >= producer_epoch


@gluon.jit
def _gluon_ws_cluster_barrier_partition(payload_ptr, out_ptr, partition_offset: gl.constexpr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])

    offsets = gl.arange(0, 256, data_layout)
    payload0_ptrs = payload_ptr + partition_offset + offsets * 0
    payload1_ptrs = payload_ptr + partition_offset + 1 + offsets * 0
    out0_ptrs = out_ptr + partition_offset + offsets * 0
    out1_ptrs = out_ptr + partition_offset + 1 + offsets * 0
    gl.store(payload0_ptrs, partition_offset + 1, mask=offsets == 0)
    gl.store(payload1_ptrs, partition_offset + 2, mask=offsets == 128)
    gl.barrier(cluster=True)
    value0 = gl.load(payload0_ptrs, mask=offsets == 128, other=0)
    value1 = gl.load(payload1_ptrs, mask=offsets == 0, other=0)
    gl.store(out0_ptrs, value0, mask=offsets == 128)
    gl.store(out1_ptrs, value1, mask=offsets == 0)


@gluon.jit
def _gluon_ws_cluster_barrier_kernel(payload_ptr, out_ptr):
    gl.warp_specialize([
        (_gluon_ws_cluster_barrier_partition, (payload_ptr, out_ptr, 0)),
        (_gluon_ws_cluster_barrier_partition, (payload_ptr, out_ptr, 2)),
    ], [4])


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
def test_gluon_cluster_barriers_in_warp_specialize_synchronize_vector_clocks(with_gsan):
    payload = torch.zeros(4, dtype=torch.int32, device="cuda")
    out = torch.full((4, ), -1, dtype=torch.int32, device="cuda")

    _gluon_ws_cluster_barrier_kernel[(1, )](payload, out, num_warps=4, num_ctas=2)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, torch.arange(1, 5, dtype=torch.int32, device="cuda"))
    for offset in range(4):
        payload_cell = shadow_cell_from_address(payload.data_ptr() + offset * payload.element_size())
        producer_tid = payload_cell.write_clock.thread_id
        producer_epoch = payload_cell.write_clock.epoch
        consumer_tid = payload_cell.read_clocks[0].thread_id
        assert consumer_tid != producer_tid
        consumer_state = thread_state_from_smid(consumer_tid)
        assert consumer_state.vector_clock[producer_tid] >= producer_epoch


@gluon.jit
def _gluon_mbarrier_initial_empty_phase_kernel(out_ptr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])
    barrier = hopper.mbarrier.allocate_mbarrier(two_ctas=True)
    hopper.mbarrier.init(barrier, count=1)
    hopper.mbarrier.wait(barrier, phase=1)
    offsets = gl.arange(0, 256, data_layout)
    gl.store(out_ptr + offsets * 0, 1, mask=offsets == 0)
    hopper.mbarrier.invalidate(barrier)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
def test_gluon_mbarrier_initial_empty_phase_is_ready(with_gsan):
    out = torch.zeros(1, dtype=torch.int32, device="cuda")

    compiled = _gluon_mbarrier_initial_empty_phase_kernel[(1, )](out, num_warps=4, num_ctas=2)
    assert "__triton_gsan_mbarrier_wait" in compiled.asm["llir"]
    torch.cuda.synchronize()

    assert out.item() == 1


@gluon.jit
def _gluon_mbarrier_sync_kernel(payload_ptr, counter_ptr, out_ptr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])

    barrier = hopper.mbarrier.allocate_mbarrier(two_ctas=True)
    hopper.mbarrier.init(barrier, count=1)
    hopper.mbarrier.wait(barrier, phase=1)
    offsets = gl.arange(0, 256, data_layout)
    for iteration in range(4):
        payload_ptrs = payload_ptr + iteration + offsets * 0
        out_ptrs = out_ptr + iteration + offsets * 0
        gl.store(payload_ptrs, iteration + 1, mask=offsets == 128)
        # Mbarriers only propagate completed epochs. A release closes the
        # writer's epoch; direct current-epoch handoffs are tested as failures.
        gl.atomic_add(counter_ptr, 1, sem="release", scope="gpu")
        hopper.mbarrier.arrive(barrier, count=1)
        hopper.mbarrier.wait(barrier, phase=iteration % 2)
        value = gl.load(payload_ptrs, mask=offsets == 0, other=0)
        gl.store(out_ptrs, value, mask=offsets == 0)
        hopper.cluster.barrier(relaxed=True)
    hopper.mbarrier.invalidate(barrier)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
def test_gluon_mbarrier_wait_preserves_completed_epochs(with_gsan):
    payload = torch.zeros(4, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((4, ), -1, dtype=torch.int32, device="cuda")

    _gluon_mbarrier_sync_kernel[(1, )](payload, counter, out, num_warps=4, num_ctas=2)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, torch.arange(1, 5, dtype=torch.int32, device="cuda"))
    for offset in range(4):
        payload_cell = shadow_cell_from_address(payload.data_ptr() + offset * payload.element_size())
        producer_tid = payload_cell.write_clock.thread_id
        producer_epoch = payload_cell.write_clock.epoch
        consumer_tid = payload_cell.read_clocks[0].thread_id
        assert consumer_tid != producer_tid
        consumer_state = thread_state_from_smid(consumer_tid)
        assert consumer_state.vector_clock[producer_tid] >= producer_epoch


@gluon.jit
def _gluon_ws_mbarrier_partition(payload_ptr, counter_ptr, out_ptr, barrier, index: gl.constexpr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])

    offsets = gl.arange(0, 256, data_layout)
    payload_ptrs = payload_ptr + index + offsets * 0
    out_ptrs = out_ptr + index + offsets * 0
    gl.store(payload_ptrs, index + 1, mask=offsets == 128)
    gl.atomic_add(counter_ptr + index, 1, sem="release", scope="gpu")
    hopper.mbarrier.arrive(barrier, count=1)
    hopper.mbarrier.wait(barrier, phase=0)
    value = gl.load(payload_ptrs, mask=offsets == 0, other=0)
    gl.store(out_ptrs, value, mask=offsets == 0)


@gluon.jit
def _gluon_ws_mbarrier_sync_kernel(payload_ptr, counter_ptr, out_ptr):
    barriers = hopper.mbarrier.allocate_mbarrier(batch=2, two_ctas=True)
    hopper.mbarrier.init(barriers.index(0), count=1)
    hopper.mbarrier.init(barriers.index(1), count=1)
    gl.warp_specialize([
        (_gluon_ws_mbarrier_partition, (payload_ptr, counter_ptr, out_ptr, barriers.index(0), 0)),
        (_gluon_ws_mbarrier_partition, (payload_ptr, counter_ptr, out_ptr, barriers.index(1), 1)),
    ], [4])
    hopper.mbarrier.invalidate(barriers.index(0))
    hopper.mbarrier.invalidate(barriers.index(1))


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
def test_gluon_mbarrier_sync_in_warp_specialized_partitions(with_gsan):
    payload = torch.zeros(2, dtype=torch.int32, device="cuda")
    counter = torch.zeros(2, dtype=torch.int32, device="cuda")
    out = torch.full((2, ), -1, dtype=torch.int32, device="cuda")

    _gluon_ws_mbarrier_sync_kernel[(1, )](payload, counter, out, num_warps=4, num_ctas=2)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, torch.tensor([1, 2], dtype=torch.int32, device="cuda"))
    for offset in range(2):
        payload_cell = shadow_cell_from_address(payload.data_ptr() + offset * payload.element_size())
        producer_tid = payload_cell.write_clock.thread_id
        producer_epoch = payload_cell.write_clock.epoch
        consumer_tid = payload_cell.read_clocks[0].thread_id
        assert consumer_tid != producer_tid
        consumer_state = thread_state_from_smid(consumer_tid)
        assert consumer_state.vector_clock[producer_tid] >= producer_epoch


@gluon.jit
def _gluon_mbarrier_repeated_phases_kernel(markers, flags, iterations, EXPECT: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])
    offsets = gl.arange(0, 256, layout)
    cta = offsets // 128
    elected = offsets % 128 == 0
    barrier = hopper.mbarrier.allocate_mbarrier(two_ctas=True)
    hopper.mbarrier.init(barrier, count=1)
    # Keep a release snapshot live throughout the loop, as fused-communication
    # completion flags do in production kernels.
    gl.atomic_xchg(flags + cta, 1, mask=elected, sem="release", scope="gpu")
    gl.store(markers + cta, 1, mask=elected)
    for iteration in range(iterations):
        if EXPECT:
            hopper.mbarrier.expect(barrier, 0)
        else:
            hopper.mbarrier.arrive(barrier)
        hopper.mbarrier.wait(barrier, iteration % 2)
        hopper.cluster.barrier(relaxed=True)
    gl.store(markers + 2 + cta, 1, mask=elected)
    hopper.mbarrier.invalidate(barrier)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
@pytest.mark.parametrize("expect", [False, True], ids=["arrive", "expect"])
def test_gluon_mbarrier_repeated_phases_reuse_epochs_and_snapshots(with_gsan, expect):
    markers = torch.zeros(4, dtype=torch.int32, device="cuda")
    flags = torch.zeros(2, dtype=torch.int32, device="cuda")
    _gluon_mbarrier_repeated_phases_kernel[(1, )](markers, flags, 65536, expect, num_warps=4, num_ctas=2)
    torch.cuda.synchronize()

    clocks = [shadow_cell_from_address(markers.data_ptr() + i * markers.element_size()).write_clock for i in range(4)]
    assert clocks[0].thread_id != clocks[1].thread_id
    for cta in range(2):
        assert clocks[cta] == clocks[cta + 2]
        state = thread_state_from_smid(clocks[cta].thread_id)
        release = shadow_cell_from_address(flags.data_ptr() + cta * flags.element_size()).write_clock
        assert release.is_release
        assert release.thread_id == state.thread_id
        # Publishing once and importing the peer's completed epoch can create
        # a few snapshots, but the phase count must not control buffer usage.
        assert 0 <= state.clock_buffer_head - release.epoch <= 4
        assert state.vector_clock[state.thread_id] == clocks[cta].epoch
    consumer = thread_state_from_smid(clocks[0].thread_id)
    assert consumer.vector_clock[clocks[1].thread_id] == clocks[1].epoch - 1


@triton.jit
def _gsan_warm_all_sms_kernel(markers, counter, num_sms: tl.constexpr):
    smid = tl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r", [], tl.int32, is_pure=False, pack=1)
    tl.store(markers + smid, 1)
    tl.atomic_add(counter, 1, sem="relaxed", scope="gpu")
    # GSan reserves all shared memory, so these resident CTAs occupy distinct
    # SMs. Keep them alive until every SM has advanced its local epoch.
    atomic_poll(counter, num_sms)


@gluon.jit
def _gluon_mbarrier_transitive_clock_kernel(markers):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1], [2]])
    pairs01 = gl.allocate_shared_memory(gl.int64, [2], hopper.mbarrier.MBarrierLayout([[0], [1]]))
    pairs02 = gl.allocate_shared_memory(gl.int64, [2], hopper.mbarrier.MBarrierLayout([[1], [0]]))
    hopper.mbarrier.init(pairs01, count=1)
    hopper.mbarrier.init(pairs02, count=1)
    offsets = gl.arange(0, 512, layout)
    gl.store(markers + offsets // 128, 1, mask=offsets % 128 == 0)
    # CTA 2 first acquires CTA 3's completed epoch, then passes it to CTA 0.
    hopper.mbarrier.arrive(pairs01)
    hopper.mbarrier.wait(pairs01, 0)
    hopper.cluster.barrier(relaxed=True)
    hopper.mbarrier.arrive(pairs02)
    hopper.mbarrier.wait(pairs02, 0)
    hopper.cluster.barrier(relaxed=True)
    hopper.mbarrier.invalidate(pairs01)
    hopper.mbarrier.invalidate(pairs02)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
def test_gluon_mbarrier_preserves_transitively_imported_clocks(with_gsan):
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    warm = torch.zeros(num_sms, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    markers = torch.zeros(4, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _gsan_warm_all_sms_kernel[(num_sms, )](warm, counter, num_sms, num_warps=1)
    stream.synchronize()
    # A host wait does not import this independent stream's GSan clock.
    _gluon_mbarrier_transitive_clock_kernel[(1, )](markers, num_warps=4, num_ctas=4)
    torch.cuda.synchronize()

    clocks = [shadow_cell_from_address(markers.data_ptr() + i * markers.element_size()).write_clock for i in range(4)]
    assert len({clock.thread_id for clock in clocks}) == 4
    consumer = thread_state_from_smid(clocks[0].thread_id)
    for clock in clocks[1:]:
        smid = clock.thread_id % num_sms
        previous = shadow_cell_from_address(warm.data_ptr() + smid * warm.element_size()).write_clock
        assert previous.thread_id == clock.thread_id
        assert previous.epoch + 1 == clock.epoch
        assert consumer.vector_clock[clock.thread_id] == previous.epoch
    # The zero CTA-layout bases predicate the waits onto the pair leaders.
    # CTA 1 acquires CTA 3 in the second phase, but neither CTA 1 nor CTA 3
    # executes a wait that could acquire CTA 2's newly completed epoch.
    nonleader = thread_state_from_smid(clocks[1].thread_id)
    assert nonleader.vector_clock[clocks[3].thread_id] == clocks[3].epoch - 1
    for cta in (1, 3):
        state = thread_state_from_smid(clocks[cta].thread_id)
        assert state.vector_clock[clocks[2].thread_id] < clocks[2].epoch - 1


@triton.jit
def _gsan_empty_kernel(out_ptr):
    tl.store(out_ptr, 0)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
def test_gsan_uses_all_available_shared_memory(with_gsan):
    out = torch.empty(1, dtype=torch.int32, device="cuda")
    compiled = _gsan_empty_kernel.warmup(out, grid=(1, ))

    device = triton.runtime.driver.active.get_current_device()
    max_shared = triton.runtime.driver.active.utils.get_device_properties(device)["max_shared_mem"]
    assert compiled.metadata.min_shared_mem == max_shared
    assert compiled.metadata.shared == max_shared
    assert compiled.packed_metadata[2] == max_shared

    _gsan_empty_kernel[(1, )](out)
    torch.cuda.synchronize()
    assert out.item() == 0


@triton.jit
def atomic_add_kernel(ptr, sem: tl.constexpr, scope: tl.constexpr = "gpu"):
    tl.atomic_add(ptr, 1, sem=sem, scope=scope)


@triton.jit
def atomic_cas_kernel(ptr, out_ptr, expect, sem: tl.constexpr, scope: tl.constexpr = "gpu"):
    old = tl.atomic_cas(ptr, expect, 2, sem=sem, scope=scope)
    tl.store(out_ptr, old)


@triton.jit
def _scalar_atomic_rmw_cluster_kernel(ptr, out_ptr):
    old = tl.atomic_add(ptr, 1, sem="relaxed", scope="gpu")
    offsets = tl.arange(0, 32)
    tl.store(out_ptr + offsets, old)


@triton.jit
def _scalar_atomic_cas_cluster_kernel(ptr, out_ptr, expected, desired):
    old = tl.atomic_cas(ptr, expected, desired, sem="relaxed", scope="gpu")
    offsets = tl.arange(0, 32)
    tl.store(out_ptr + offsets, old)


def _assert_cluster_scalar_atomic_result(kernel):
    initial = 0x12345678
    target = torch.full((1, ), initial, dtype=torch.int32, device="cuda")
    out = torch.full((32, ), -1, dtype=torch.int32, device="cuda")

    if kernel is _scalar_atomic_rmw_cluster_kernel:
        kernel[(1, )](target, out, num_warps=1, num_ctas=2)
    else:
        kernel[(1, )](target, out, initial, initial + 1, num_warps=1, num_ctas=2)
    torch.cuda.synchronize()

    assert target.item() == initial + 1
    torch.testing.assert_close(out, torch.full_like(out, initial))


@pytest.mark.skipif(not is_hopper_or_newer() or is_sm12x(),
                    reason="scalar multi-CTA atomics require Hopper+ and are unsupported on sm12x")
def test_scalar_atomic_rmw_cluster_result_broadcast(with_gsan):
    _assert_cluster_scalar_atomic_result(_scalar_atomic_rmw_cluster_kernel)


@pytest.mark.skipif(not is_hopper_or_newer() or is_sm12x(),
                    reason="scalar multi-CTA atomics require Hopper+ and are unsupported on sm12x")
def test_scalar_atomic_cas_cluster_result_broadcast(with_gsan):
    _assert_cluster_scalar_atomic_result(_scalar_atomic_cas_cluster_kernel)


@triton.jit
def atomic_poll_kernel(ptr, expect, sem: tl.constexpr, scope: tl.constexpr = "gpu"):
    tl.atomic_poll(ptr, expect, sem=sem, scope=scope)


@triton.jit
def atomic_poll_timeout_kernel(ptr, out_ptr):
    matched = tl.atomic_poll(ptr, 1, timeout_ns=0)
    tl.store(out_ptr, matched)


@gluon.jit
def _atomic_poll_tensor_kernel(ptr, out_ptr, BLOCK: gl.constexpr, STRIDE: gl.constexpr, sem: gl.constexpr,
                               scope: gl.constexpr, TIMEOUT: gl.constexpr):
    offsets = gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [32], [4], [0]))
    matched = gl.atomic_poll(ptr + offsets * STRIDE, offsets + 1, sem=sem, scope=scope, timeout_ns=TIMEOUT)
    gl.store(out_ptr + offsets, matched)


@gluon.jit
def _atomic_poll_tensor_sync_kernel(payload_ptr, flag_ptr, out_ptr, BLOCK: gl.constexpr, STRIDE: gl.constexpr,
                                    scope: gl.constexpr):
    offsets = gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [32], [4], [0]))
    pid = gl.program_id(0)
    if pid < 2:
        mask = offsets % 2 == pid
        gl.store(payload_ptr + offsets * STRIDE, offsets + 1000, mask)
        gl.atomic_xchg(flag_ptr + offsets * STRIDE, 1, mask=mask, sem="release", scope=scope)
    else:
        gl.atomic_poll(flag_ptr + offsets * STRIDE, 1, sem="acquire", scope=scope)
        result = gl.load(payload_ptr + offsets * STRIDE)
        gl.store(out_ptr + offsets, result)


@triton.jit
def _cross_sm_atomic_sync_kernel(payload_ptr, flag_ptr, out_ptr, producer_sem: tl.constexpr, consumer_sem: tl.constexpr,
                                 scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem=producer_sem, scope=scope)
    elif pid == 1:
        atomic_poll(flag_ptr, 1, sem=consumer_sem, scope=scope)
        result = tl.load(payload_ptr)
        tl.store(out_ptr, result)


@triton.jit
def _cross_sm_atomic_poll_sync_kernel(payload_ptr, flag_ptr, out_ptr, scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem="release", scope=scope)
    elif pid == 1:
        tl.atomic_poll(flag_ptr, 1, sem="acquire", scope=scope)
        result = tl.load(payload_ptr)
        tl.store(out_ptr, result)


@triton.jit
def _transitive_atomic_sync_kernel(payload_ptr, flag0_ptr, flag1_ptr, out_ptr, release_sem: tl.constexpr,
                                   acquire_sem: tl.constexpr, scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag0_ptr, 1, sem=release_sem, scope=scope)
    elif pid == 1:
        atomic_poll(flag0_ptr, 1, sem=acquire_sem, scope=scope)
        tl.atomic_xchg(flag1_ptr, 1, sem=release_sem, scope=scope)
    elif pid == 2:
        atomic_poll(flag1_ptr, 1, sem=acquire_sem, scope=scope)
        result = tl.load(payload_ptr)
        tl.store(out_ptr, result)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES)
@pytest.mark.parametrize("sem, is_release", ATOMIC_SEMANTIC_CASES)
def test_atomic_add_updates_atomic_shadow(with_gsan, sem, is_release, scope, expected_scope):
    target = torch.zeros(1, dtype=torch.int32, device="cuda")

    atomic_add_kernel[(1, )](target, sem=sem, scope=scope, num_warps=1)
    assert target.item() == 1

    _assert_atomic_rmw_shadow(target.data_ptr(), expected_scope, is_release=is_release)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES)
@pytest.mark.parametrize("sem, _", ATOMIC_SEMANTIC_CASES)
def test_atomic_cas_failed_only_records_read(with_gsan, sem, _, scope, expected_scope):
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")

    atomic_cas_kernel[(1, )](target, out, expect=1, sem=sem, scope=scope, num_warps=1)

    assert target.item() == 0
    assert out.item() == 0

    _assert_atomic_read_only_shadow(target.data_ptr(), expected_scope)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES)
@pytest.mark.parametrize("sem", ["relaxed", "acquire"])
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_atomic_poll_only_records_read(with_gsan, dtype, sem, scope, expected_scope):
    target = torch.ones(1, dtype=dtype, device="cuda")

    atomic_poll_kernel[(1, )](target, 1, sem=sem, scope=scope, num_warps=4)

    _assert_atomic_read_only_shadow(target.data_ptr(), expected_scope)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
def test_atomic_poll_timeout_does_not_record_read(with_gsan):
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.ones(1, dtype=torch.bool, device="cuda")

    atomic_poll_timeout_kernel[(1, )](target, out, num_warps=4)

    assert not out.item()
    cell = shadow_cell_from_address(target.data_ptr())
    assert cell.write_clock == ScalarClock(0, 0, AtomicScope.NON_ATOMIC)
    assert cell.num_reads == 0


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
@pytest.mark.parametrize("block_size", [16, 256])
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize("sem", ["relaxed", "acquire"])
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES)
@pytest.mark.parametrize("timeout", [None, 0])
def test_atomic_poll_tensor_only_records_matched_reads(with_gsan, block_size, dtype, sem, scope, expected_scope,
                                                       timeout):
    # Separate shadow cells let us check successful and timed-out elements independently.
    stride = max(1, SHADOW_GRANULARITY_BYTES // torch.empty((), dtype=dtype).element_size())
    target = torch.zeros(block_size * stride, dtype=dtype, device="cuda")
    expected = torch.arange(1, block_size + 1, dtype=dtype, device="cuda")
    target[::stride] = expected
    if timeout is not None:
        target[::2 * stride] = 0
    out = torch.empty(block_size, dtype=torch.bool, device="cuda")

    _atomic_poll_tensor_kernel[(1, )](target, out, block_size, stride, sem, scope, timeout, num_warps=4)
    torch.testing.assert_close(out, target[::stride] == expected)

    for index in range(block_size):
        for byte_offset in range(0, target.element_size(), SHADOW_GRANULARITY_BYTES):
            address = target.data_ptr() + index * stride * target.element_size() + byte_offset
            if timeout is None or index % 2:
                _assert_atomic_read_only_shadow(address, expected_scope)
            else:
                cell = shadow_cell_from_address(address)
                assert cell.write_clock == ScalarClock(0, 0, AtomicScope.NON_ATOMIC)
                assert cell.num_reads == 0


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
@pytest.mark.parametrize("block_size", [16, 256])
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES[1:])
def test_atomic_poll_tensor_acquires_all_producers(with_gsan, capfd, block_size, scope, expected_scope):
    stride = SHADOW_GRANULARITY_BYTES // 4
    payload = torch.zeros(block_size * stride, dtype=torch.int32, device="cuda")
    flags = torch.zeros_like(payload)
    out = torch.full((block_size, ), -1, dtype=torch.int32, device="cuda")

    _atomic_poll_tensor_sync_kernel[(3, )](payload, flags, out, block_size, stride, scope, num_warps=4)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, torch.arange(1000, 1000 + block_size, dtype=torch.int32, device="cuda"))
    for index in range(block_size):
        _assert_cross_sm_sync(payload[index * stride:], flags[index * stride:], expected_scope)
    _assert_no_gsan_runtime_output(capfd)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES[1:])
def test_atomic_poll_acquire_synchronizes_cross_sm(with_gsan, capfd, scope, expected_scope):
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")

    _cross_sm_atomic_poll_sync_kernel[(2, )](payload, flag, out, scope=scope, num_warps=4)
    torch.cuda.synchronize()

    assert out.item() == 1000
    _assert_cross_sm_sync(payload, flag, expected_scope)
    _assert_no_gsan_runtime_output(capfd)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES)
@pytest.mark.parametrize("sem, is_release", ATOMIC_SEMANTIC_CASES)
def test_atomic_cas_success_updates_atomic_shadow(with_gsan, sem, is_release, scope, expected_scope):
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")

    atomic_cas_kernel[(1, )](target, out, expect=0, sem=sem, scope=scope, num_warps=1)

    assert target.item() == 2
    assert out.item() == 0

    _assert_atomic_rmw_shadow(target.data_ptr(), expected_scope, is_release=is_release)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES[1:])
@pytest.mark.parametrize("producer_sem", RELEASE_SEMANTIC_CASES)
@pytest.mark.parametrize("consumer_sem", ACQUIRE_SEMANTIC_CASES)
def test_atomic_release_acquire_synchronizes_cross_sm(with_gsan, capfd, producer_sem, consumer_sem, scope,
                                                      expected_scope):
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _cross_sm_atomic_sync_kernel[(2, )](
        payload,
        flags,
        out,
        producer_sem=producer_sem,
        consumer_sem=consumer_sem,
        scope=scope,
        num_warps=1,
    )
    torch.cuda.synchronize()

    assert out.item() == 1000

    _assert_cross_sm_sync(payload, flags, expected_scope)
    _assert_no_gsan_runtime_output(capfd)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES[1:])
@pytest.mark.parametrize("release_sem", RELEASE_SEMANTIC_CASES)
@pytest.mark.parametrize("acquire_sem", ACQUIRE_SEMANTIC_CASES)
def test_atomic_release_acquire_transitively_synchronizes_cross_sm(with_gsan, capfd, release_sem, acquire_sem, scope,
                                                                   expected_scope):
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag0 = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag1 = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _transitive_atomic_sync_kernel[(3, )](
        payload,
        flag0,
        flag1,
        out,
        release_sem=release_sem,
        acquire_sem=acquire_sem,
        scope=scope,
        num_warps=1,
    )
    torch.cuda.synchronize()

    assert out.item() == 1000

    payload_cell = shadow_cell_from_address(payload.data_ptr())
    flag1_cell = shadow_cell_from_address(flag1.data_ptr())
    producer_tid = payload_cell.write_clock.thread_id
    producer_epoch = payload_cell.write_clock.epoch

    relay_state = thread_state_from_smid(flag1_cell.write_clock.thread_id)
    snapshot_idx = _clock_buffer_snapshot_idx(flag1_cell.write_clock.epoch, relay_state, producer_tid)

    assert flag1_cell.write_clock.scope == expected_scope
    assert flag1_cell.write_clock.is_release
    assert relay_state.clock_buffer[snapshot_idx] >= producer_epoch

    consumer_tid = payload_cell.read_clocks[0].thread_id
    consumer_state = thread_state_from_smid(consumer_tid)

    assert consumer_state.vector_clock[producer_tid] >= producer_epoch

    _assert_no_gsan_runtime_output(capfd)


@triton.jit
def _release_rmw_chain_kernel(payload_ptr, counter_ptr, out_ptr, scope: tl.constexpr, NUM_WRITERS: tl.constexpr):
    pid = tl.program_id(0)
    if pid == NUM_WRITERS:
        atomic_poll(counter_ptr, NUM_WRITERS, sem="acquire", scope=scope)
        idx = tl.arange(0, triton.next_power_of_2(NUM_WRITERS))
        value = tl.load(payload_ptr + idx, mask=idx < NUM_WRITERS)
        tl.store(out_ptr + idx, value, mask=idx < NUM_WRITERS)
    else:
        tl.store(payload_ptr + pid, 1000 + pid)
        tl.atomic_add(counter_ptr, 1, sem="release", scope=scope)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES[1:])
def test_atomic_release_rmw_chain_synchronizes_all_writers(with_gsan, capfd, scope, expected_scope):
    num_writers = 3
    payload = torch.zeros(num_writers, dtype=torch.int32, device="cuda")

    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((num_writers, ), -1, dtype=torch.int32, device="cuda")
    _release_rmw_chain_kernel[(num_writers + 1, )](
        payload,
        counter,
        out,
        scope=scope,
        NUM_WRITERS=num_writers,
        num_warps=1,
    )
    torch.cuda.synchronize()

    expected = torch.arange(1000, 1000 + num_writers, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(out, expected)

    writer_tids = set()
    for index in range(num_writers):
        payload_cell = shadow_cell_from_address(payload[index].data_ptr())
        writer_tid = payload_cell.write_clock.thread_id
        writer_epoch = payload_cell.write_clock.epoch
        writer_tids.add(writer_tid)
        consumer_tid = payload_cell.read_clocks[0].thread_id
        consumer_state = thread_state_from_smid(consumer_tid)
        assert consumer_state.vector_clock[writer_tid] >= writer_epoch
    assert len(writer_tids) == num_writers

    counter_cell = shadow_cell_from_address(counter.data_ptr())
    assert counter_cell.write_clock.scope == expected_scope
    assert counter_cell.write_clock.is_release

    _assert_no_gsan_runtime_output(capfd)


@triton.jit
def _ordered_mixed_scope_release_rmw_kernel(payload_ptr, counter_ptr, ready_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_add(counter_ptr, 1, sem="release", scope="gpu")
        tl.atomic_xchg(ready_ptr, 1, sem="relaxed", scope="gpu")
    elif pid == 1:
        atomic_poll(ready_ptr, 1)
        tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="sys")


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
def test_ordered_mixed_scope_release_rmw_is_allowed(with_gsan, capfd):
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    ready = torch.zeros(1, dtype=torch.int32, device="cuda")
    _ordered_mixed_scope_release_rmw_kernel[(2, )](payload, counter, ready, num_warps=1)
    torch.cuda.synchronize()

    assert counter.item() == 2

    payload_cell = shadow_cell_from_address(payload.data_ptr())
    counter_cell = shadow_cell_from_address(counter.data_ptr())
    assert counter_cell.write_clock.scope == AtomicScope.SYSTEM
    assert counter_cell.write_clock.is_release

    counter_writer_state = thread_state_from_smid(counter_cell.write_clock.thread_id)
    snapshot_idx = _clock_buffer_snapshot_idx(counter_cell.write_clock.epoch, counter_writer_state,
                                              payload_cell.write_clock.thread_id)
    assert counter_writer_state.clock_buffer[snapshot_idx] >= payload_cell.write_clock.epoch

    _assert_no_gsan_runtime_output(capfd)


@triton.jit
def _write_blocks_kernel(ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(ptr + offsets, 1, mask=mask)


@triton.jit
def _read_reversed_blocks_kernel(ptr, scratch_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    src_pid = tl.num_programs(0) - 1 - pid
    src_offsets = src_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dst_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_offsets < n_elements
    value = tl.load(ptr + src_offsets, mask=mask)
    tl.store(scratch_ptr + dst_offsets, value, mask=mask)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
def test_implicit_stream_ordering(with_gsan):
    block_size = 128
    size = block_size * 1024
    target = torch.zeros(size, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(size, dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(size, block_size), )
    _write_blocks_kernel[grid](target, size, BLOCK_SIZE=block_size)
    _read_reversed_blocks_kernel[grid](target, scratch, size, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()

    assert scratch.sum().item() == size


@gluon.jit
def _gluon_async_copy_masked_kernel(out_ptr, in_ptr, n_elements, start_idx, BLOCK: gl.constexpr):
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    block_layout: gl.constexpr = gl.BlockedLayout([2], [32], [2], [0])
    smem = gl.allocate_shared_memory(in_ptr.dtype.element_ty, [BLOCK], smem_layout)

    offsets = start_idx + gl.arange(0, BLOCK, block_layout)
    mask = offsets < n_elements
    async_copy.async_load(smem, in_ptr + offsets, mask=mask)
    async_copy.commit_group()
    async_copy.wait_group(0)

    values = smem.load(block_layout)
    gl.store(out_ptr + offsets, values, mask=mask)


@triton.jit
def _device_tma_masked_store_kernel(ptr, m_size, n_size, row_idx, col_idx, stride_0, BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(ptr, [m_size, n_size], [stride_0, 1], [BLOCK, BLOCK])
    values = tl.full((BLOCK, BLOCK), 1, dtype=tl.int32)
    desc.store([row_idx, col_idx], values)


@triton.jit
def _device_tma_masked_load_kernel(out_ptr, ptr, m_size, n_size, row_idx, col_idx, stride_0, BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(ptr, [m_size, n_size], [stride_0, 1], [BLOCK, BLOCK])
    values = desc.load([row_idx, col_idx])
    offsets = tl.arange(0, BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    tl.store(out_ptr + offsets, values)


@triton.jit
def _host_tma_gather_kernel(out_ptr, out_stride_0, out_stride_1, desc, x_offsets_ptr, y_offset, BLOCK_X: tl.constexpr):
    BLOCK_Y: tl.constexpr = desc.block_shape[1]
    x_offsets = tl.load(x_offsets_ptr + tl.arange(0, BLOCK_X))
    out = desc.gather(x_offsets, y_offset)
    indices_x = tl.arange(0, BLOCK_X)[:, None] * out_stride_0
    indices_y = tl.arange(0, BLOCK_Y)[None, :] * out_stride_1
    tl.store(out_ptr + indices_x + indices_y, out)


@triton.jit
def _host_tma_scatter_kernel(desc, x_offsets_ptr, y_offset, src_ptr, src_stride_0, src_stride_1, BLOCK_X: tl.constexpr):
    BLOCK_Y: tl.constexpr = desc.block_shape[1]
    indices_x = tl.arange(0, BLOCK_X)[:, None] * src_stride_0
    indices_y = tl.arange(0, BLOCK_Y)[None, :] * src_stride_1
    src = tl.load(src_ptr + indices_x + indices_y)
    x_offsets = tl.load(x_offsets_ptr + tl.arange(0, BLOCK_X))
    desc.scatter(src, x_offsets, y_offset)


@triton.jit
def _host_tma_reduce_add_kernel(desc, src_ptr, src_stride_0, src_stride_1, BLOCK_X: tl.constexpr):
    BLOCK_Y: tl.constexpr = desc.block_shape[1]
    indices_x = tl.arange(0, BLOCK_X)[:, None] * src_stride_0
    indices_y = tl.arange(0, BLOCK_Y)[None, :] * src_stride_1
    src = tl.load(src_ptr + indices_x + indices_y)
    desc.atomic_add([0, 0], src)


def _shadow_cells_for_tensor(tensor: torch.Tensor):
    assert tensor.ndim >= 1
    if tensor.ndim > 1:
        return [_shadow_cells_for_tensor(tensor[i]) for i in range(tensor.shape[0])]

    device_idx = tensor.device.index
    row = []
    for i in range(tensor.shape[0]):
        real_ptr = tensor[i].data_ptr()
        assert real_ptr % SHADOW_GRANULARITY_BYTES == 0
        row.append(shadow_cell_from_address(real_ptr, device_index=device_idx))
    return row


def _assert_shadow_mask(before, after, changed_mask: torch.Tensor, *, access_kind: str) -> None:
    assert access_kind in {"read", "write"}
    assert len(before) == changed_mask.shape[0]
    assert len(before[0]) == changed_mask.shape[1]

    for row_idx in range(changed_mask.shape[0]):
        for col_idx in range(changed_mask.shape[1]):
            before_cell = before[row_idx][col_idx]
            after_cell = after[row_idx][col_idx]

            if changed_mask[row_idx, col_idx].item():
                assert after_cell != before_cell
                if access_kind == "read":
                    assert after_cell.write_clock == before_cell.write_clock
                else:
                    assert after_cell.write_clock != before_cell.write_clock
                    assert after_cell.write_clock.epoch != 0
            else:
                assert after_cell == before_cell


def _masked_tma_change_mask(storage: torch.Tensor, m_size: int, n_size: int, row_idx: int, col_idx: int,
                            block: int) -> torch.Tensor:
    changed_mask = torch.zeros(storage.shape, dtype=torch.bool)
    first_row = max(row_idx, 0)
    last_row = min(row_idx + block, m_size)
    first_col = max(col_idx, 0)
    last_col = min(col_idx + block, n_size)
    changed_mask[first_row:last_row, first_col:last_col] = True
    return changed_mask


def _gather_scatter_change_mask(storage: torch.Tensor, x_offsets: torch.Tensor, y_offset: int, m_size: int, n_size: int,
                                block_y: int) -> torch.Tensor:
    changed_mask = torch.zeros(storage.shape, dtype=torch.bool)
    valid_cols = max(min(n_size - y_offset, block_y), 0)
    if valid_cols == 0:
        return changed_mask

    for row_idx in x_offsets.tolist():
        if 0 <= row_idx < m_size:
            changed_mask[row_idx, y_offset:y_offset + valid_cols] = True
    return changed_mask


def _gather_reference(target: torch.Tensor, x_offsets: torch.Tensor, y_offset: int, block_y: int) -> torch.Tensor:
    result = torch.zeros((x_offsets.numel(), block_y), dtype=target.dtype, device=target.device)
    valid_rows = (x_offsets >= 0) & (x_offsets < target.shape[0])
    valid_cols = max(min(target.shape[1] - y_offset, block_y), 0)
    if valid_cols == 0:
        return result

    safe_rows = torch.where(valid_rows, x_offsets, 0)
    gathered = target[safe_rows.long(), y_offset:y_offset + valid_cols]
    result[:, :valid_cols] = gathered * valid_rows[:, None]
    return result


def _scatter_reference(dst: torch.Tensor, src: torch.Tensor, x_offsets: torch.Tensor, y_offset: int) -> torch.Tensor:
    result = torch.zeros_like(dst)
    valid_cols = max(min(dst.shape[1] - y_offset, src.shape[1]), 0)
    if valid_cols == 0:
        return result

    for src_row, dst_row in enumerate(x_offsets.tolist()):
        if 0 <= dst_row < dst.shape[0]:
            result[dst_row, y_offset:y_offset + valid_cols] = src[src_row, :valid_cols]
    return result


@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
def test_gluon_async_copy_updates_shadow(with_gsan):
    block = 128
    start_idx = 5
    n_elements = 117
    padded = 160
    inp = torch.arange(padded, dtype=torch.float32, device="cuda")
    out = torch.zeros_like(inp)
    shadow0 = [_shadow_cells_for_tensor(inp)]
    changed_mask = torch.zeros((1, inp.numel()), dtype=torch.bool)
    changed_mask[0, start_idx:n_elements] = True

    _gluon_async_copy_masked_kernel[(1, )](out, inp, n_elements, start_idx, BLOCK=block, num_warps=2)

    expected = torch.zeros_like(out)
    expected[start_idx:n_elements] = inp[start_idx:n_elements]
    torch.testing.assert_close(out, expected)

    shadow1 = [_shadow_cells_for_tensor(inp)]
    _assert_shadow_mask(shadow0, shadow1, changed_mask, access_kind="read")
    assert out[n_elements].item() == 0
    assert n_elements - start_idx < block


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
@pytest.mark.parametrize("num_ctas", [
    1,
    pytest.param(
        2, marks=pytest.mark.skipif(not is_hopper_or_newer() or is_sm12x(),
                                    reason="Multi-CTA TMA requires Hopper or Blackwell")),
])
@pytest.mark.parametrize("row_idx,col_idx", [(5, 8), (30, 8), (5, 32)])
def test_tma_masked_load_updates_shadow(with_gsan, with_allocator, row_idx, col_idx, num_ctas):
    block = 32
    m_size = 35
    n_size = 37
    padded_m = 40
    padded_n = 40
    first_row = max(row_idx, 0)
    last_row = min(row_idx + block, m_size)
    first_col = max(col_idx, 0)
    last_col = min(col_idx + block, n_size)
    target_storage = torch.arange(padded_m * padded_n, dtype=torch.int32, device="cuda").reshape(padded_m, padded_n)
    target = target_storage[:m_size, :n_size]
    output = torch.empty((block, block), dtype=torch.int32, device="cuda")
    shadow0 = _shadow_cells_for_tensor(target_storage)
    changed_mask = _masked_tma_change_mask(target_storage, m_size, n_size, row_idx, col_idx, block)

    _device_tma_masked_load_kernel[(1, )](output, target, m_size, n_size, row_idx, col_idx, target.stride(0),
                                          BLOCK=block, num_ctas=num_ctas)
    torch.cuda.synchronize()

    expected = torch.zeros_like(output)
    expected[:last_row - first_row, :last_col - first_col] = target[first_row:last_row, first_col:last_col]
    torch.testing.assert_close(output, expected)

    shadow1 = _shadow_cells_for_tensor(target_storage)
    _assert_shadow_mask(shadow0, shadow1, changed_mask, access_kind="read")


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
@pytest.mark.parametrize("num_ctas", [
    1,
    pytest.param(
        2, marks=pytest.mark.skipif(not is_hopper_or_newer() or is_sm12x(),
                                    reason="Multi-CTA TMA requires Hopper or Blackwell")),
])
@pytest.mark.parametrize("row_idx,col_idx", [(5, 8), (30, 8), (5, 32)])
def test_tma_masked_store_updates_shadow(with_gsan, with_allocator, row_idx, col_idx, num_ctas):
    block = 32
    m_size = 35
    n_size = 37
    padded_m = 40
    padded_n = 40
    first_row = max(row_idx, 0)
    last_row = min(row_idx + block, m_size)
    first_col = max(col_idx, 0)
    last_col = min(col_idx + block, n_size)
    valid_rows = max(last_row - first_row, 0)
    valid_cols = max(last_col - first_col, 0)
    target_storage = torch.zeros((padded_m, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:m_size, :n_size]
    shadow0 = _shadow_cells_for_tensor(target_storage)
    changed_mask = _masked_tma_change_mask(target_storage, m_size, n_size, row_idx, col_idx, block)

    _device_tma_masked_store_kernel[(1, )](target, m_size, n_size, row_idx, col_idx, target.stride(0), BLOCK=block,
                                           num_ctas=num_ctas)
    torch.cuda.synchronize()

    expected = torch.zeros_like(target)
    expected[first_row:last_row, first_col:last_col] = 1
    torch.testing.assert_close(target, expected)

    shadow1 = _shadow_cells_for_tensor(target_storage)
    _assert_shadow_mask(shadow0, shadow1, changed_mask, access_kind="write")
    assert target_storage[m_size, col_idx].item() == 0
    assert valid_rows < block
    assert valid_cols < block


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_host_tma_gather_updates_shadow(with_gsan):
    block_x = 8
    block_y = 8
    m_size = 11
    n_size = 13
    padded_m = 16
    padded_n = 16
    y_offset = 8
    x_offsets = torch.tensor([1, 3, 5, 7, 9, 10, 11, 13], dtype=torch.int32, device="cuda")
    target_storage = torch.arange(padded_m * padded_n, dtype=torch.int32, device="cuda").reshape(padded_m, padded_n)
    target = target_storage[:m_size, :n_size]
    target_desc = TensorDescriptor.from_tensor(target, [1, block_y])
    out = torch.empty((block_x, block_y), dtype=torch.int32, device="cuda")
    shadow0 = _shadow_cells_for_tensor(target_storage)
    changed_mask = _gather_scatter_change_mask(target_storage, x_offsets, y_offset, m_size, n_size, block_y)

    compiled = _host_tma_gather_kernel[(1, )](out, out.stride(0), out.stride(1), target_desc, x_offsets, y_offset,
                                              BLOCK_X=block_x)
    assert "ttng.async_tma_gather" in compiled.asm["ttgir"]
    torch.cuda.synchronize()

    torch.testing.assert_close(out, _gather_reference(target, x_offsets, y_offset, block_y))

    shadow1 = _shadow_cells_for_tensor(target_storage)
    _assert_shadow_mask(shadow0, shadow1, changed_mask, access_kind="read")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_host_tma_scatter_updates_shadow(with_gsan):
    block_x = 8
    block_y = 8
    m_size = 11
    n_size = 13
    padded_m = 16
    padded_n = 16
    y_offset = 8
    x_offsets = torch.tensor([1, 3, 5, 7, 9, 10, 11, 13], dtype=torch.int32, device="cuda")
    target_storage = torch.zeros((padded_m, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:m_size, :n_size]
    target_desc = TensorDescriptor.from_tensor(target, [1, block_y])
    src = torch.arange(1, block_x * block_y + 1, dtype=torch.int32, device="cuda").reshape(block_x, block_y)
    shadow0 = _shadow_cells_for_tensor(target_storage)
    changed_mask = _gather_scatter_change_mask(target_storage, x_offsets, y_offset, m_size, n_size, block_y)

    compiled = _host_tma_scatter_kernel[(1, )](target_desc, x_offsets, y_offset, src, src.stride(0), src.stride(1),
                                               BLOCK_X=block_x)
    assert "ttng.async_tma_scatter" in compiled.asm["ttgir"]
    torch.cuda.synchronize()

    torch.testing.assert_close(target, _scatter_reference(target, src, x_offsets, y_offset))

    shadow1 = _shadow_cells_for_tensor(target_storage)
    _assert_shadow_mask(shadow0, shadow1, changed_mask, access_kind="write")
    assert target_storage[m_size, y_offset].item() == 0


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires Hopper or newer")
@pytest.mark.parametrize("dtype", (torch.int32, torch.float16, torch.bfloat16, torch.uint64))
@pytest.mark.parametrize("block_x", (1, 8))
def test_host_tma_reduce_updates_atomic_shadow(with_gsan, block_x, dtype):
    block_y = 16
    target = torch.zeros((block_x, block_y), dtype=dtype, device="cuda")
    src = torch.arange(1, block_x * block_y + 1, dtype=torch.int32, device="cuda").to(dtype).reshape(block_x, block_y)
    target_desc = TensorDescriptor.from_tensor(target, [block_x, block_y])

    compiled = _host_tma_reduce_add_kernel[(1, )](target_desc, src, src.stride(0), src.stride(1), BLOCK_X=block_x)
    assert "ttng.async_tma_reduce" in compiled.asm["ttgir"]
    torch.cuda.synchronize()

    torch.testing.assert_close(target, src)
    for row in range(block_x):
        for col in range(block_y):
            for byte_offset in range(0, target.element_size(), SHADOW_GRANULARITY_BYTES):
                address = target[row, col].data_ptr() + byte_offset
                _assert_atomic_rmw_shadow(address, AtomicScope.GPU, is_release=False)
