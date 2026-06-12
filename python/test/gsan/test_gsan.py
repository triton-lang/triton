from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.tools.tensor_descriptor import TensorDescriptor

from triton._internal_testing import is_blackwell, is_cuda, is_ampere_or_newer
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


@gluon.jit
def _gluon_ws_completion_default(out_ptr, layout: gl.constexpr):
    offsets = gl.arange(0, 128, layout=layout)
    gl.store(out_ptr + offsets, offsets)


@gluon.jit
def _gluon_ws_completion_worker(out_ptr, layout: gl.constexpr):
    offsets = 128 + gl.arange(0, 128, layout=layout)
    gl.store(out_ptr + offsets, offsets)


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


@triton.jit
def atomic_add_kernel(ptr, sem: tl.constexpr, scope: tl.constexpr = "gpu"):
    tl.atomic_add(ptr, 1, sem=sem, scope=scope)


@triton.jit
def atomic_cas_kernel(ptr, out_ptr, expect, sem: tl.constexpr, scope: tl.constexpr = "gpu"):
    old = tl.atomic_cas(ptr, expect, 2, sem=sem, scope=scope)
    tl.store(out_ptr, old)


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
    async_copy.async_copy_global_to_shared(smem, in_ptr + offsets, mask=mask)
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


def _shadow_cell_state(cell) -> tuple[int, object, tuple[object, ...]]:
    return (cell.num_reads, cell.write_clock, tuple(cell.read_clocks))


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
            before_state = _shadow_cell_state(before_cell)
            after_state = _shadow_cell_state(after_cell)

            if changed_mask[row_idx, col_idx].item():
                assert after_state != before_state
                if access_kind == "read":
                    assert after_cell.write_clock == before_cell.write_clock
                else:
                    assert after_cell.write_clock != before_cell.write_clock
                    assert after_cell.write_clock.epoch != 0
            else:
                assert after_state == before_state


def _masked_store_change_mask(storage: torch.Tensor, m_size: int, n_size: int, row_idx: int,
                              col_idx: int) -> torch.Tensor:
    changed_mask = torch.zeros(storage.shape, dtype=torch.bool)
    changed_mask[row_idx:m_size, col_idx:n_size] = True
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
def test_tma_masked_store_updates_shadow(with_gsan, with_allocator):
    block = 32
    m_size = 35
    n_size = 37
    padded_m = 40
    padded_n = 40
    row_idx = 5
    col_idx = 8
    valid_rows = m_size - row_idx
    valid_cols = n_size - col_idx
    target_storage = torch.zeros((padded_m, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:m_size, :n_size]
    shadow0 = _shadow_cells_for_tensor(target_storage)
    changed_mask = _masked_store_change_mask(target_storage, m_size, n_size, row_idx, col_idx)

    _device_tma_masked_store_kernel[(1, )](target, m_size, n_size, row_idx, col_idx, target.stride(0), BLOCK=block)
    torch.cuda.synchronize()

    expected = torch.zeros_like(target)
    expected[row_idx:, col_idx:] = 1
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
def test_host_tma_reduce_updates_atomic_shadow(with_gsan):
    block_x = 1
    block_y = 16
    target = torch.zeros((block_x, block_y), dtype=torch.int32, device="cuda")
    src = torch.arange(1, block_y + 1, dtype=torch.int32, device="cuda").reshape(block_x, block_y)
    target_desc = TensorDescriptor.from_tensor(target, [block_x, block_y])

    compiled = _host_tma_reduce_add_kernel[(1, )](target_desc, src, src.stride(0), src.stride(1), BLOCK_X=block_x)
    assert "ttng.async_tma_reduce" in compiled.asm["ttgir"]
    torch.cuda.synchronize()

    torch.testing.assert_close(target, src)
    _assert_atomic_rmw_shadow(target[0, 0].data_ptr(), AtomicScope.GPU, is_release=False)
