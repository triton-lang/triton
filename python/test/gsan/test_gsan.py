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
from triton.experimental.gsan._testing_utils import (load_one_i32, shadow_cell_from_address, store_one_i32,
                                                     thread_state_from_smid)


@pytest.fixture()
def with_gsan(fresh_knobs):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        yield


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
